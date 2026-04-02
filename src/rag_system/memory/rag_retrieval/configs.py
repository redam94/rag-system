"""
NodeConfig definitions for RAG retrieval workflow nodes.

Migrates plan_queries, rewrite_query, and synthesize from hand-written
async functions to declarative NodeConfig instances processed by NodeFactory.
"""

from typing import Dict, List, Any

from loguru import logger

from ...ai.factory import NodeConfig
from ...ai.llm_provider import get_llm as provider_get_llm, ProviderConfig
from ...ai.prompts import (
    RETRIEVAL_PLAN_QUERIES,
    RETRIEVAL_REWRITE_QUERY,
    RETRIEVAL_SYNTHESIZE,
    RETRIEVAL_PLAN_QUERIES_TEMPLATE,
    RETRIEVAL_REWRITE_QUERY_TEMPLATE,
    RETRIEVAL_SYNTHESIZE_TEMPLATE,
)
from .state import RetrievalConfig, QueryTask
from .models import QueryPlan, RewrittenQuery, SynthesisResult


# =============================================================================
# SHARED: LLM provider for retrieval nodes
# =============================================================================

def _retrieval_llm_provider(deps: Dict):
    """Build LLM from retrieval RetrievalConfig in deps."""
    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    provider_config = ProviderConfig(
        provider=getattr(config, "provider", "ollama"),
        model=config.llm_model,
        base_url=config.base_url,
        api_key=getattr(config, "api_key", ""),
    )
    return provider_get_llm(provider_config)


# =============================================================================
# PLAN QUERIES
# =============================================================================

def _plan_queries_vars(state: Dict, config: NodeConfig, deps: Dict) -> Dict[str, str]:
    return {
        "original_query": state.get("original_query", ""),
    }


def _plan_queries_post(output: Dict, state: Dict, deps: Dict) -> Dict:
    original_query = state.get("original_query", "")

    if "error" in output:
        logger.error(f"Query planning failed: {output['error']}")
        return {
            "query_todo": [QueryTask(query=original_query, query_type="specific", priority=2)],
            "iteration": state.get("iteration", 0) + 1,
        }

    todo: List[QueryTask] = []

    for q in output.get("specific_queries", []):
        todo.append(QueryTask(query=q, query_type="specific", priority=2))

    for q in output.get("broad_queries", []):
        todo.append(QueryTask(query=q, query_type="broad", priority=1))

    logger.info(f"📋 Planned {len(todo)} queries: {[t.query for t in todo]}")

    return {
        "query_todo": todo,
        "iteration": state.get("iteration", 0) + 1,
    }


PLAN_QUERIES_CONFIG = NodeConfig(
    name="plan_queries",
    system_prompt=RETRIEVAL_PLAN_QUERIES,
    output_schema=QueryPlan,
    user_prompt_template=RETRIEVAL_PLAN_QUERIES_TEMPLATE,
    prompt_vars_extractor=_plan_queries_vars,
    llm_provider=_retrieval_llm_provider,
    post_process=_plan_queries_post,
)


# =============================================================================
# REWRITE QUERY
# =============================================================================

def _rewrite_pre(state: Dict, deps: Dict) -> Dict:
    todo = state.get("query_todo", [])
    if not todo:
        state["_skip_llm"] = {"should_stop": True, "error": "No queries to process"}
        return state

    todo = sorted(todo, key=lambda x: -x.priority)
    state["_current"] = todo[0]
    state["_remaining"] = todo[1:]

    # Fetch RAG stats for doc type hints
    rag = deps.get("rag")
    state["_doc_types_hint"] = ""
    if rag and hasattr(rag, "get_stats"):
        try:
            stats = rag.get_stats()
            types = stats.get("type_breakdown", {})
            if types:
                state["_doc_types_hint"] = f"\nDocument types in RAG: {', '.join(types.keys())}"
        except Exception:
            pass

    return state


def _rewrite_vars(state: Dict, config: NodeConfig, deps: Dict) -> Dict[str, str]:
    current = state.get("_current")
    return {
        "current_query": current.query if current else "",
        "query_type": current.query_type if current else "",
        "doc_types_hint": state.get("_doc_types_hint", ""),
    }


def _rewrite_post(output: Dict, state: Dict, deps: Dict) -> Dict:
    current = state.get("_current")
    remaining = state.get("_remaining", [])

    if "error" in output:
        logger.error(f"Query rewriting failed: {output['error']}")
        return {
            "current_query": current,
            "rewritten_query": current.query if current else "",
            "query_todo": remaining,
        }

    rewritten = output.get("rewritten", current.query if current else "")
    logger.info(f"✏️ Rewritten: '{current.query}' → '{rewritten}'")

    return {
        "current_query": current,
        "rewritten_query": rewritten,
        "query_todo": remaining,
    }


REWRITE_QUERY_CONFIG = NodeConfig(
    name="rewrite_query",
    system_prompt=RETRIEVAL_REWRITE_QUERY,
    output_schema=RewrittenQuery,
    user_prompt_template=RETRIEVAL_REWRITE_QUERY_TEMPLATE,
    pre_process=_rewrite_pre,
    prompt_vars_extractor=_rewrite_vars,
    llm_provider=_retrieval_llm_provider,
    post_process=_rewrite_post,
)


# =============================================================================
# SYNTHESIZE
# =============================================================================

def _synthesize_pre(state: Dict, deps: Dict) -> Dict:
    relevant = state.get("relevant_chunks", [])

    if not relevant:
        state["_skip_llm"] = {
            "synthesized_context": "",
            "key_findings": [],
            "should_stop": True,
        }
        return state

    config: RetrievalConfig = deps.get("config", RetrievalConfig())

    # Group by document type
    by_type: Dict[str, list] = {}
    for chunk in relevant:
        doc_type = chunk.source_type
        by_type.setdefault(doc_type, []).append(chunk)

    # Format grouped context
    grouped_text = ""
    for doc_type, chunks in by_type.items():
        grouped_text += f"\n\n## {doc_type.upper()} SOURCES:\n"
        for c in chunks[:5]:
            grouped_text += f"\n{c.content[:800]}\n---"

    if len(grouped_text) > config.max_context_chars:
        grouped_text = grouped_text[: config.max_context_chars] + "\n[truncated]"

    state["_grouped_text"] = grouped_text
    state["_by_type_keys"] = list(by_type.keys())
    return state


def _synthesize_vars(state: Dict, config: NodeConfig, deps: Dict) -> Dict[str, str]:
    return {
        "original_query": state.get("original_query", ""),
        "grouped_text": state.get("_grouped_text", ""),
    }


def _synthesize_post(output: Dict, state: Dict, deps: Dict) -> Dict:
    if "error" in output:
        logger.error(f"Synthesis failed: {output['error']}")
        relevant = state.get("relevant_chunks", [])
        config: RetrievalConfig = deps.get("config", RetrievalConfig())
        fallback_context = "\n\n".join([c.content for c in relevant[:10]])
        return {
            "synthesized_context": fallback_context[: config.max_context_chars],
            "key_findings": [],
            "document_types_found": state.get("_by_type_keys", []),
            "should_stop": True,
        }

    return {
        "synthesized_context": output.get("combined_context", ""),
        "key_findings": output.get("key_findings", []),
        "document_types_found": output.get("document_types_found", []),
        "should_stop": True,
    }


SYNTHESIZE_CONFIG = NodeConfig(
    name="synthesize",
    system_prompt=RETRIEVAL_SYNTHESIZE,
    output_schema=SynthesisResult,
    user_prompt_template=RETRIEVAL_SYNTHESIZE_TEMPLATE,
    pre_process=_synthesize_pre,
    prompt_vars_extractor=_synthesize_vars,
    llm_provider=_retrieval_llm_provider,
    post_process=_synthesize_post,
)
