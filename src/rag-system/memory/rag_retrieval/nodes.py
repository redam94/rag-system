"""
Node functions for RAG retrieval workflow.

Nodes:
- plan_queries: Generate initial specific + broad queries (factory-generated)
- rewrite_query: Transform query to match RAG semantics (factory-generated)
- execute_query: Run query against RAG with recency scoring (hand-written)
- grade_context: Grade relevance and extract subtopics (hand-written)
- synthesize: Combine all relevant context (factory-generated)
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from loguru import logger

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from langgraph.types import RuntimeValue

from .state import (
    RetrievalState,
    RetrievalConfig,
    QueryTask,
    RetrievedChunk,
    Deps,
)
from .models import ContextGrade
from .configs import PLAN_QUERIES_CONFIG, REWRITE_QUERY_CONFIG, SYNTHESIZE_CONFIG
from ...ai.prompts import RETRIEVAL_GRADE_CONTEXT, RETRIEVAL_GRADE_TEMPLATE
from ...ai.factory import NodeFactory


# =============================================================================
# HELPERS
# =============================================================================

def get_deps(runtime: Runtime[Deps]) -> Deps:
    """Extract deps from runtime context."""
    return runtime.context if hasattr(runtime, 'context') else {}


def get_llm(deps: Deps) -> BaseChatModel:
    """Get configured LLM from dependencies via central provider."""
    from ...ai.llm_provider import get_llm as provider_get_llm, ProviderConfig

    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    provider_config = ProviderConfig(
        provider=getattr(config, "provider", "ollama"),
        model=config.llm_model,
        base_url=config.base_url,
        api_key=getattr(config, "api_key", ""),
    )
    return provider_get_llm(provider_config)


def get_rag(deps: Deps):
    """Get RAG instance from dependencies."""
    return deps.get("rag")


def emit_progress(deps: Deps, message: str):
    """Emit progress if callback available."""
    callback = deps.get("on_progress")
    if callback:
        callback(message)
    logger.debug(f"RAG Retrieval: {message}")


def _calculate_recency_score(metadata: Dict[str, Any], decay_days: int = 30) -> float:
    """Calculate recency score from metadata timestamp.

    Returns 1.0 for brand-new documents, decaying toward 0.0 for older ones.
    Returns 0.5 for documents with no timestamp.
    """
    timestamp_str = metadata.get("timestamp")
    if not timestamp_str:
        return 0.5

    try:
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        else:
            timestamp = timestamp_str

        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
        age_days = (now - timestamp).total_seconds() / 86400

        return max(0.0, min(1.0, 1 - (age_days / (decay_days * 2))))
    except Exception as e:
        logger.debug(f"Could not parse timestamp: {e}")
        return 0.5


def _broaden_queries(original_query: str, used_queries: List[str]) -> List[QueryTask]:
    """Generate broader query tasks when no results found.

    Splits the query into sub-phrases and significant individual words.
    """
    words = original_query.split()
    tasks: List[QueryTask] = []

    if len(words) > 2:
        mid = len(words) // 2
        tasks.append(QueryTask(
            query=" ".join(words[:mid]),
            query_type="broadened",
            priority=1,
        ))
        tasks.append(QueryTask(
            query=" ".join(words[mid:]),
            query_type="broadened",
            priority=1,
        ))

    for word in words:
        if len(word) > 4 and word.lower() not in {q.lower() for q in used_queries}:
            tasks.append(QueryTask(query=word, query_type="broadened", priority=0))

    return tasks[:3]


# =============================================================================
# FACTORY-GENERATED NODES
# =============================================================================

_factory = NodeFactory()

plan_queries = _factory.create_node(PLAN_QUERIES_CONFIG)
rewrite_query = _factory.create_node(REWRITE_QUERY_CONFIG)
synthesize = _factory.create_node(SYNTHESIZE_CONFIG)


# =============================================================================
# NODE: EXECUTE QUERY (hand-written — RAG retrieval + recency scoring)
# =============================================================================

async def execute_query(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Execute the rewritten query against RAG.

    Scores each chunk with a weighted combination of embedding relevance
    and recency so fresher context ranks higher.
    """
    deps = get_deps(runtime)
    rewritten = state.get("rewritten_query")
    current = state.get("current_query")
    todo = state.get("query_todo", [])
    queries: List[str] = [current.query]
    if todo:
        todo = sorted(todo, key=lambda x: -x.priority)
        queries = [q.query for q in todo[-3:]]
        state["query_todo"] = todo[:-3]

    if not rewritten or not current:
        return {"error": "No query to execute"}

    emit_progress(deps, f"🔍 Searching RAG: {rewritten}")

    rag = get_rag(deps)
    if not rag:
        return {"error": "RAG not available"}

    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    workflow_id = state.get("workflow_id")

    try:
        # Query RAG
        results = []
        for q in queries:
            results.extend(rag.query_relevant_context(
                query=q,
                n_results=config.max_chunks_per_query,
                workflow_id=workflow_id,
            ))

        # Convert to RetrievedChunk objects with scoring
        chunks: List[RetrievedChunk] = []

        if results:
            docs = [res.get('document') for res in results if res.get('document')]
            metas = [res.get('metadata', {}) for res in results if res.get('document')]
            distances = [res.get('distance', 1.0) for res in results if res.get('document')]

            for doc, meta, dist in zip(docs, metas, distances):
                # Relevance: convert embedding distance to 0-1
                relevance = max(0.0, min(1.0, 1 - (dist / 1.5)))

                # Recency: time-based decay
                recency = _calculate_recency_score(meta, config.recency_decay_days)

                # Combined weighted score
                combined = (
                    config.relevance_weight * relevance
                    + config.recency_weight * recency
                )

                chunk = RetrievedChunk(
                    content=doc,
                    source_type=meta.get("type", "unknown"),
                    metadata=meta,
                    relevance_score=relevance,
                    recency_score=recency,
                    combined_score=combined,
                    query_used=rewritten,
                )
                chunks.append(chunk)

        # Dedupe by content
        existing = state.get("retrieved_chunks", [])
        existing_content = {c.content for c in existing}

        new_chunks = [c for c in chunks if c.content not in existing_content]
        all_chunks = existing + new_chunks

        # Track completed queries
        completed = state.get("completed_queries", [])
        for q in queries:
            if q not in completed:
                completed = completed + [q]

        emit_progress(deps, f"🔍 Found {len(new_chunks)} new chunks ({len(chunks)} total)")

        return {
            "retrieved_chunks": all_chunks,
            "completed_queries": completed,
        }

    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return {"error": f"Query failed: {e}"}


# =============================================================================
# NODE: GRADE CONTEXT (hand-written — LLM grading + state management)
# =============================================================================

async def grade_context(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Grade retrieved context for relevance.
    Extract subtopics for further exploration.
    """
    deps = get_deps(runtime)
    retrieved = state.get("retrieved_chunks", [])
    original_query = state["original_query"]

    if not retrieved:
        emit_progress(deps, "⚠️ No chunks to grade")
        return {
            "last_grade_score": 0.0,
            "consecutive_low_relevance": state.get("consecutive_low_relevance", 0) + 1,
        }

    # Get chunks from current query that haven't been graded yet
    relevant_existing = state.get("relevant_chunks", [])
    relevant_content = {c.content for c in relevant_existing}

    current_query = state.get("rewritten_query", "")
    chunks_to_grade = [
        c for c in retrieved
        if c.query_used == current_query and c.content not in relevant_content
    ]

    if not chunks_to_grade:
        return {
            "last_grade_score": 0.0,
            "consecutive_low_relevance": state.get("consecutive_low_relevance", 0) + 1,
        }

    emit_progress(deps, f"📊 Grading {len(chunks_to_grade)} chunks...")

    llm = get_llm(deps).with_structured_output(ContextGrade)
    config: RetrievalConfig = deps.get("config", RetrievalConfig())

    # Format chunks for grading (sorted by combined score)
    sorted_chunks = sorted(chunks_to_grade, key=lambda c: c.combined_score, reverse=True)
    chunks_text = "\n\n---\n\n".join([
        f"[{c.source_type}] (relevance: {c.relevance_score:.2f}, recency: {c.recency_score:.2f})\n{c.content}"
        for c in sorted_chunks
    ])

    existing_context = chr(10).join([f"- {c.content[:100]}..." for c in relevant_existing[-5:]])
    prompt = RETRIEVAL_GRADE_TEMPLATE.format(
        original_query=original_query,
        current_query=current_query,
        existing_context=existing_context,
        chunks_text=chunks_text,
    )

    try:
        result = llm.invoke([
            SystemMessage(content=RETRIEVAL_GRADE_CONTEXT),
            HumanMessage(content=prompt),
        ])

        # Filter chunks by combined score threshold
        new_relevant = [c for c in chunks_to_grade if c.combined_score >= 0.3]

        all_relevant = relevant_existing + new_relevant

        # Track subtopics
        discovered = state.get("discovered_subtopics", [])
        explored = state.get("explored_subtopics", [])

        new_subtopics = [
            s for s in result.extracted_subtopics
            if s not in discovered and s not in explored
        ]
        all_discovered = discovered + new_subtopics

        # Add subtopic queries to todo if we should continue
        todo = state.get("query_todo", [])
        if result.should_continue and new_subtopics:
            for subtopic in new_subtopics[:config.max_subtopics_to_explore]:
                todo.append(QueryTask(
                    query=subtopic,
                    query_type="subtopic",
                    priority=1,
                    parent_query=current_query,
                ))

        # Track consecutive low relevance
        consecutive_low = state.get("consecutive_low_relevance", 0)
        if result.relevance_score < config.relevance_threshold:
            consecutive_low += 1
        else:
            consecutive_low = 0

        emit_progress(
            deps,
            f"📊 Grade: {result.relevance_score:.2f}, "
            f"subtopics: {new_subtopics}, "
            f"relevant chunks: {len(new_relevant)}"
        )

        return {
            "relevant_chunks": all_relevant,
            "last_grade_score": result.relevance_score,
            "discovered_subtopics": all_discovered,
            "query_todo": todo,
            "consecutive_low_relevance": consecutive_low,
            "total_relevant_found": len(all_relevant),
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"Context grading failed: {e}")
        return {
            "last_grade_score": 0.5,
            "consecutive_low_relevance": state.get("consecutive_low_relevance", 0),
            "iteration": state.get("iteration", 0) + 1,
        }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_continue(state: RetrievalState) -> str:
    """
    Determine next action after grading.

    Returns:
        "rewrite_query" - More queries to process
        "synthesize" - Done searching, combine results
    """
    config_max = state.get("max_iterations", 10)
    iteration = state.get("iteration", 0)
    todo = state.get("query_todo", [])
    consecutive_low = state.get("consecutive_low_relevance", 0)
    total_relevant = state.get("total_relevant_found", 0)

    # Stop conditions
    if state.get("should_stop"):
        return "synthesize"

    if iteration >= config_max:
        logger.info(f"Max iterations ({config_max}) reached")
        return "synthesize"

    if consecutive_low >= 3:
        logger.info("Too many low-relevance results, stopping")
        return "synthesize"

    if not todo and total_relevant > 0:
        logger.info("No more queries and have relevant results")
        return "synthesize"

    if todo:
        return "rewrite_query"

    # No todo and no relevant results — broaden and retry
    if total_relevant == 0 and iteration < 3:
        original_query = state.get("original_query", "")
        completed = state.get("completed_queries", [])
        broadened = _broaden_queries(original_query, completed)
        if broadened:
            # Inject broadened queries into state for next rewrite_query pass
            state["query_todo"] = broadened
            return "rewrite_query"

    return "synthesize"
