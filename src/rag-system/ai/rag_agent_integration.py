"""
RAG Agent Integration with Emitter.

Bridges the RAG retrieval LangGraph workflow with the main analysis
workflow, handling emitter events for real-time frontend display.
"""

from typing import Dict, Any, Optional
from loguru import logger
import asyncio

from ..memory.rag_retrieval.workflow import search_with_workflow
from ..memory.rag_retrieval.state import RetrievalConfig
from .llm_provider import get_llm_from_deps
from .state import State, Deps, Context, DEFAULTS


async def gather_rag_context(
    query: str,
    rag,
    workflow_id: str,
    deps: Deps,
) -> Dict[str, Any]:
    """
    Gather RAG context using the retrieval workflow with emitter integration.

    The emitter is retrieved from deps and used to emit events that
    the frontend displays in collapsible RAG message groups.
    """
    if not rag or not rag.enabled:
        return {"rag_context": "", "doc_context": "", "search_result": None}

    emitter = deps.get("emitter")
    stage = "gather_context"

    retrieval_config = RetrievalConfig(
        max_iterations=5,
        relevance_threshold=0.6,
        max_chunks_per_query=30,
        max_total_chunks=50,
        max_context_chars=8000,
    )

    try:
        if emitter:
            await emitter.rag_search_start(stage)

        # Progress callback for the retrieval workflow
        def on_progress(msg: str):
            logger.debug(f"RAG Retrieval: {msg}")

        result = await search_with_workflow(
            query=query,
            rag=rag,
            workflow_id=workflow_id,
            config=retrieval_config,
            on_progress=on_progress,
        )

        if emitter:
            await emitter.rag_search_end(
                stage=stage,
                total_iterations=result.get("iterations", 0),
                total_chunks=result.get("chunks_found", 0),
                final_relevance=0.0,
                accepted=bool(result.get("context")),
            )

        rag_context = result.get("context", "")

        logger.info(
            f"📚 RAG Retrieval: {result.get('chunks_found', 0)} chunks, "
            f"{result.get('iterations', 0)} iterations"
        )

        # Query documents separately
        doc_context = await _query_documents(query, rag, workflow_id, emitter)

        return {
            "rag_context": rag_context,
            "doc_context": doc_context,
            "search_result": result,
        }

    except Exception as e:
        logger.warning(f"RAG retrieval workflow failed: {e}")

        if emitter:
            await emitter.error(stage, f"RAG search failed: {e}")

        return await _fallback_rag_search(query, rag, workflow_id, emitter)


async def _query_documents(
    query: str,
    rag,
    workflow_id: str,
    emitter,
) -> str:
    """Query uploaded documents separately."""
    try:
        results = rag.query_documents(
            query=query,
            workflow_id=workflow_id,
            n_results=5,
        )

        if not results:
            return ""

        doc_texts = []
        for r in results:
            title = r.get("metadata", {}).get("title", "Document")
            content = r.get("document", "")[:500]
            doc_texts.append(f"[{title}]: {content}")

        return "\n\n".join(doc_texts)

    except Exception as e:
        logger.warning(f"Document query failed: {e}")
        return ""


async def _fallback_rag_search(
    query: str,
    rag,
    workflow_id: str,
    emitter,
) -> Dict[str, Any]:
    """Simple fallback when the retrieval workflow fails."""
    try:
        rag_summary = rag.get_context_summary(
            query=query,
            workflow_id=workflow_id,
            max_tokens=1500,
        )

        return {
            "rag_context": rag_summary or "",
            "doc_context": "",
            "search_result": None,
        }

    except Exception:
        return {"rag_context": "", "doc_context": "", "search_result": None}


# =============================================================================
# GATHER CONTEXT NODE
# =============================================================================

def _run_async(coro):
    """
    Run async coroutine from sync context.

    Handles the case where we're already inside an event loop (FastAPI)
    by scheduling the task instead of blocking.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


async def gather_context_with_agent(state: State, runtime) -> dict:
    """
    Gather all context using the RAG retrieval workflow.

    This is a drop-in replacement for the gather_context node.
    """
    deps = runtime.context
    emitter = deps.get("emitter")

    if emitter:
        await emitter.stage_start("gather_context", "Gathering context with RAG retrieval")

    # Extract query
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "analysis")

    context = Context(rag="", web="", outputs="", plots=[], combined="")
    parts = []

    # Get LLM once for web search sub-calls
    llm = get_llm_from_deps(deps)

    # 1. RAG Context using retrieval workflow
    rag = deps.get("rag")
    if rag and rag.enabled:
        rag_result = await gather_rag_context(
            query=query,
            rag=rag,
            workflow_id=workflow_id,
            deps=deps,
        )

        if rag_result["rag_context"]:
            context["rag"] = rag_result["rag_context"]
            parts.append(f"[Previous Analysis]\n{rag_result['rag_context']}")

        if rag_result["doc_context"]:
            parts.append(f"[Reference Documents]\n{rag_result['doc_context']}")

    # 2. Web search
    if state.get("web_search_enabled"):
        try:
            from .web_search import search_and_synthesize

            if emitter:
                await emitter.stage_start("gather_context", "🌐 Searching web...")

            def on_progress(msg):
                if emitter:
                    _run_async(emitter.stage_start("gather_context", f"🌐 {msg}"))

            search_result = await search_and_synthesize(
                query=query,
                context=context.get("rag", ""),
                llm=llm,
                on_progress=on_progress
            )

            if search_result["results"]:
                context["web"] = search_result["formatted_text"]
                parts.append(f"[Web Research]\n{search_result['formatted_text']}")

                # Store in RAG
                if rag and rag.enabled:
                    web_results = [
                        {
                            "url": r.url,
                            "title": r.title,
                            "content": r.content,
                            "score": r.score,
                            "source": r.source,
                            "enriched": "crawl4ai" in r.source.lower(),
                            "query_used": r.query_used
                        }
                        for r in search_result["results"]
                    ]
                    rag.add_web_search_batch(
                        query=query,
                        results=web_results,
                        stage_name=stage_name,
                        workflow_id=workflow_id,
                    )

                logger.info(f"🌐 Web: {len(search_result['results'])} results")
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

    # 3. Existing outputs
    output_mgr = deps.get("output_manager")
    if output_mgr:
        try:
            existing = output_mgr.get_stage_summary(stage_name)
            if existing:
                context["outputs"] = existing
                parts.append(f"[Existing Outputs]\n{existing}")

                plots = output_mgr.get_plots(stage_name)
                if plots:
                    context["plots"] = plots
        except Exception as e:
            logger.warning(f"Output manager failed: {e}")

    # Combine context
    context["combined"] = "\n\n---\n\n".join(parts) if parts else ""

    if emitter:
        await emitter.stage_end("gather_context", success=True)

    logger.info(f"📋 Context gathered: {len(context['combined'])} chars")

    return {"context": context}
