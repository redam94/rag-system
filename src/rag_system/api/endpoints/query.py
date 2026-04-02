"""RAG search and workflow execution endpoints."""

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends
from langchain.messages import HumanMessage
from loguru import logger

from ..dependencies import get_rag, get_session_id, get_session
from ..models import (
    QueryRequest,
    QueryResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGSearchResult,
)
from ...memory.context_rag import ContextRAG

router = APIRouter()


@router.post("/rag", response_model=RAGSearchResponse)
async def rag_search(
    body: RAGSearchRequest,
    session_id: str = Depends(get_session_id),
):
    """Direct vector search against the RAG store (no workflow)."""
    rag = get_rag()

    results = rag.query_relevant_context(
        query=body.query,
        doc_types=body.doc_types,
        document_titles=body.document_titles,
        n_results=body.n_results,
    )

    items = [
        RAGSearchResult(
            text=r.get("text", r.get("document", "")),
            metadata=r.get("metadata", {}),
            distance=r.get("distance"),
        )
        for r in results
    ]

    return RAGSearchResponse(
        query=body.query,
        results=items,
        total_results=len(items),
    )


@router.post("/workflow", response_model=QueryResponse)
async def run_workflow(
    body: QueryRequest,
    session_id: str = Depends(get_session_id),
):
    """Execute a full LangGraph workflow against the query."""
    from ...ai.workflow import build_simple_workflow, build_workflow_v2
    from ...utils.code_executer import OutputCapturingExecutor
    from ...utils.output_manager import OutputManager

    session = get_session(session_id)
    deps = dict(session.deps)

    workflow_id = f"api_{uuid.uuid4().hex[:8]}"
    output_manager = OutputManager(workflow_id=workflow_id)
    executor = OutputCapturingExecutor()

    rag = get_rag()

    deps["executor"] = executor
    deps["output_manager"] = output_manager
    deps["rag"] = rag

    initial_state = {
        "messages": [HumanMessage(content=body.query)],
        "workflow_id": workflow_id,
        "web_search_enabled": body.web_search_enabled,
        "retry_count": 0,
    }
    if body.data_path:
        initial_state["data_path"] = body.data_path

    if body.workflow_type == "full":
        compiled = build_workflow_v2()
    else:
        compiled = build_simple_workflow()

    start = time.time()

    try:
        final_state = await compiled.ainvoke(
            initial_state,
            context=deps,
        )
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return QueryResponse(
            query=body.query,
            summary=f"Workflow failed: {e}",
            workflow_used=body.workflow_type,
            execution_time_seconds=round(time.time() - start, 2),
        )

    elapsed = round(time.time() - start, 2)
    summary = final_state.get("summary", "No summary generated.")

    return QueryResponse(
        query=body.query,
        summary=summary,
        workflow_used=body.workflow_type,
        execution_time_seconds=elapsed,
    )
