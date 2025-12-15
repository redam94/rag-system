"""
LangGraph workflow for RAG retrieval.

Graph Structure:

    ┌─────────────────┐
    │  plan_queries   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ rewrite_query   │◄────────┐
    └────────┬────────┘         │
             │                  │
    ┌────────▼────────┐         │
    │  execute_query  │         │
    └────────┬────────┘         │
             │                  │
    ┌────────▼────────┐         │
    │  grade_context  │         │
    └────────┬────────┘         │
             │                  │
      ┌──────┴──────┐           │
      │             │           │
      ▼             ▼           │
 [synthesize]  [continue]───────┘
      │
      ▼
    [END]
"""

from typing import Dict, Any, Optional, Callable
from loguru import logger

from langgraph.graph import StateGraph, START, END

from .state import RetrievalState, RetrievalConfig, Deps, create_initial_state
from .nodes import (
    plan_queries,
    rewrite_query,
    execute_query,
    grade_context,
    synthesize,
    should_continue,
)


def build_retrieval_workflow() -> StateGraph:
    """Build the RAG retrieval workflow graph."""
    
    logger.info("🔧 Building RAG retrieval workflow...")
    
    # Create graph with state schema and deps type
    graph = StateGraph(RetrievalState, Deps)
    
    # Add nodes
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("execute_query", execute_query)
    graph.add_node("grade_context", grade_context)
    graph.add_node("synthesize", synthesize)
    
    # Define flow
    graph.add_edge(START, "plan_queries")
    graph.add_edge("plan_queries", "rewrite_query")
    graph.add_edge("rewrite_query", "execute_query")
    graph.add_edge("execute_query", "grade_context")
    
    # Conditional routing after grading
    graph.add_conditional_edges(
        "grade_context",
        should_continue,
        {
            "rewrite_query": "rewrite_query",
            "synthesize": "synthesize",
        }
    )
    
    graph.add_edge("synthesize", END)
    
    logger.info("✅ RAG retrieval workflow built")
    
    return graph.compile()


# Pre-compiled workflow instance
retrieval_workflow = build_retrieval_workflow()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def search_with_workflow(
    query: str,
    rag,
    workflow_id: str = "default",
    config: Optional[RetrievalConfig] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the RAG retrieval workflow.
    
    Args:
        query: User query
        rag: ContextRAG instance
        workflow_id: Workflow ID for filtering
        config: Retrieval configuration
        on_progress: Progress callback
        
    Returns:
        Dict with synthesized_context, key_findings, etc.
    """
    cfg = config or RetrievalConfig()
    
    # Create initial state
    initial_state = create_initial_state(
        query=query,
        workflow_id=workflow_id,
        config=cfg,
    )
    
    # Build dependencies
    deps: Deps = {
        "rag": rag,
        "config": cfg,
        "on_progress": on_progress,
    }
    
    logger.info(f"🚀 Starting RAG retrieval workflow for: {query[:50]}...")
    
    try:
        # Run workflow with deps as context
        result = await retrieval_workflow.ainvoke(
            initial_state,
            context=deps,
            config={"recursion_limit": 100}
        )
        
        logger.info(
            f"✅ Retrieval complete: "
            f"{result.get('total_relevant_found', 0)} relevant chunks, "
            f"{result.get('iteration', 0)} iterations"
        )
        
        return {
            "context": result.get("synthesized_context", ""),
            "key_findings": result.get("key_findings", []),
            "document_types": result.get("document_types_found", []),
            "chunks_found": result.get("total_relevant_found", 0),
            "iterations": result.get("iteration", 0),
            "queries_used": result.get("completed_queries", []),
            "subtopics_discovered": result.get("discovered_subtopics", []),
        }
        
    except Exception as e:
        logger.error(f"RAG retrieval workflow failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "context": "",
            "key_findings": [],
            "error": str(e),
        }