"""
RAG Retrieval Workflow Module

Multi-stage LangGraph workflow for intelligent RAG context retrieval.

Features:
- Query planning with specific + broad queries
- Semantic query rewriting to match document patterns
- Iterative refinement with subtopic discovery
- Context grading with relevance scoring
- Synthesis across document types

Usage:
    from rag_retrieval import search_with_workflow, RetrievalConfig
    
    result = await search_with_workflow(
        query="how can I build a linear regression",
        rag=my_rag_instance,
        workflow_id="my_workflow",
        config=RetrievalConfig(max_iterations=8),
        on_progress=lambda msg: print(msg),
    )
    
    print(result["context"])
    print(result["key_findings"])
    print(result["subtopics_discovered"])

Workflow Graph:
    plan_queries → rewrite_query → execute_query → grade_context
                        ↑                              │
                        └──────── [continue] ──────────┘
                                       │
                                  [synthesize] → END
"""

from .models import (
    QueryPlan,
    RewrittenQuery,
    ContextGrade,
    SynthesisResult,
)

from .state import (
    RetrievalState,
    RetrievalConfig,
    RetrievedChunk,
    QueryTask,
    Deps,
    create_initial_state,
)

from .nodes import (
    plan_queries,
    rewrite_query,
    execute_query,
    grade_context,
    synthesize,
    should_continue,
)

from .workflow import (
    build_retrieval_workflow,
    retrieval_workflow,
    search_with_workflow,
)


__all__ = [
    # Main API
    "search_with_workflow",
    "retrieval_workflow",
    "build_retrieval_workflow",
    
    # Configuration
    "RetrievalConfig",
    "RetrievalState",
    "create_initial_state",
    
    # Data types
    "RetrievedChunk",
    "QueryTask",
    "Deps",
    
    # Models (for extension)
    "QueryPlan",
    "RewrittenQuery", 
    "ContextGrade",
    "SynthesisResult",
    
    # Nodes (for custom workflows)
    "plan_queries",
    "rewrite_query",
    "execute_query",
    "grade_context",
    "synthesize",
    "should_continue",
]