"""
State schema for RAG retrieval workflow.

Tracks:
- Query todo list and completed queries
- Discovered subtopics
- Retrieved chunks with grades
- Configuration
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A chunk retrieved from RAG with metadata."""
    content: str
    source_type: str  # pdf, text, url, code, summary, web_result
    metadata: Dict[str, Any]
    relevance_score: float
    query_used: str
    
    def __hash__(self):
        return hash(self.content[:100])


@dataclass 
class QueryTask:
    """A query task in the todo list."""
    query: str
    query_type: str  # specific, broad, subtopic, refined
    priority: int = 1  # Higher = more important
    parent_query: Optional[str] = None  # Query that spawned this one


class RetrievalState(TypedDict, total=False):
    """State for RAG retrieval workflow."""
    
    # Input
    original_query: str
    workflow_id: str
    
    # Query management
    query_todo: List[QueryTask]
    completed_queries: List[str]
    current_query: Optional[QueryTask]
    rewritten_query: Optional[str]
    
    # Subtopic tracking
    discovered_subtopics: List[str]
    explored_subtopics: List[str]
    
    # Retrieved context
    retrieved_chunks: List[RetrievedChunk]
    relevant_chunks: List[RetrievedChunk]  # Chunks that passed grading
    
    # Grading state
    last_grade_score: float
    total_relevant_found: int
    consecutive_low_relevance: int  # Track iterations without good results
    
    # Output
    synthesized_context: str
    key_findings: List[str]
    document_types_found: List[str]
    
    # Control
    iteration: int
    max_iterations: int
    should_stop: bool
    error: Optional[str]


@dataclass
class RetrievalConfig:
    """Configuration for retrieval workflow."""
    
    max_iterations: int = 10
    max_chunks_per_query: int = 20
    max_total_chunks: int = 100
    relevance_threshold: float = 0.2
    max_consecutive_low_relevance: int = 5
    max_subtopics_to_explore: int = 5
    max_context_chars: int = 20_000
    
    # LLM settings
    llm_model: str = "qwen3:30b"
    base_url: str = "http://100.91.155.118:11434"


# Type alias for dependencies
class Deps(TypedDict):
    rag: Any  # ContextRAG instance
    config: RetrievalConfig
    on_progress: Optional[Any]  # Callable[[str], None]


def create_initial_state(
    query: str,
    workflow_id: str = "default",
    config: Optional[RetrievalConfig] = None,
) -> RetrievalState:
    """Create initial state for retrieval workflow."""
    cfg = config or RetrievalConfig()
    
    return RetrievalState(
        original_query=query,
        workflow_id=workflow_id,
        query_todo=[],
        completed_queries=[],
        current_query=None,
        rewritten_query=None,
        discovered_subtopics=[],
        explored_subtopics=[],
        retrieved_chunks=[],
        relevant_chunks=[],
        last_grade_score=0.0,
        total_relevant_found=0,
        consecutive_low_relevance=0,
        synthesized_context="",
        key_findings=[],
        document_types_found=[],
        iteration=0,
        max_iterations=cfg.max_iterations,
        should_stop=False,
        error=None,
    )