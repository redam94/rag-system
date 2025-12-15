"""
Pydantic models for RAG retrieval workflow.

Structured outputs for LLM nodes:
- QueryPlan: Initial query generation
- RewrittenQuery: Semantic query transformation
- ContextGrade: Relevance grading with subtopic extraction
- SynthesisResult: Final context synthesis
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    """Initial query plan with specific and broad queries."""
    
    specific_queries: List[str] = Field(
        description="Specific queries targeting the exact user request"
    )
    broad_queries: List[str] = Field(
        description="Broader queries to capture related concepts"
    )
    reasoning: str = Field(
        description="Explanation of query strategy"
    )


class RewrittenQuery(BaseModel):
    """Query rewritten to match RAG document semantics."""
    
    original: str = Field(description="Original query")
    rewritten: str = Field(description="Semantically transformed query")
    alternative_phrasings: List[str] = Field(
        default_factory=list,
        description="Alternative ways to phrase this query"
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Key technical terms to search for"
    )


class ContextGrade(BaseModel):
    """Grading result for retrieved context."""
    
    is_relevant: bool = Field(description="Whether context is relevant to the query")
    relevance_score: float = Field(
        description="Relevance score 0-1",
        ge=0,
        le=1
    )
    extracted_subtopics: List[str] = Field(
        default_factory=list,
        description="New subtopics discovered in the context worth exploring"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects of the query not yet covered"
    )
    should_continue: bool = Field(
        description="Whether to continue searching for more context"
    )
    summary: str = Field(
        description="Brief summary of what this context contributes"
    )


class SynthesisResult(BaseModel):
    """Final synthesized context result."""
    
    combined_context: str = Field(
        description="Synthesized context combining all relevant chunks"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from the retrieved context"
    )
    document_types_found: List[str] = Field(
        default_factory=list,
        description="Types of documents that contributed context"
    )
    coverage_assessment: str = Field(
        description="Assessment of how well the context covers the query"
    )
    confidence: float = Field(
        description="Confidence in the synthesized context",
        ge=0,
        le=1
    )