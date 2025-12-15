"""
API Schemas - Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowAction(str, Enum):
    """Possible workflow actions."""

    EXECUTE = "execute"
    ANSWER = "answer"


class DocumentSourceType(str, Enum):
    """Supported document source types."""

    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    JSON = "json"
    URL = "url"


# =============================================================================
# WORKFLOW SCHEMAS
# =============================================================================


class DataFileUploadResponse(BaseModel):
    """Response after uploading a data file for workflow analysis."""

    success: bool
    filename: str
    file_path: str = Field(..., description="Temporary path to use as data_path in workflow")
    file_size: int
    content_type: str
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "filename": "sales_data.csv",
                "file_path": "/tmp/workflow_data/abc123/sales_data.csv",
                "file_size": 1024,
                "content_type": "text/csv",
                "message": "File uploaded successfully",
            }
        }
    }


class WorkflowRequest(BaseModel):
    """Request to run the workflow."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    stage_name: str = Field(default="analysis", description="Stage name within workflow")
    query: str = Field(..., description="User query to process")
    data_path: Optional[str] = Field(default=None, description="Path to data file (CSV)")
    web_search_enabled: bool = Field(default=False, description="Enable web search")
    rag_enabled: bool = Field(default=True, description="Enable RAG context retrieval")

    model_config = {
        "json_schema_extra": {
            "example": {
                "workflow_id": "analysis_001",
                "stage_name": "exploration",
                "query": "Analyze the correlation between price and sales",
                "data_path": "/data/sales.csv",
                "web_search_enabled": False,
                "rag_enabled": True,
            }
        }
    }


class WorkflowMessage(BaseModel):
    """A message from the workflow."""

    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""

    workflow_id: str
    stage_name: str
    status: str = Field(..., description="Status: completed, error, in_progress")
    message: WorkflowMessage
    action_taken: Optional[str] = None
    code_executed: Optional[str] = None
    plots: Optional[List[str]] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)
    execution_time_seconds: float = Field(default=0.0)


class WorkflowStatus(BaseModel):
    """Status of a running workflow."""

    task_id: Optional[str] = None
    workflow_id: str
    stage_name: str
    status: str
    current_node: Optional[str] = None
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    elapsed_seconds: Optional[float] = None


# =============================================================================
# CHAT SCHEMAS
# =============================================================================


class ChatMessage(BaseModel):
    """Chat message for streaming endpoint."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request for streaming chat with RAG."""

    workflow_id: str = Field(..., description="Workflow context for RAG")
    message: str = Field(..., description="User message")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Previous messages")
    model: Optional[str] = Field(default="qwen3:30b", description="LLM model to use")

    model_config = {
        "json_schema_extra": {
            "example": {
                "workflow_id": "analysis_001",
                "message": "What patterns did we find in the sales data?",
                "history": [
                    {"role": "user", "content": "Analyze sales trends"},
                    {"role": "assistant", "content": "I found three key patterns..."},
                ],
            }
        }
    }


class ChatChunk(BaseModel):
    """Streaming chat chunk."""

    content: str
    done: bool = False
    tool_call: Optional[str] = None


# =============================================================================
# DOCUMENT SCHEMAS
# =============================================================================


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    success: bool
    title: str
    source_type: DocumentSourceType
    chunk_count: int
    content_length: int
    workflow_id: str
    message: str


class URLScrapeRequest(BaseModel):
    """Request to scrape a URL."""

    url: str = Field(..., description="URL to scrape")
    workflow_id: str = Field(..., description="Workflow to associate with")
    title: Optional[str] = Field(default=None, description="Optional title override")


class URLScrapeResponse(BaseModel):
    """Response from URL scraping."""

    success: bool
    title: str
    url: str
    chunk_count: int
    content_length: int
    workflow_id: str
    message: str


# =============================================================================
# RAG SCHEMAS
# =============================================================================


class RAGQueryRequest(BaseModel):
    """Request to query RAG."""

    query: str
    workflow_id: str
    n_results: int = Field(default=5, ge=1, le=20)
    doc_types: Optional[List[str]] = Field(default=None, description="Filter by document types")


class RAGChunk(BaseModel):
    """A RAG result chunk."""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = Field(default=0.0)


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""

    query: str
    results: List[RAGChunk]
    total_found: int


class RAGStats(BaseModel):
    """RAG collection statistics."""

    workflow_id: str
    total_documents: int
    total_chunks: int
    sources: List[str]


# =============================================================================
# HEALTH SCHEMAS
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "1.0.0"
    ollama_connected: bool = False
    rag_available: bool = False
    workflow_ready: bool = False


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_type: Optional[str] = None


# =============================================================================
# DATABASE MODELS
# =============================================================================

class WorkflowCreate(BaseModel):
    workflow_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=512)

class WorkflowInfo(BaseModel):
    workflow_id: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    stage_count: int = 0
    has_results: bool = False