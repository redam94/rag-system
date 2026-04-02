"""Pydantic request/response models for the RAG System API."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --- LLM Configuration ---

class LLMConfigRequest(BaseModel):
    provider: Optional[Literal["ollama", "openai", "anthropic", "google_vertexai"]] = None
    model: Optional[str] = None
    code_model: Optional[str] = None
    vision_model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(None, exclude=True)


class LLMConfigResponse(BaseModel):
    provider: str
    model: str
    code_model: str
    vision_model: str
    base_url: str
    api_key_set: bool


class APIKeyRequest(BaseModel):
    api_key: str


class APIKeyResponse(BaseModel):
    message: str
    api_key_set: bool


# --- Sessions ---

class SessionResponse(BaseModel):
    session_id: str
    llm_config: LLMConfigResponse
    created_at: datetime


# --- Documents ---

class DocumentUploadResponse(BaseModel):
    filename: str
    source_type: str
    chunks_added: int
    message: str


class DocumentListItem(BaseModel):
    title: str
    source_type: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int


# --- RAG Search ---

class RAGSearchRequest(BaseModel):
    query: str
    n_results: int = 10
    doc_types: Optional[List[str]] = None
    document_titles: Optional[List[str]] = None


class RAGSearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class RAGSearchResponse(BaseModel):
    query: str
    results: List[RAGSearchResult]
    total_results: int


# --- Workflow Query ---

class QueryRequest(BaseModel):
    query: str
    workflow_type: Literal["simple", "full"] = "simple"
    data_path: Optional[str] = None
    web_search_enabled: bool = False


class QueryResponse(BaseModel):
    query: str
    summary: str
    workflow_used: str
    execution_time_seconds: float


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    rag_enabled: bool
    llm_provider: str
    model: str
