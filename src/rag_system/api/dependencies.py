"""Shared FastAPI dependencies: ContextRAG instance and session manager."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException
from loguru import logger

from ..ai.llm_provider import clear_cache
from ..ai.state import DEFAULTS
from ..memory.context_rag import ContextRAG


# --- Shared ContextRAG instance ---

_rag_instance: Optional[ContextRAG] = None


def init_rag(
    collection_name: str = "workflow_context",
    persist_directory: str = "cache/rag_db",
) -> None:
    """Initialize the shared ContextRAG instance (called during app lifespan)."""
    global _rag_instance
    _rag_instance = ContextRAG(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    logger.info("RAG instance initialized")


def get_rag() -> ContextRAG:
    """FastAPI dependency returning the shared ContextRAG."""
    if _rag_instance is None:
        raise RuntimeError("ContextRAG not initialized. Call init_rag() first.")
    return _rag_instance


# --- Session Manager ---

class SessionData:
    __slots__ = ("deps", "created_at", "last_access")

    def __init__(self, deps: Dict[str, Any]) -> None:
        self.deps = deps
        self.created_at = datetime.utcnow()
        self.last_access = self.created_at


_sessions: Dict[str, SessionData] = {}


def _default_deps() -> Dict[str, Any]:
    return {
        "provider": DEFAULTS["provider"],
        "llm": DEFAULTS["llm"],
        "code_llm": DEFAULTS["code_llm"],
        "vision_llm": DEFAULTS["vision_llm"],
        "base_url": DEFAULTS["base_url"],
        "api_key": DEFAULTS["api_key"],
        "max_retries": DEFAULTS["max_retries"],
        "verify_enabled": DEFAULTS["verify_enabled"],
        "min_quality_score": DEFAULTS["min_quality_score"],
    }


def create_session() -> str:
    """Create a new session with default config. Returns session_id."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = SessionData(deps=_default_deps())
    logger.info(f"Session created: {session_id}")
    return session_id


def get_session(session_id: str) -> SessionData:
    """Get session by ID, raising 404 if not found."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    session.last_access = datetime.utcnow()
    return session


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def update_session_config(session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Apply partial updates to a session's deps and clear the LLM cache."""
    session = get_session(session_id)
    for key, value in updates.items():
        if value is not None:
            session.deps[key] = value
    clear_cache()
    return session.deps


# --- FastAPI Dependencies ---

def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """Extract session ID from header, or create a default session."""
    if x_session_id and x_session_id in _sessions:
        return x_session_id
    if x_session_id:
        raise HTTPException(status_code=404, detail=f"Session {x_session_id} not found")
    # Auto-create a default session for headerless requests
    return create_session()


def get_session_deps(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency: returns the Deps dict for the current session."""
    sid = get_session_id(x_session_id)
    return get_session(sid).deps
