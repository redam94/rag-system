"""LLM configuration and API key management endpoints."""

from fastapi import APIRouter, Depends, Header, HTTPException
from typing import Optional

from ..dependencies import (
    create_session,
    get_session,
    get_session_id,
    update_session_config,
)
from ..models import (
    APIKeyRequest,
    APIKeyResponse,
    LLMConfigRequest,
    LLMConfigResponse,
    SessionResponse,
)

router = APIRouter()


def _config_response(deps: dict) -> LLMConfigResponse:
    return LLMConfigResponse(
        provider=deps.get("provider", ""),
        model=deps.get("llm", ""),
        code_model=deps.get("code_llm", ""),
        vision_model=deps.get("vision_llm", ""),
        base_url=deps.get("base_url", ""),
        api_key_set=bool(deps.get("api_key")),
    )


# --- Session ---

@router.post("/session", response_model=SessionResponse)
async def create_new_session():
    """Create a new session with default configuration."""
    session_id = create_session()
    session = get_session(session_id)
    return SessionResponse(
        session_id=session_id,
        llm_config=_config_response(session.deps),
        created_at=session.created_at,
    )


# --- LLM Config ---

@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_config(session_id: str = Depends(get_session_id)):
    session = get_session(session_id)
    return _config_response(session.deps)


@router.post("/llm", response_model=LLMConfigResponse)
async def update_llm_config(
    config: LLMConfigRequest,
    session_id: str = Depends(get_session_id),
):
    """Update LLM provider, model, or base_url for this session."""
    updates = {}
    if config.provider is not None:
        updates["provider"] = config.provider
    if config.model is not None:
        updates["llm"] = config.model
    if config.code_model is not None:
        updates["code_llm"] = config.code_model
    if config.vision_model is not None:
        updates["vision_llm"] = config.vision_model
    if config.base_url is not None:
        updates["base_url"] = config.base_url
    if config.api_key is not None:
        updates["api_key"] = config.api_key

    deps = update_session_config(session_id, updates)
    return _config_response(deps)


# --- API Key ---

@router.post("/api-key", response_model=APIKeyResponse)
async def set_api_key(
    body: APIKeyRequest,
    session_id: str = Depends(get_session_id),
):
    """Set the API key for this session (stored in memory only)."""
    update_session_config(session_id, {"api_key": body.api_key})
    return APIKeyResponse(message="API key updated", api_key_set=True)


@router.delete("/api-key", response_model=APIKeyResponse)
async def clear_api_key(session_id: str = Depends(get_session_id)):
    """Clear the API key for this session."""
    update_session_config(session_id, {"api_key": ""})
    return APIKeyResponse(message="API key cleared", api_key_set=False)
