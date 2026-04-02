"""Health check endpoint."""

from fastapi import APIRouter, Depends

from ..dependencies import get_rag, get_session_deps
from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(deps: dict = Depends(get_session_deps)):
    rag = get_rag()
    return HealthResponse(
        status="ok",
        rag_enabled=rag.enabled,
        llm_provider=deps.get("provider", "unknown"),
        model=deps.get("llm", "unknown"),
    )
