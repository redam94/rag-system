"""Main API router assembling all sub-routers."""

from fastapi import APIRouter

from .endpoints import config, documents, health, query

api_router = APIRouter(prefix="/api")

api_router.include_router(health.router, tags=["health"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
