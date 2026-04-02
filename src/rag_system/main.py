from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api.router import api_router
from .api.dependencies import init_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag()
    yield


app = FastAPI(
    title="RAG System API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {"service": "rag-system", "version": "0.1.0"}


def main():
    uvicorn.run(
        "rag_system.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
