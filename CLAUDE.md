# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG-powered data science agent system that uses LangGraph workflows to answer analytical questions. It generates and executes Python code, analyzes plots with vision models, searches the web, and retrieves context from a ChromaDB-backed RAG store. All LLM inference runs through a local Ollama instance.

## Setup & Commands

```bash
# Install dependencies (uses uv, Python 3.13)
uv sync

# Install with specific dependency groups
uv sync --group ai --group server --group stats

# Run the FastAPI server
uv run python -m rag-system.main
```

There are no tests or linting configured in this project.

## Architecture

### LangGraph Workflows

The core system is built on two LangGraph `StateGraph` workflows in `src/rag-system/ai/`:

**Main analysis workflow** (`workflow.py`): `plan -> gather_context -> plan -> [execute|answer|web_search] -> summarize -> verify -> [done|retry]`
- `build_workflow_v2()` includes a verification loop with retry (max 2 retries)
- `build_simple_workflow()` skips verification, goes straight to END after summarize
- State is defined in `state.py` as `State` (TypedDict) with `Deps` for runtime dependencies

**RAG retrieval workflow** (`memory/rag_retrieval/workflow.py`): `plan_queries -> rewrite_query -> execute_query -> grade_context -> [synthesize|continue]`
- Iterative query refinement loop that searches ChromaDB until context meets a relevance threshold

### Key Components

- **NodeFactory** (`ai/factory.py`): Creates LangGraph nodes from `NodeConfig` dataclasses. Manages LLM instance caching. Prebuilt configs: `PLANNER_CONFIG`, `CODE_GENERATOR_CONFIG`, `VERIFIER_CONFIG`, `SUMMARIZER_CONFIG`.
- **OutputCapturingExecutor** (`utils/code_executer.py`): Runs generated Python code in subprocess, captures stdout/stderr, tracks generated files. Auto-injects matplotlib styling and Agg backend.
- **OutputManager** (`utils/output_manager.py`): Structured results storage under `results/workflow_{id}_{timestamp}/` with stage-based subdirectories for code, plots, console output, and data.
- **ContextRAG** (`memory/context_rag.py`): ChromaDB-backed RAG with contextual chunking (section-aware, code-block-aware). Stores code executions, plot analyses, web search results, and document uploads.
- **RAGSearchAgent** (`ai/rag_agent.py`): Agentic RAG search with LLM-driven query generation, relevance scoring (relevance + recency weighting), and iterative refinement.

### LLM Configuration

All LLM construction is centralized in `ai/llm_provider.py`. It returns `BaseChatModel` (LangChain's abstract interface) and supports four providers: `ollama`, `openai`, `anthropic`, `google_vertexai`.

Configuration is loaded from environment variables (see `.env.example`) with fallbacks in `state.py` DEFAULTS:
- `LLM_PROVIDER` — provider name (default: `ollama`)
- `LLM_MODEL` / `CODE_LLM_MODEL` / `VISION_LLM_MODEL` — model names per role
- `LLM_BASE_URL` — Ollama/OpenAI-compatible endpoint
- `LLM_API_KEY` — API key for cloud providers

All classes (`SearchAgent`, `OutputVerifier`, `DynamicRouter`, `RAGSearchAgent`, etc.) accept a `BaseChatModel` instance rather than constructing their own. Use `get_llm_from_deps(deps, model_key)` in node functions to get a cached LLM.

## Package Structure

The Python package lives under `src/rag-system/` (note: hyphenated directory name). Key subdirectories:
- `ai/` — LangGraph workflows, nodes, routing, verification, web search, factory, LLM provider
- `memory/` — ChromaDB RAG store, contextual chunking, retrieval workflow
- `utils/` — code executor, output manager, progress events, model utilities
