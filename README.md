# RAG System

An agentic data science system that uses RAG (Retrieval-Augmented Generation) and LangGraph workflows to answer analytical questions. Given a user query and a data file, the system plans an analysis approach, generates and executes Python code, analyzes resulting visualizations with a vision model, searches the web for methodology guidance, and verifies the quality of its own output -- all orchestrated as a directed graph with retry loops.

All LLM inference runs through a local [Ollama](https://ollama.com/) instance using Qwen3 models.

## Features

- **Automated data analysis** -- upload a CSV, ask a question, get code-generated insights with plots
- **RAG-powered context** -- stores and retrieves past analyses, uploaded documents, and web search results via ChromaDB
- **Self-verifying workflows** -- a verification node scores output quality and can trigger retries (re-execute code, gather more context, search the web, or refine the summary)
- **Vision-based plot analysis** -- generated plots are analyzed by a vision LLM to extract insights
- **Agentic web search** -- LLM-guided DuckDuckGo search with optional deep crawling via crawl4ai
- **Background question embedding** -- a daemon thread generates hypothetical questions for each RAG chunk to improve retrieval quality
- **Structured output storage** -- all code, console output, plots, data files, and metadata are saved in an organized directory tree per workflow run

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com/) running with the following models pulled:
  - `qwen3:30b` (general reasoning)
  - `qwen3-coder:30b` (code generation)
  - `qwen3-vl:30b` (vision / plot analysis)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd rag-system

# Install dependencies
uv sync

# Install with optional dependency groups as needed
uv sync --group ai        # LangChain, LangGraph, Ollama, ChromaDB, crawl4ai
uv sync --group server    # FastAPI, Redis, uvicorn
uv sync --group stats     # scikit-learn, XGBoost, statsmodels, imbalanced-learn
uv sync --group prob      # PyMC, NumPyro, PyTensor (Bayesian modeling)
uv sync --group frontend  # Streamlit
uv sync --group jupyter   # ipykernel, ipywidgets
uv sync --group api       # FastAPI, auth (bcrypt, passlib, python-jose), arq
```

## Configuration

The system expects an Ollama instance at `http://100.91.155.118:11434` by default. To change this, override the `base_url` in the `Deps` dictionary when invoking a workflow, or modify the defaults in `src/rag-system/ai/state.py`:

```python
DEFAULTS = {
    "llm": "qwen3:30b",
    "code_llm": "qwen3-coder:30b",
    "vision_llm": "qwen3-vl:30b",
    "base_url": "http://100.91.155.118:11434",
    "max_retries": 3,
    "verify_enabled": True,
    "min_quality_score": 0.7,
}
```

## Running

```bash
# Start the FastAPI server
uv run python -m rag-system.main
```

The server starts at `http://127.0.0.1:8000`.

## Architecture

### Workflow Engine

The system is built on [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` workflows. Each node is an async function that reads from and writes to a shared `State` TypedDict, with runtime dependencies injected via a `Deps` TypedDict.

### Main Analysis Workflow

Defined in `src/rag-system/ai/workflow.py`. Two variants are available:

**`build_workflow_v2()`** -- full workflow with verification loop:

```
plan_initial → gather_context → plan → [route] → summarize → verify → [done or retry]
                                          │
                                    ┌─────┼──────────┐
                                    │     │           │
                                execute  answer   web_search
                                    │
                              analyze_plots
```

After `verify`, if the quality score is below 0.7 or requirements are unmet, the workflow retries by looping back to `execute`, `gather_context`, `web_search`, or `summarize` (up to 2 retries).

**`build_simple_workflow()`** -- skips verification, ends after `summarize`.

### Node Descriptions

| Node | Purpose |
|------|---------|
| `gather_context` | Queries RAG (ChromaDB) for past analyses and documents; optionally runs web search; collects existing outputs |
| `plan` | LLM generates a step-by-step analysis plan and decides the primary action (`execute`, `answer`, `web_search`, or `plot_analysis`) |
| `execute` | LLM generates Python code, runs it in a subprocess, retries on errors (up to `max_retries`) |
| `analyze_plots` | Encodes generated PNG plots as base64, sends to vision LLM for interpretation |
| `web_search` | Agent-guided DuckDuckGo search with optional crawl4ai deep crawling of top results |
| `answer` | Answers directly from gathered context (no code execution) |
| `summarize` | LLM synthesizes all results (code output, plot analyses, context) into a comprehensive answer |
| `verify` | LLM checks output against original query; scores quality 0-1; suggests corrective action if needed |

### RAG Retrieval Workflow

Defined in `src/rag-system/memory/rag_retrieval/workflow.py`. A separate LangGraph workflow dedicated to intelligent context retrieval:

```
plan_queries → rewrite_query ←──┐
                   │             │
            execute_query        │
                   │             │
            grade_context        │
                   │             │
             ┌─────┴─────┐      │
             │           │      │
         synthesize   continue──┘
             │
            END
```

This workflow:
1. Generates specific and broad search queries from the user's question
2. Rewrites queries to match RAG document semantics (e.g., "how do I build" becomes "implementation", "tutorial")
3. Executes queries against ChromaDB
4. Grades retrieved chunks for relevance and extracts subtopics for further exploration
5. Loops back to search subtopics until relevance is satisfied or max iterations reached
6. Synthesizes all relevant chunks into coherent context

### RAG Storage (ChromaDB)

`src/rag-system/memory/context_rag.py` provides `ContextRAG`, a ChromaDB-backed store with:

- **Contextual chunking** -- section-aware text splitting that preserves headers, paragraphs, list groups, and code blocks as semantic units
- **Multiple content types** -- stores code executions, plot analyses, web search results, conversation summaries, and uploaded documents (PDF, text, URLs)
- **Content-based deduplication** via SHA256 hashing

### Background Question Embedding

`src/rag-system/utils/question_generator.py` runs a daemon thread (`QuestionGenerationWorker`) that generates 3-5 hypothetical questions per RAG chunk using a fast LLM. These questions are stored as additional embeddings, improving retrieval because user queries (phrased as questions) are more semantically similar to generated questions than to raw chunk text. The worker cooperatively pauses when the LLM is needed for user queries.

### Code Execution

`src/rag-system/utils/code_executer.py` provides `OutputCapturingExecutor`:

- Runs generated Python in an `asyncio` subprocess with configurable timeout (default 300s)
- Auto-injects matplotlib configuration (Agg backend, professional styling, 300 DPI)
- Captures stdout, stderr, and tracks all generated files
- Supports passing data between stages via pickle files
- Integrates with `OutputManager` for structured result storage

### Output Management

`src/rag-system/utils/output_manager.py` provides `OutputManager`, which creates an organized results directory per workflow:

```
results/
  workflow_{id}_{timestamp}/
    00_orchestrator/
    01_data_acquisition/
      console_output.txt
      data/
    02_eda/
      code.py
      console_output.txt
      plots/
        distribution_*.png
        correlation_heatmap.png
    03_modeling/
      model_code.py
      model_summary.txt
      plots/
      fitted_model.pkl
    manifest.json
```

### Node Factory

`src/rag-system/ai/factory.py` provides `NodeFactory` for creating LangGraph nodes from declarative `NodeConfig` dataclasses. It manages LLM instance caching (keyed by model name, base URL, and temperature) and includes prebuilt configs for planning, code generation, verification, and summarization.

### Web Search

`src/rag-system/ai/web_search.py` implements a `SearchAgent` that:

1. Uses an LLM to generate 2-4 optimized DuckDuckGo search queries
2. Executes searches and evaluates result quality
3. Refines queries iteratively if results are insufficient
4. Optionally enriches top results by crawling full page content with crawl4ai
5. Uses LLM to intelligently select which URLs to crawl (prioritizes official docs, tutorials, authoritative domains)
6. Synthesizes all results into a summary with key insights and relevance score

### Progress Events

`src/rag-system/utils/progress_events.py` provides a file-based event streaming system (`ProgressEmitter` / `ProgressReader`) using JSON Lines format. The UI reads these events for real-time workflow status updates.

### Background Tasks

`src/rag-system/utils/background_tasks.py` provides `BackgroundTaskManager` for running async workflows in background threads. Task status is persisted to disk so it survives page refreshes.

### Plot Analysis Cache

`src/rag-system/memory/plot_analysis_cache.py` provides `PlotAnalysisCache` with content-based caching (SHA256 hash). If a plot file hasn't changed, the cached vision LLM analysis is reused instead of re-analyzing.

## Project Structure

```
src/rag-system/
  main.py                         # FastAPI app entry point
  config.py                       # Application configuration

  ai/                             # Core AI workflow engine
    workflow.py                   # LangGraph workflow definitions (v2 + simple)
    state.py                      # State and Deps TypedDicts, default config
    nodes.py                      # Workflow node implementations
    factory.py                    # NodeFactory and NodeConfig
    routing.py                    # DynamicRouter for intelligent step routing
    verification.py               # OutputVerifier for quality checking
    web_search.py                 # SearchAgent with DuckDuckGo + crawl4ai
    tools.py                      # WorkflowFileTools for inter-stage file access
    rag_agent.py                  # RAGSearchAgent with iterative query refinement
    rag_agent_integration.py      # Connects RAG agent to workflow with emitter events

  memory/                         # RAG and caching
    context_rag.py                # ContextRAG (ChromaDB) with contextual chunking
    plot_analysis_cache.py        # Content-based plot analysis cache

    rag_retrieval/                # Dedicated RAG retrieval workflow
      workflow.py                 # LangGraph retrieval graph
      state.py                    # RetrievalState, RetrievalConfig, QueryTask
      nodes.py                    # plan_queries, rewrite_query, execute_query, grade_context, synthesize
      models.py                   # Pydantic models for structured LLM outputs

  utils/                          # Shared utilities
    code_executer.py              # OutputCapturingExecutor (subprocess code runner)
    output_manager.py             # Structured results directory management
    progress_events.py            # JSON Lines event streaming for UI
    background_tasks.py           # Background thread task manager
    question_generator.py         # Background question embedding worker
    model_getter.py               # Model utilities
```

## Dependency Groups

| Group | Purpose | Key packages |
|-------|---------|-------------|
| (default) | Core runtime | LangChain, LangGraph, ChromaDB, pandas, matplotlib, PyMuPDF, torch |
| `ai` | Extended AI | crawl4ai, deepagents, faster-whisper, langchain-openai |
| `server` | API server | FastAPI, Redis, uvicorn |
| `api` | Full API with auth | FastAPI, bcrypt, passlib, python-jose, arq, crawl4ai |
| `stats` | Statistical modeling | scikit-learn, XGBoost, statsmodels, imbalanced-learn |
| `prob` | Bayesian modeling | PyMC, NumPyro, PyTensor, nutpie |
| `frontend` | UI | Streamlit |
| `jupyter` | Notebooks | ipykernel, ipywidgets |
| `dev` | Development | ipykernel, ipywidgets |
