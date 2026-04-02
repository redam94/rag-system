"""
Workflow State V2 - Extended state for modular workflow.

Includes:
- Verification results
- Retry tracking
- Plan steps
- Plot analyses
"""

import os
from typing import TypedDict, Any, List, Union, Annotated, Optional, Dict
import operator
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class Context(TypedDict, total=False):
    """All gathered context in one place."""
    rag: str
    web: str
    outputs: str
    plots: List[str]
    combined: str


class VerificationResult(TypedDict, total=False):
    """Verification output."""
    is_complete: bool
    quality_score: float
    missing_items: List[str]
    strengths: List[str]
    weaknesses: List[str]
    suggested_action: str
    feedback: str


class State(TypedDict, total=False):
    """Extended workflow state."""

    # Input
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    data_path: str
    stage_name: str
    workflow_id: str
    web_search_enabled: bool

    # Context (gathered once, may be refreshed on retry)
    context: Context

    # Plan & Decision
    plan: str
    plan_steps: List[str]
    action: str  # "answer" | "execute" | "web_search" | "plot_analysis"

    # Execution
    code: str
    output: Any
    error: str

    # Plot Analysis
    plot_analyses: List[Dict[str, str]]

    # Result
    summary: str

    # Verification
    verification: VerificationResult
    verified: bool

    # Retry tracking
    retry_count: int


class Deps(TypedDict, total=False):
    """Runtime dependencies - passed as context to workflow."""
    executor: Any
    output_manager: Any
    rag: Any
    plot_cache: Any
    emitter: Any

    # Progress tracking
    progress_emitter: Any

    # LLM provider config
    provider: str       # "ollama", "openai", "anthropic", "google_vertexai"
    api_key: str        # API key for non-ollama providers

    # Model names
    llm: str
    code_llm: str
    vision_llm: str

    # Config
    base_url: str
    max_retries: int

    # Verification config
    verify_enabled: bool
    min_quality_score: float


# Default configuration (loaded from environment with fallbacks)
DEFAULTS = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),
    "llm": os.getenv("LLM_MODEL", "qwen3:30b"),
    "code_llm": os.getenv("CODE_LLM_MODEL", "qwen3-coder:30b"),
    "vision_llm": os.getenv("VISION_LLM_MODEL", "qwen3-vl:30b"),
    "base_url": os.getenv("LLM_BASE_URL", "http://100.91.155.118:11434"),
    "api_key": os.getenv("LLM_API_KEY", ""),
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "verify_enabled": os.getenv("VERIFY_ENABLED", "true").lower() == "true",
    "min_quality_score": float(os.getenv("MIN_QUALITY_SCORE", "0.7")),
}
