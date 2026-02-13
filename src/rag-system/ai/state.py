"""
Workflow State V2 - Extended state for modular workflow.

Includes:
- Verification results
- Retry tracking
- Plan steps
- Plot analyses
"""

from typing import TypedDict, Any, List, Union, Annotated, Optional, Dict
import operator
from langchain.messages import HumanMessage, AIMessage, SystemMessage


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
    
    # Progress tracking
    progress_emitter: Any
    
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


# Default configuration
DEFAULTS = {
    "llm": "qwen3:30b",
    "code_llm": "qwen3-coder:30b",
    "vision_llm": "qwen3-vl:30b",
    "base_url": "http://100.91.155.118:11434",
    "max_retries": 3,
    "verify_enabled": True,
    "min_quality_score": 0.7,
}