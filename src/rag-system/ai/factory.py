"""
Node Factory - Configurable node creation for workflows.

Provides:
- NodeConfig: Configuration for a node (model, prompts, tools)
- NodeFactory: Creates node functions with custom configurations
- Prebuilt node configs for common patterns
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel
from loguru import logger

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.messages import SystemMessage, HumanMessage

from .llm_provider import get_llm_from_deps
from .prompts import (
    FACTORY_PLANNER,
    FACTORY_CODE_GENERATOR,
    FACTORY_VERIFIER,
    FACTORY_SUMMARIZER,
    FACTORY_PLANNER_TEMPLATE,
    FACTORY_CODE_GENERATOR_TEMPLATE,
    FACTORY_VERIFIER_TEMPLATE,
    FACTORY_SUMMARIZER_TEMPLATE,
)


@dataclass
class NodeConfig:
    """Configuration for a workflow node."""

    name: str
    system_prompt: str
    model_key: str = "llm"  # Key in deps: llm, code_llm, vision_llm
    temperature: float = 0.0
    output_schema: Optional[Type[BaseModel]] = None
    tools: List[Callable] = field(default_factory=list)
    max_context_chars: int = 3000

    # Prompt templates - use {var} placeholders
    user_prompt_template: str = "{query}"

    # Optional pre/post processors: (state, deps) -> state / (output, state, deps) -> dict
    pre_process: Optional[Callable[..., Dict]] = None
    post_process: Optional[Callable[..., Dict]] = None

    # Custom prompt variable extractor: (state, config, deps) -> dict of template vars.
    # When set, replaces the default _extract_prompt_vars.
    prompt_vars_extractor: Optional[Callable[..., Dict[str, str]]] = None

    # Custom LLM provider: (deps) -> BaseChatModel.
    # When set, replaces the default get_llm lookup.
    llm_provider: Optional[Callable[..., BaseChatModel]] = None


class NodeFactory:
    """
    Factory for creating workflow nodes with custom configurations.

    Usage:
        factory = NodeFactory()

        config = NodeConfig(
            name="planner",
            system_prompt="You are a planning agent...",
            output_schema=PlanOutput,
        )

        planner_node = factory.create_node(config)
    """

    def __init__(self):
        pass

    def get_llm(
        self,
        model_key: str,
        deps: Optional[Dict] = None,
        temperature: float = 0.0,
    ) -> BaseChatModel:
        """Get a cached LLM instance via the central provider."""
        if deps is None:
            deps = {}
        return get_llm_from_deps(deps, model_key)

    def create_node(self, config: NodeConfig) -> Callable:
        """
        Create a node function from configuration.

        Returns an async function compatible with LangGraph.
        """

        async def node_fn(state: Dict, runtime) -> Dict:
            deps = runtime.context
            emitter = deps.get("progress_emitter")

            # Emit start
            if emitter:
                emitter.stage_start(config.name, f"Running {config.name}")

            logger.info(f"🔧 Node [{config.name}] starting")

            # Pre-process state if configured
            if config.pre_process:
                state = config.pre_process(state, deps)

            # Early exit: pre_process can set _skip_llm to a dict to return immediately
            skip = state.get("_skip_llm")
            if isinstance(skip, dict):
                logger.info(f"⏭️ Node [{config.name}] skipping LLM (early exit)")
                if emitter:
                    emitter.stage_end(config.name, success=True)
                return skip

            # Build prompt variables from state
            if config.prompt_vars_extractor:
                prompt_vars = config.prompt_vars_extractor(state, config, deps)
            else:
                prompt_vars = self._extract_prompt_vars(state, config)

            # Format user prompt
            try:
                user_prompt = config.user_prompt_template.format(**prompt_vars)
            except KeyError as e:
                logger.warning(f"Missing prompt variable: {e}")
                user_prompt = config.user_prompt_template

            # Get LLM
            if config.llm_provider:
                llm = config.llm_provider(deps)
            else:
                llm = self.get_llm(config.model_key, deps, config.temperature)

            # Apply structured output if schema provided
            if config.output_schema:
                llm = llm.with_structured_output(config.output_schema)

            # Invoke
            try:
                messages = [
                    SystemMessage(content=config.system_prompt),
                    HumanMessage(content=user_prompt),
                ]

                result = llm.invoke(messages)

                # Convert pydantic model to dict if needed
                if config.output_schema and hasattr(result, "model_dump"):
                    output = result.model_dump()
                elif isinstance(result, BaseModel):
                    output = result.model_dump()
                else:
                    output = {"response": result.content if hasattr(result, "content") else str(result)}

                logger.info(f"✅ Node [{config.name}] completed")

            except Exception as e:
                logger.error(f"❌ Node [{config.name}] failed: {e}")
                output = {"error": str(e)}

            # Post-process if configured
            if config.post_process:
                output = config.post_process(output, state, deps)

            # Emit end
            if emitter:
                emitter.stage_end(config.name, success="error" not in output)

            return output

        # Set function name for debugging
        node_fn.__name__ = config.name
        return node_fn

    def _extract_prompt_vars(self, state: Dict, config: NodeConfig) -> Dict[str, str]:
        """Extract variables for prompt template from state."""
        vars = {}

        # Query from messages
        messages = state.get("messages", [])
        vars["query"] = messages[-1].content if messages else ""

        # Context
        context = state.get("context", {})
        if isinstance(context, dict):
            vars["context"] = context.get("combined", "")[:config.max_context_chars]
            vars["rag_context"] = context.get("rag", "")[:config.max_context_chars]
            vars["web_context"] = context.get("web", "")[:config.max_context_chars]
        else:
            vars["context"] = str(context)[:config.max_context_chars]
            vars["rag_context"] = ""
            vars["web_context"] = ""

        # Plan
        vars["plan"] = state.get("plan", "")

        # Data path
        vars["data_path"] = state.get("data_path", "")

        # Code and output
        vars["code"] = state.get("code", "")
        vars["stdout"] = ""
        vars["stderr"] = ""
        if state.get("output"):
            output = state["output"]
            if hasattr(output, "stdout"):
                vars["stdout"] = output.stdout[:2000] if output.stdout else ""
            if hasattr(output, "stderr"):
                vars["stderr"] = output.stderr[:1000] if output.stderr else ""

        # Error
        vars["error"] = state.get("error", "")

        # Summary
        vars["summary"] = state.get("summary", "")

        return vars


# =============================================================================
# PREBUILT NODE CONFIGS
# =============================================================================

class PlanOutput(BaseModel):
    """Output schema for planning node."""
    plan: str
    steps: List[str]
    requires_code: bool
    requires_web_search: bool
    requires_rag: bool
    requires_plot_analysis: bool


class CodeOutput(BaseModel):
    """Output schema for code generation."""
    code: str
    reasoning: str


class VerificationOutput(BaseModel):
    """Output schema for verification node."""
    is_complete: bool
    missing_items: List[str]
    quality_score: float  # 0-1
    feedback: str
    suggested_action: str  # "done", "retry_code", "add_context", "refine"


# Pre-configured node configs
PLANNER_CONFIG = NodeConfig(
    name="planner",
    system_prompt=FACTORY_PLANNER,
    output_schema=PlanOutput,
    user_prompt_template=FACTORY_PLANNER_TEMPLATE,
)


CODE_GENERATOR_CONFIG = NodeConfig(
    name="code_generator",
    model_key="code_llm",
    system_prompt=FACTORY_CODE_GENERATOR,
    output_schema=CodeOutput,
    user_prompt_template=FACTORY_CODE_GENERATOR_TEMPLATE,
)


VERIFIER_CONFIG = NodeConfig(
    name="verifier",
    system_prompt=FACTORY_VERIFIER,
    output_schema=VerificationOutput,
    user_prompt_template=FACTORY_VERIFIER_TEMPLATE,
)


SUMMARIZER_CONFIG = NodeConfig(
    name="summarizer",
    system_prompt=FACTORY_SUMMARIZER,
    user_prompt_template=FACTORY_SUMMARIZER_TEMPLATE,
)
