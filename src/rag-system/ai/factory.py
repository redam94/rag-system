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

from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage


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
    
    # Optional pre/post processors
    pre_process: Optional[Callable[[Dict], Dict]] = None
    post_process: Optional[Callable[[Any, Dict], Dict]] = None


class NodeFactory:
    """
    Factory for creating workflow nodes with custom configurations.
    
    Usage:
        factory = NodeFactory(base_url="http://...", defaults={...})
        
        config = NodeConfig(
            name="planner",
            system_prompt="You are a planning agent...",
            output_schema=PlanOutput,
        )
        
        planner_node = factory.create_node(config)
    """
    
    def __init__(
        self,
        base_url: str = "http://100.91.155.118:11434",
        default_models: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url
        self.default_models = default_models or {
            "llm": "qwen3:30b",
            "code_llm": "qwen3-coder:30b",
            "vision_llm": "qwen3-vl:30b",
        }
        self._llm_cache: Dict[str, ChatOllama] = {}
    
    def get_llm(
        self,
        model_key: str,
        deps: Optional[Dict] = None,
        temperature: float = 0.0,
    ) -> ChatOllama:
        """Get or create an LLM instance."""
        # Resolve model name from deps or defaults
        if deps and model_key in deps:
            model_name = deps[model_key]
        else:
            model_name = self.default_models.get(model_key, self.default_models["llm"])
        
        base_url = deps.get("base_url", self.base_url) if deps else self.base_url
        
        cache_key = f"{model_name}:{base_url}:{temperature}"
        
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
            )
        
        return self._llm_cache[cache_key]
    
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
                state = config.pre_process(state)
            
            # Build prompt variables from state
            prompt_vars = self._extract_prompt_vars(state, config)
            
            # Format user prompt
            try:
                user_prompt = config.user_prompt_template.format(**prompt_vars)
            except KeyError as e:
                logger.warning(f"Missing prompt variable: {e}")
                user_prompt = config.user_prompt_template
            
            # Get LLM
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
                output = config.post_process(output, state)
            
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
    system_prompt="""You are an expert data science planner. Analyze the user's request and create a detailed plan.

Determine what resources are needed:
- requires_code: True if calculations, data processing, or visualizations needed
- requires_web_search: True if external methodology or best practices needed
- requires_rag: True if previous analysis context would help
- requires_plot_analysis: True if existing plots need to be analyzed

Create clear, actionable steps.""",
    output_schema=PlanOutput,
    user_prompt_template="""Create an analysis plan for this request:

Query: {query}

Available context:
{context}

Data file: {data_path}

Return a structured plan with clear steps and resource requirements.""",
)


CODE_GENERATOR_CONFIG = NodeConfig(
    name="code_generator",
    model_key="code_llm",
    system_prompt="""You are an expert Python data scientist. Write clean, executable code.

Requirements:
- Use pandas for data manipulation
- Use matplotlib with 'Agg' backend for plots
- Save all plots with plt.savefig() using descriptive names
- Print key findings to stdout
- Handle errors gracefully with clear messages
- Include docstrings and comments
- Break complex tasks into functions""",
    output_schema=CodeOutput,
    user_prompt_template="""Write Python code to accomplish this task:

Plan: {plan}

Query: {query}

Data file: {data_path}

Previous context:
{context}

Write complete, executable code.""",
)


VERIFIER_CONFIG = NodeConfig(
    name="verifier",
    system_prompt="""You are a quality assurance expert. Evaluate if the analysis output fully addresses the original request.

Check:
1. Does the output answer the original query?
2. Were all planned steps completed?
3. Are the results clear and actionable?
4. Is anything missing?

Be critical but fair. Score quality from 0-1.""",
    output_schema=VerificationOutput,
    user_prompt_template="""Evaluate this analysis output:

ORIGINAL QUERY:
{query}

PLAN:
{plan}

CODE OUTPUT:
{stdout}

ERRORS:
{stderr}

SUMMARY:
{summary}

Does this fully address the request? What's missing?""",
)


SUMMARIZER_CONFIG = NodeConfig(
    name="summarizer",
    system_prompt="""You are an expert data analyst. Create a comprehensive, actionable summary of the analysis results.

Include:
- Key findings and insights
- Specific numbers and metrics
- Actionable recommendations
- Any caveats or limitations

Be thorough but concise. Use the actual data from the output.""",
    user_prompt_template="""Summarize this analysis:

QUERY: {query}

PLAN: {plan}

CODE OUTPUT:
{stdout}

CONTEXT:
{context}

Provide a comprehensive summary with key insights and recommendations.""",
)