"""
Dynamic Router - Intelligent routing for workflow steps.

Routes to:
- code_execution: When calculations, data processing, or visualizations needed
- web_search: When external methodology or information needed
- rag_lookup: When previous analysis context would help
- plot_analysis: When existing plots need interpretation
- answer: When context is sufficient to answer directly
- verify: After execution to check completeness
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from loguru import logger

from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage


# =============================================================================
# ROUTING MODELS
# =============================================================================

class RouteDecision(BaseModel):
    """Single routing decision."""
    route: Literal[
        "code_execution",
        "web_search",
        "rag_lookup",
        "plot_analysis",
        "answer",
        "verify",
        "done",
    ]
    reasoning: str
    priority: int = Field(default=1, ge=1, le=10)


class RoutingPlan(BaseModel):
    """Multi-step routing plan."""
    routes: List[RouteDecision]
    parallel_possible: bool = Field(
        default=False,
        description="Whether some routes can run in parallel"
    )


class VerificationRoute(BaseModel):
    """Routing decision after verification."""
    action: Literal["done", "retry_code", "add_web", "add_rag", "refine_summary"]
    reasoning: str
    specific_instruction: str = ""


# =============================================================================
# ROUTER CLASS
# =============================================================================

class DynamicRouter:
    """
    Routes workflow steps based on current state and requirements.
    
    Uses LLM to make intelligent routing decisions based on:
    - User query
    - Available context
    - Current state
    - Previous results
    """
    
    def __init__(
        self,
        llm_model: str = "qwen3:30b",
        base_url: str = "http://100.91.155.118:11434",
    ):
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0,
            base_url=base_url,
        )
    
    def plan_routes(
        self,
        query: str,
        context: str = "",
        has_data: bool = False,
        existing_plots: List[str] = None,
        web_enabled: bool = True,
        rag_enabled: bool = True,
    ) -> RoutingPlan:
        """
        Create a routing plan for the query.
        
        Decides which steps are needed and in what order.
        """
        existing_plots = existing_plots or []
        
        prompt = f"""Analyze this data analysis request and determine what steps are needed.

QUERY: {query}

AVAILABLE RESOURCES:
- Data file available: {has_data}
- RAG context available: {bool(context)}
- Web search enabled: {web_enabled}
- Existing plots: {len(existing_plots)} ({', '.join(existing_plots[:5]) if existing_plots else 'none'})

EXISTING CONTEXT:
{context[:1500] if context else 'No context yet.'}

AVAILABLE ROUTES:
- rag_lookup: Query previous analysis for relevant context
- web_search: Search web for methodology/best practices
- code_execution: Run Python code for calculations/visualizations
- plot_analysis: Analyze existing plots with vision model
- answer: Answer directly from available context (only if NO new analysis needed)

RULES:
1. If query asks for calculations, stats, or new visualizations → code_execution required
2. If methodology guidance would help → web_search
3. If previous analysis is relevant → rag_lookup
4. If plots exist and need interpretation → plot_analysis
5. answer only works if context fully addresses query with no new work needed
6. Order routes by priority (gather context first, then execute, then analyze)

Return the routing plan with ordered steps."""

        structured_llm = self.llm.with_structured_output(RoutingPlan)
        
        try:
            plan = structured_llm.invoke([
                SystemMessage(content="You are a workflow router. Determine the optimal sequence of steps to answer data analysis queries."),
                HumanMessage(content=prompt),
            ])
            
            logger.info(f"📍 Route plan: {[r.route for r in plan.routes]}")
            return plan
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Default fallback plan
            routes = []
            if rag_enabled:
                routes.append(RouteDecision(route="rag_lookup", reasoning="Default: check context", priority=1))
            if has_data:
                routes.append(RouteDecision(route="code_execution", reasoning="Default: data available", priority=2))
            if not routes:
                routes.append(RouteDecision(route="answer", reasoning="Fallback", priority=1))
            
            return RoutingPlan(routes=routes)
    
    def route_after_execution(
        self,
        query: str,
        plan: str,
        code_output: str,
        error: str = "",
        has_plots: bool = False,
    ) -> RouteDecision:
        """
        Decide next step after code execution.
        """
        prompt = f"""Code execution completed. Decide next step.

QUERY: {query}
PLAN: {plan}

EXECUTION OUTPUT:
{code_output[:2000] if code_output else 'No output'}

ERRORS:
{error[:500] if error else 'None'}

PLOTS GENERATED: {has_plots}

OPTIONS:
- plot_analysis: If plots were generated and need interpretation
- verify: If execution succeeded and results should be verified
- code_execution: If there were errors that need fixing
- done: If everything is complete (rare - usually verify first)

What should happen next?"""

        structured_llm = self.llm.with_structured_output(RouteDecision)
        
        try:
            decision = structured_llm.invoke([
                SystemMessage(content="Decide the next workflow step after code execution."),
                HumanMessage(content=prompt),
            ])
            
            logger.info(f"📍 Post-execution route: {decision.route}")
            return decision
            
        except Exception as e:
            logger.error(f"Post-execution routing failed: {e}")
            if error:
                return RouteDecision(route="code_execution", reasoning="Error occurred, retry")
            elif has_plots:
                return RouteDecision(route="plot_analysis", reasoning="Plots need analysis")
            else:
                return RouteDecision(route="verify", reasoning="Check results")
    
    def route_after_verification(
        self,
        is_complete: bool,
        missing_items: List[str],
        quality_score: float,
        feedback: str,
    ) -> VerificationRoute:
        """
        Decide action after verification.
        """
        if is_complete and quality_score >= 0.8:
            return VerificationRoute(
                action="done",
                reasoning="Verification passed",
            )
        
        prompt = f"""Verification found issues. Decide corrective action.

COMPLETE: {is_complete}
QUALITY SCORE: {quality_score}
MISSING ITEMS: {missing_items}
FEEDBACK: {feedback}

OPTIONS:
- done: Accept as good enough (quality > 0.6 and mostly complete)
- retry_code: Re-run code with fixes
- add_web: Search for additional methodology guidance
- add_rag: Get more context from previous analyses
- refine_summary: Just improve the summary wording

What action should be taken?"""

        structured_llm = self.llm.with_structured_output(VerificationRoute)
        
        try:
            decision = structured_llm.invoke([
                SystemMessage(content="Decide corrective action after verification."),
                HumanMessage(content=prompt),
            ])
            
            logger.info(f"📍 Post-verification action: {decision.action}")
            return decision
            
        except Exception as e:
            logger.error(f"Post-verification routing failed: {e}")
            if quality_score >= 0.6:
                return VerificationRoute(action="done", reasoning="Acceptable quality")
            else:
                return VerificationRoute(action="retry_code", reasoning="Quality too low")


# =============================================================================
# SIMPLE ROUTING FUNCTIONS (for LangGraph conditional edges)
# =============================================================================

def route_from_plan(state: dict) -> str:
    """
    Route based on planned action in state.
    
    Used as conditional edge function.
    """
    action = state.get("action", "execute")
    
    # Map plan actions to node names
    route_map = {
        "answer": "answer",
        "execute": "execute",
        "code_execution": "execute",
        "web_search": "web_search",
        "rag_lookup": "rag_lookup",
        "plot_analysis": "plot_analysis",
    }
    
    return route_map.get(action, "execute")


def route_from_verification(state: dict) -> str:
    """
    Route based on verification results.
    """
    verification = state.get("verification", {})
    
    if verification.get("is_complete") and verification.get("quality_score", 0) >= 0.7:
        return "done"
    
    action = verification.get("suggested_action", "done")
    
    action_map = {
        "done": "done",
        "retry_code": "execute",
        "add_context": "gather_context",
        "refine": "summarize",
    }
    
    return action_map.get(action, "done")


def should_verify(state: dict) -> bool:
    """Check if verification should run."""
    # Skip verification if already verified or no execution happened
    if state.get("verified"):
        return False
    
    # Verify if we have output or summary
    return bool(state.get("output") or state.get("summary"))


def needs_more_context(state: dict) -> bool:
    """Check if more context gathering is needed."""
    context = state.get("context", {})
    
    # Check if context is empty or minimal
    combined = context.get("combined", "") if isinstance(context, dict) else str(context)
    
    return len(combined) < 100