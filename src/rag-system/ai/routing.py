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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.messages import SystemMessage, HumanMessage

from .prompts import (
    ROUTER_PLAN, ROUTER_POST_EXECUTION, ROUTER_POST_VERIFICATION,
    ROUTER_PLAN_TEMPLATE, ROUTER_POST_EXECUTION_TEMPLATE, ROUTER_POST_VERIFICATION_TEMPLATE,
)


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

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

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

        plots_info = f"{len(existing_plots)} ({', '.join(existing_plots[:5]) if existing_plots else 'none'})"
        prompt = ROUTER_PLAN_TEMPLATE.format(
            query=query,
            has_data=has_data,
            has_context=bool(context),
            web_enabled=web_enabled,
            plots_info=plots_info,
            context=context[:1500] if context else "No context yet.",
        )

        structured_llm = self.llm.with_structured_output(RoutingPlan)

        try:
            plan = structured_llm.invoke([
                SystemMessage(content=ROUTER_PLAN),
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
        prompt = ROUTER_POST_EXECUTION_TEMPLATE.format(
            query=query,
            plan=plan,
            code_output=code_output[:2000] if code_output else "No output",
            error=error[:500] if error else "None",
            has_plots=has_plots,
        )

        structured_llm = self.llm.with_structured_output(RouteDecision)

        try:
            decision = structured_llm.invoke([
                SystemMessage(content=ROUTER_POST_EXECUTION),
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

        prompt = ROUTER_POST_VERIFICATION_TEMPLATE.format(
            is_complete=is_complete,
            quality_score=quality_score,
            missing_items=missing_items,
            feedback=feedback,
        )

        structured_llm = self.llm.with_structured_output(VerificationRoute)

        try:
            decision = structured_llm.invoke([
                SystemMessage(content=ROUTER_POST_VERIFICATION),
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
