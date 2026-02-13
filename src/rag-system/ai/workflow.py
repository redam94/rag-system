"""
Workflow Graph - Modular workflow with dynamic routing and verification.

Graph Structure:

    ┌─────────────────┐
    │  gather_context │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │      plan       │
    └────────┬────────┘
             │
      ┌──────┴──────────────────┬───────────────┐
      │                         │               │
      ▼                         ▼               ▼
  [execute]                [answer]        [web_search]
      │                         │               │
      ▼                         │               │
 [plot_analysis]                │               │
      │                         │               │
      └──────┬──────────────────┴───────────────┘
             │
    ┌────────▼────────┐
    │    summarize    │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │     verify      │
    └────────┬────────┘
             │
      ┌──────┴──────────────────┬───────────────┐
      │             │           │               │
      ▼             ▼           ▼               ▼
   [done]      [execute]  [gather_context] [web_search]
               (retry)       (retry)         (retry)
"""

from loguru import logger
from langgraph.graph import StateGraph, START, END

from .state import State, Deps
from .nodes import (
    gather_context,
    plan,
    execute,
    analyze_plots,
    summarize,
    answer_from_context,
    verify,
    web_search,
    route_action,
    route_after_verify,
)


def build_workflow_v2() -> StateGraph:
    """Build workflow with verification loop and web search."""
    
    logger.info("🔧 Building workflow v2...")
    
    graph = StateGraph(State, Deps)
    
    # Add nodes
    graph.add_node("gather_context", gather_context)
    graph.add_node("plan_initial", plan)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("analyze_plots", analyze_plots)
    graph.add_node("answer", answer_from_context)
    graph.add_node("summarize", summarize)
    graph.add_node("verify", verify)
    graph.add_node("web_search", web_search)
    
    # Increment retry counter node
    def increment_retry(state: State, runtime) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}
    
    graph.add_node("increment_retry", increment_retry)
    
    # Main flow
    graph.add_edge(START, "plan_initial")
    graph.add_edge("plan_initial", "gather_context")
    graph.add_edge("gather_context", "plan")
    
    # Route based on plan decision
    graph.add_conditional_edges(
        "plan",
        route_action,
        {
            "execute": "execute",
            "answer": "answer",
            "analyze_plots": "analyze_plots",
            "web_search": "web_search",
        }
    )
    
    # Execute -> analyze plots -> summarize
    graph.add_edge("execute", "analyze_plots")
    graph.add_edge("analyze_plots", "summarize")
    
    # Answer -> summarize
    graph.add_edge("answer", "summarize")
    
    # Web search -> summarize (after getting context, summarize findings)
    graph.add_edge("web_search", "summarize")
    
    # Summarize -> verify
    graph.add_edge("summarize", "verify")
    
    # Verify -> done or retry paths
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "done": END,
            "execute": "increment_retry",
            "gather_context": "increment_retry",
            "summarize": "increment_retry",
            "web_search": "increment_retry",
        }
    )
    
    # Retry increments counter then routes to appropriate node
    def route_retry(state: State, runtime) -> str:
        action = state.get("verification", {}).get("suggested_action", "done")
        route_map = {
            "retry_code": "execute",
            "add_context": "gather_context",
            "web_search": "web_search",
            "refine": "summarize",
        }
        return route_map.get(action, "summarize")
    
    graph.add_conditional_edges(
        "increment_retry",
        route_retry,
        {
            "execute": "execute",
            "gather_context": "gather_context",
            "web_search": "web_search",
            "summarize": "summarize",
        }
    )
    
    logger.info("✅ Workflow v2 built with verification loop and web search")
    
    return graph.compile()


def build_simple_workflow() -> StateGraph:
    """Build simplified workflow without verification (for faster iteration)."""
    
    logger.info("🔧 Building simple workflow...")
    
    graph = StateGraph(State, Deps)
    
    # Add nodes
    graph.add_node("gather_context", gather_context)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("analyze_plots", analyze_plots)
    graph.add_node("answer", answer_from_context)
    graph.add_node("summarize", summarize)
    graph.add_node("web_search", web_search)
    
    # Flow
    graph.add_edge(START, "gather_context")
    graph.add_edge("gather_context", "plan")
    
    graph.add_conditional_edges(
        "plan",
        route_action,
        {
            "execute": "execute",
            "answer": "answer",
            "analyze_plots": "analyze_plots",
            "web_search": "web_search",
        }
    )
    
    graph.add_edge("execute", "analyze_plots")
    graph.add_edge("analyze_plots", "summarize")
    graph.add_edge("answer", "summarize")
    graph.add_edge("web_search", "summarize")
    graph.add_edge("summarize", END)
    
    logger.info("✅ Simple workflow built")
    
    return graph.compile()


# Default workflows
workflow = build_workflow_v2()
simple_workflow = build_simple_workflow()