"""
Workflow Nodes - Modular nodes using factory pattern.

Nodes:
- gather_context: RAG, web, outputs, documents
- plan: Create analysis plan with routing decisions
- execute: Generate and run code with file tools
- web_search: Search for methodology guidance
- rag_lookup: Query previous analyses
- plot_analysis: Analyze plots with vision model
- summarize: Create comprehensive summary
- verify: Check output completeness
- answer: Answer from context (no execution)
"""

import asyncio
import base64
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

from .state import State, Deps, Context, DEFAULTS
from .factory import NodeFactory, NodeConfig, CodeOutput
from .tools import WorkflowFileTools, create_file_tools_for_code
from .routing import DynamicRouter, RouteDecision
from .rag_agent_integration import gather_context_with_agent as gather_context

# =============================================================================
# FACTORY INSTANCE
# =============================================================================

_factory: Optional[NodeFactory] = None


def get_factory(deps: Optional[Deps] = None) -> NodeFactory:
    """Get or create the node factory."""
    global _factory
    if _factory is None:
        base_url = deps.get("base_url", DEFAULTS["base_url"]) if deps else DEFAULTS["base_url"]
        _factory = NodeFactory(base_url=base_url)
    return _factory


# =============================================================================
# EMITTER HELPERS
# =============================================================================

def get_emitter(deps: Deps):
    """Get emitter from deps (injected by API)."""
    return deps.get("emitter")


# def _run_async(coro):
#     """
#     Run async coroutine from sync context.
    
#     Handles the case where we're already inside an event loop (FastAPI)
#     by scheduling the task instead of blocking.
#     """
#     try:
#         loop = asyncio.get_running_loop()
#         # Already in async context - schedule as task (fire-and-forget)
#         loop.create_task(coro)
#     except RuntimeError:
#         # No running loop - create one and run
#         asyncio.run(coro)


async def emit_async(deps: Deps, event_type: str, stage: str, message: str, data: Optional[Dict] = None):
    """Emit event asynchronously."""
    emitter = get_emitter(deps)
    if emitter:
        await emitter.emit(event_type, stage, message, data)


async def emit(deps: Deps, stage: str, message: str, data: Optional[Dict] = None):
    """
    Emit progress event (sync wrapper).
    
    Safe to call from both sync and async contexts.
    """
    emitter = get_emitter(deps)
    if emitter:
        from ..utils.progress_events import EventType
        await emitter.emit(str(EventType.PROGRESS), stage, message, data)


async def stage_start(deps: Deps, stage: str, description: str = ""):
    """Signal stage start."""
    emitter = get_emitter(deps)
    if emitter:
        await emitter.stage_start(stage, description)


async def stage_end(deps: Deps, stage: str, success: bool = True):
    """Signal stage end."""
    emitter = get_emitter(deps)
    if emitter:
        await emitter.stage_end(stage, success)


async def stage_start_sync(deps, stage, description):
    emitter = deps.get("emitter")
    if emitter:
        await emitter.stage_start(stage, description)


async def stage_end_sync(deps: Deps, stage: str, success: bool = True):
    """Signal stage end (sync wrapper)."""
    emitter = deps.get("emitter")
    if emitter:
        await emitter.stage_end(stage, success)


# =============================================================================
# LLM HELPERS
# =============================================================================

def get_llm(deps: Deps, model_key: str = "llm") -> ChatOllama:
    """Get configured LLM."""
    factory = get_factory(deps)
    return factory.get_llm(model_key, deps)


def get_query(state: State) -> str:
    """Extract latest user query."""
    messages = state.get("messages", [])
    return messages[-1].content if messages else ""


# =============================================================================
# NODE: GATHER CONTEXT (imported from rag_agent_integration)
# =============================================================================

# gather_context is imported from rag_agent_integration module


# =============================================================================
# NODE: PLAN
# =============================================================================

class PlanOutput(BaseModel):
    """Planning output with routing decisions."""
    plan: str = Field(description="Step-by-step analysis plan")
    steps: List[str] = Field(description="Ordered list of steps")
    action: str = Field(description="Primary action: answer, execute, web_search, plot_analysis")
    requires_code: bool = Field(default=True)
    requires_web: bool = Field(default=False)
    requires_plots: bool = Field(default=False)
    reasoning: str = Field(description="Why this approach")


async def plan(state: State, runtime) -> dict:
    """Create analysis plan and decide primary action."""
    deps = runtime.context
    
    await stage_start_sync(deps, "plan", "Creating analysis plan")
    
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    has_data = bool(state.get("data_path"))
    existing_plots = state.get("context", {}).get("plots", [])
    
    llm = get_llm(deps).with_structured_output(PlanOutput)
    
    prompt = f"""Create an analysis plan for this request.

Query: {query}

Context Available:
{context[:2000]}

Data File: {has_data}
Existing Plots: {len(existing_plots)}

Decide the primary action:
- "answer" ONLY if context fully answers the query (no new analysis)
- "execute" if calculations, visualizations, or data processing needed
- "web_search" if methodology guidance needed first
- "plot_analysis" if existing plots need interpretation

Create specific, actionable steps. Be conservative - prefer execute over answer."""

    await emit(deps, "plan", "🧠 Analyzing query...")
    
    try:
        result = llm.invoke([
            SystemMessage(content="You are a data science planner. Create actionable analysis plans."),
            HumanMessage(content=prompt),
        ])
        
        await emit(deps, "plan", f"📋 Action: {result.action}")
        await stage_end_sync(deps, "plan", success=True)
        
        logger.info(f"🎯 Plan: {result.action} ({len(result.steps)} steps)")
        
        return {
            "plan": result.plan,
            "action": result.action,
            "plan_steps": result.steps,
        }
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        await stage_end_sync(deps, "plan", success=False)
        return {
            "plan": "Execute data analysis",
            "action": "execute",
            "plan_steps": ["Analyze data"],
        }


# =============================================================================
# NODE: EXECUTE (with file tools)
# =============================================================================

async def execute(state: State, runtime) -> dict:
    """Generate code with file tools, execute, and fix errors."""
    deps = runtime.context
    emitter = get_emitter(deps)
    
    await stage_start(deps, "execute", "Generating and executing code")
    
    query = get_query(state)
    data_path = Path(state.get("data_path", ""))
    
    if not data_path.exists():
        await stage_end(deps, "execute", success=False)
        return {"error": "Data file not found", "code": "", "output": None}
    
    output_manager = deps.get("output_manager")
    stage_name = state.get("stage_name", "analysis")
    stage_dir = output_manager.get_stage_dir(stage_name)
    exec_dir = stage_dir / "execution"
    exec_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy data file
    shutil.copy(data_path, exec_dir / data_path.name)
    
    # Get file tools info for the code agent
    file_tools = WorkflowFileTools(output_manager.workflow_dir)
    available_files = file_tools.list_files()
    files_info = file_tools.format_file_listing(available_files, max_files=15)
    
    # Inject file tools into code
    file_tools_code = create_file_tools_for_code(output_manager.workflow_dir)
    
    context = state.get("context", {})
    context_text = context.get("combined", "")[:2000]
    plan = state.get("plan", "")
    
    # Code generation
    await emit_async(deps, "progress", "execute", "✍️ Generating code...")
    
    code_llm = get_llm(deps, "code_llm").with_structured_output(CodeOutput)
    
    code_prompt = f"""Write Python code to analyze '{data_path.name}'.

PLAN:
{plan[:800]}

QUERY:
{query}

CONTEXT:
{context_text}

AVAILABLE FILES IN WORKFLOW:
{files_info}

REQUIREMENTS:
- Load data: pd.read_csv('{data_path.name}')
- Use matplotlib with 'Agg' backend
- Save plots with descriptive names (plt.savefig) at 300 dpi
- Print key findings to stdout
- Use the file tools (list_workflow_files, read_workflow_file) to access previous outputs if needed
- Handle errors with clear messages
- Write modular, documented code

The following file tools are auto-injected:
- list_workflow_files(pattern, stage): List files in workflow
- read_workflow_file(path): Read file content
- get_data_files(stage): Get CSV/JSON files
- get_previous_output(stage): Get console output from a stage"""

    try:
        result = code_llm.invoke([
            SystemMessage(content="Write complete, executable Python code. Use the available file tools to access previous workflow outputs when relevant."),
            HumanMessage(content=code_prompt),
        ])
        
        # Prepend file tools to code
        code = file_tools_code + "\n" + result.code
        
        if emitter:
            await emitter.code_generated("execute", result.code[:300], len(result.code.splitlines()))
        
        logger.info(f"✍️ Code generated ({len(result.code)} chars)")
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        await stage_end(deps, "execute", success=False)
        return {"error": str(e), "code": "", "output": None}
    
    # Execute with retries
    executor = deps.get("executor")
    max_retries = deps.get("max_retries", DEFAULTS["max_retries"])
    output = None
    error = ""
    
    for attempt in range(max_retries + 1):
        await emit_async(deps, "progress", "execute", f"⚡ Attempt {attempt + 1}/{max_retries + 1}")
        
        output = await executor.execute_with_output_manager(
            code=code,
            stage_name=stage_name,
            output_manager=output_manager,
            code_filename=f"code_v{attempt + 1}.py"
        )
        
        if output.success:
            await emit_async(deps, "progress", "execute", "✅ Execution succeeded")
            logger.info("✅ Code executed successfully")
            error = ""
            break
        
        error = output.stderr or output.error or "Unknown error"
        
        if emitter:
            await emitter.execution_result("execute", success=False, output=error[:300])
        
        logger.warning(f"❌ Attempt {attempt + 1} failed: {error[:100]}")
        
        if attempt < max_retries:
            await emit_async(deps, "progress", "execute", "🔧 Fixing code...")
            
            fix_prompt = f"""Fix this code error:

CODE:
```python
{result.code}
```

ERROR:
{error}

OUTPUT:
{output.stdout[:500] if output.stdout else 'None'}

Return the complete fixed code."""

            try:
                fix_result = code_llm.invoke([
                    SystemMessage(content="Fix the Python code error."),
                    HumanMessage(content=fix_prompt),
                ])
                code = file_tools_code + "\n" + fix_result.code
                logger.info("🔧 Code fixed")
            except Exception as e:
                logger.error(f"Fix failed: {e}")
                break
    
    # Store in RAG
    rag = deps.get("rag")
    if output and output.success and rag and rag.enabled:
        rag.add_code_execution(
            code=result.code,
            stdout=output.stdout,
            stderr=output.stderr or "",
            stage_name=stage_name,
            workflow_id=state.get("workflow_id", "default"),
            success=True
        )
    
    await stage_end(deps, "execute", success=(output and output.success))
    
    return {
        "code": result.code,
        "output": output,
        "error": error,
    }


# =============================================================================
# NODE: PLOT ANALYSIS
# =============================================================================

async def analyze_plots(state: State, runtime) -> dict:
    """Analyze plots with vision LLM."""
    deps = runtime.context
    emitter = get_emitter(deps)
    
    await stage_start(deps, "plot_analysis", "Analyzing visualizations")
    
    output_manager = deps.get("output_manager")
    plot_cache = deps.get("plot_cache")
    stage_name = state.get("stage_name", "analysis")
    
    analyses = []
    
    if output_manager:
        try:
            stage_dir = output_manager.get_stage_dir(stage_name)
            plots_dir = stage_dir / "plots"
            
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                
                if plot_files:
                    await emit_async(deps, "progress", "plot_analysis", f"🔍 Analyzing {len(plot_files)} plots")
                    
                    vision_llm = get_llm(deps, "vision_llm")
                    
                    for plot_path in plot_files:
                        # Check cache
                        if plot_cache:
                            cached = plot_cache.get(str(plot_path))
                            if cached:
                                analyses.append(cached)
                                await emit_async(deps, "progress", "plot_analysis", f"⚡ Cached: {plot_path.name}")
                                continue
                        
                        try:
                            await emit_async(deps, "progress", "plot_analysis", f"🔍 {plot_path.name}")
                            
                            with open(plot_path, "rb") as f:
                                img_data = base64.b64encode(f.read()).decode()
                            
                            response = vision_llm.invoke([
                                SystemMessage(content="Describe key insights from this visualization. Be specific about trends, patterns, and notable values."),
                                HumanMessage(content=[
                                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_data}"},
                                    {"type": "text", "text": "Analyze this plot. Focus on actionable insights."}
                                ])
                            ])
                            
                            analysis = {
                                "plot": plot_path.name,
                                "path": str(plot_path),
                                "analysis": response.content
                            }
                            analyses.append(analysis)
                            
                            # Cache and store in RAG
                            if plot_cache:
                                plot_cache.set(str(plot_path), analysis)
                            
                            rag = deps.get("rag")
                            if rag and rag.enabled:
                                rag.add_plot_analysis(
                                    plot_name=plot_path.name,
                                    plot_path=str(plot_path),
                                    analysis=response.content,
                                    stage_name=stage_name,
                                    workflow_id=state.get("workflow_id", "default")
                                )
                            
                        except Exception as e:
                            logger.warning(f"Plot analysis failed for {plot_path}: {e}")
                            
        except Exception as e:
            logger.warning(f"Plot gathering failed: {e}")
    
    await stage_end(deps, "plot_analysis", success=True)
    
    logger.info(f"📊 Analyzed {len(analyses)} plots")
    
    return {"plot_analyses": analyses}


# =============================================================================
# NODE: WEB SEARCH
# =============================================================================

async def web_search(state: State, runtime) -> dict:
    """
    Search web for additional methodology guidance or context.
    
    Used when:
    - Verification determines more context is needed
    - Plan decides web research would help
    - User explicitly requests web search
    
    Results are stored in RAG for future retrieval.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    await stage_start(deps, "web_search", "Searching web for additional context")
    
    query = get_query(state)
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "analysis")
    
    # Build search query from original query + any gaps identified
    verification = state.get("verification", {})
    missing_info = verification.get("missing_info", [])
    
    if missing_info:
        search_query = f"{query} {' '.join(missing_info[:2])}"
    else:
        search_query = query
    
    await emit_async(deps, "progress", "web_search", f"🌐 Searching: {search_query[:50]}...")
    
    try:
        from .web_search import search_and_synthesize
        
        llm_model = deps.get("llm", DEFAULTS["llm"])
        base_url = deps.get("base_url", DEFAULTS["base_url"])
        
        async def on_progress(msg: str):
            await emit_async(deps, "progress", "web_search", f"🌐 {msg}")
        
        # Use sync callback wrapper since search_and_synthesize expects sync
        def sync_progress(msg: str):
            emitter.emit("progress", "web_search", f"🌐 {msg}")
        
        search_result = await search_and_synthesize(
            query=search_query,
            context=state.get("context", {}).get("combined", "")[:1000],
            llm_model=llm_model,
            base_url=base_url,
            enrich_with_crawl=True,
            max_crawl_urls=3,
            on_progress=sync_progress
        )
        
        web_context = ""
        results_count = 0
        
        if search_result.get("results"):
            results_count = len(search_result["results"])
            web_context = search_result.get("formatted_text", "")
            
            await emit_async(
                deps, "progress", "web_search", 
                f"✅ Found {results_count} results"
            )
            
            # Store in RAG for future use
            rag = deps.get("rag")
            if rag and rag.enabled:
                web_results = [
                    {
                        "url": r.url,
                        "title": r.title,
                        "content": r.content,
                        "score": getattr(r, "score", 0.5),
                        "source": getattr(r, "source", "web"),
                        "enriched": "crawl4ai" in getattr(r, "source", "").lower(),
                        "query_used": getattr(r, "query_used", search_query)
                    }
                    for r in search_result["results"]
                ]
                rag.add_web_search_batch(
                    query=search_query,
                    results=web_results,
                    stage_name=stage_name,
                    workflow_id=workflow_id,
                )
                logger.info(f"📚 Stored {len(web_results)} web results in RAG")
        else:
            await emit_async(deps, "progress", "web_search", "⚠️ No results found")
        
        await stage_end(deps, "web_search", success=bool(web_context))
        
        logger.info(f"🌐 Web search complete: {results_count} results")
        
        # Update context with web results
        current_context = state.get("context", {})
        current_context["web"] = web_context
        
        # Rebuild combined context
        parts = []
        if current_context.get("rag"):
            parts.append(f"[Previous Analysis]\n{current_context['rag']}")
        if web_context:
            parts.append(f"[Web Research]\n{web_context}")
        if current_context.get("outputs"):
            parts.append(f"[Previous Output]\n{current_context['outputs']}")
        
        current_context["combined"] = "\n\n".join(parts) if parts else current_context.get("combined", "")
        
        return {
            "context": current_context,
            "web_search_results": search_result.get("results", []),
            "web_search_count": results_count,
        }
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        await stage_end(deps, "web_search", success=False)
        
        if emitter:
            await emitter.error("web_search", str(e))
        
        return {
            "web_search_error": str(e),
            "web_search_count": 0,
        }


# =============================================================================
# NODE: SUMMARIZE
# =============================================================================

async def summarize(state: State, runtime) -> dict:
    """Create comprehensive summary from all results."""
    deps = runtime.context
    
    await stage_start(deps, "summarize", "Creating summary")
    
    query = get_query(state)
    
    # Gather all results
    output = state.get("output")
    stdout = output.stdout if output and hasattr(output, "stdout") else ""
    
    context = state.get("context", {})
    plot_analyses = state.get("plot_analyses", [])
    
    # Build summary input
    parts = []
    
    if stdout:
        parts.append(f"Code Output:\n{stdout[:2000]}")
    
    if plot_analyses:
        plots_text = "\n".join(
            f"- {p['plot']}: {p['analysis'][:300]}"
            for p in plot_analyses if 'plot' in p and 'analysis' in p
        )
        parts.append(f"Plot Analyses:\n{plots_text}")
    
    if context.get("rag"):
        parts.append(f"Previous Context:\n{context['rag'][:500]}")
    
    if context.get("web"):
        parts.append(f"Research:\n{context['web'][:300]}")
    
    results_text = "\n\n".join(parts) if parts else "No results available."
    
    await emit_async(deps, "progress", "summarize", "📝 Generating summary...")
    
    llm = get_llm(deps)
    
    prompt = f"""Create a comprehensive answer to the user's query.

QUERY: {query}

PLAN: {state.get("plan", "")[:500]}

RESULTS:
{results_text}

Be thorough, specific, and actionable. Reference actual findings from the results.
Include specific numbers and metrics where available."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a data scientist providing analysis results. Be specific and data-driven."),
            HumanMessage(content=prompt),
        ])
        
        summary = response.content
        
        # Store in RAG
        rag = deps.get("rag")
        if rag and rag.enabled:
            rag.add_summary(
                summary=summary,
                stage_name=state.get("stage_name", "analysis"),
                workflow_id=state.get("workflow_id", "default")
            )
        
        await stage_end(deps, "summarize", success=True)
        
        logger.info("✅ Summary complete")
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        await stage_end(deps, "summarize", success=False)
        return {"summary": f"Summary generation failed: {e}"}


# =============================================================================
# NODE: ANSWER (from context only)
# =============================================================================

async def answer_from_context(state: State, runtime) -> dict:
    """Answer directly from gathered context."""
    deps = runtime.context
    
    await stage_start_sync(deps, "answer", "Answering from context")
    
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    
    llm = get_llm(deps)
    
    await emit(deps, "answer", "💬 Generating answer...")
    
    prompt = f"""Answer the user's query using the available context.

Query: {query}

Context:
{context[:3000]}

Plan: {state.get("plan", "")}

Provide a clear, helpful answer based on the context."""

    try:
        response = llm.invoke([
            SystemMessage(content="Answer based on the provided context."),
            HumanMessage(content=prompt),
        ])
        
        await stage_end_sync(deps, "answer", success=True)
        
        return {"summary": response.content}
        
    except Exception as e:
        logger.error(f"Answer failed: {e}")
        await stage_end_sync(deps, "answer", success=False)
        return {"summary": f"Failed to generate answer: {e}"}


# =============================================================================
# NODE: VERIFY
# =============================================================================

async def verify(state: State, runtime) -> dict:
    """Verify output completeness."""
    from .verification import verify_output
    return await verify_output(state, runtime)


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_action(state: State, runtime) -> str:
    """Route based on plan decision."""
    action = state.get("action", "execute")
    
    route_map = {
        "answer": "answer",
        "execute": "execute",
        "web_search": "web_search",
        "plot_analysis": "analyze_plots",
    }
    
    result = route_map.get(action, "execute")
    logger.info(f"➡️ Routing to: {result}")
    return result


def route_after_verify(state: State, runtime) -> str:
    """Route based on verification results."""
    verification = state.get("verification", {})
    
    is_complete = verification.get("is_complete", True)
    quality = verification.get("quality_score", 1.0)
    action = verification.get("suggested_action", "done")
    
    # Check retry count to prevent infinite loops
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    
    if retry_count >= max_retries:
        logger.info("➡️ Max retries reached, completing")
        return "done"
    
    if is_complete and quality >= 0.7:
        logger.info("➡️ Verification passed")
        return "done"
    
    if action == "retry_code":
        logger.info("➡️ Retrying code execution")
        return "execute"
    elif action == "add_context":
        logger.info("➡️ Adding more context")
        return "gather_context"
    elif action == "web_search":
        logger.info("➡️ Searching web for more info")
        return "web_search"
    elif action == "refine":
        logger.info("➡️ Refining summary")
        return "summarize"
    
    return "done"