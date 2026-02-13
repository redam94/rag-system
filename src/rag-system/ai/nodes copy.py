"""
Workflow Nodes - All node functions with progress streaming.

Simplified to 4 main nodes:
1. gather_context - Get RAG, web, existing outputs (stores web results in RAG)
2. plan_and_decide - Create plan, decide action
3. execute - Generate code, run it, fix errors
4. summarize - Analyze results, create summary
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
from langgraph.runtime import Runtime

from .state import State, Deps, Context, DEFAULTS


# =============================================================================
# HELPERS
# =============================================================================

def get_llm(deps: Deps, model_key: str = "llm") -> ChatOllama:
    """Get configured LLM."""
    model = deps.get(model_key, DEFAULTS.get(model_key, "qwen3:30b"))
    base_url = deps.get("base_url", DEFAULTS["base_url"])
    return ChatOllama(model=model, temperature=0, base_url=base_url)


def get_query(state: State) -> str:
    """Extract latest user query."""
    messages = state.get("messages", [])
    return messages[-1].content if messages else ""


def get_emitter(deps: Deps):
    """Get progress emitter from deps if available."""
    return deps.get("progress_emitter")


def emit_progress(deps: Deps, stage: str, message: str, data: Optional[Dict] = None):
    """Emit progress event if emitter available."""
    emitter = get_emitter(deps)
    if emitter:
        from utils.progress_events import EventType
        emitter.emit(EventType.PROGRESS, stage, message, data)


# =============================================================================
# NODE 1: GATHER CONTEXT
# =============================================================================

async def gather_context(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Gather all context in parallel: RAG, web search, existing outputs, documents.
    
    Web search results are stored in RAG for future retrieval.
    Uploaded documents are also queried for relevant context.
    
    Returns combined context for use in planning and code generation.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("gather_context", "Gathering context from RAG, web, documents, and existing outputs")
    
    query = get_query(state)
    workflow_id = state.get("workflow_id", "default")
    stage_name = state.get("stage_name", "analysis")
    
    context = Context(rag="", web="", outputs="", plots=[], combined="")
    parts = []
    
    # 1. RAG Context (includes previous web results and analysis)
    rag = deps.get("rag")
    if rag and rag.enabled:
        try:
            if emitter:
                emitter.rag_query("gather_context", 0)
            
            rag_summary = rag.get_context_summary(
                query=query,
                workflow_id=workflow_id,
                max_tokens=1500
            )
            if rag_summary:
                context["rag"] = rag_summary
                parts.append(f"[Previous Analysis]\n{rag_summary}")
                
                if emitter:
                    emitter.rag_query("gather_context", len(rag_summary.split('\n')))
                
                logger.info(f"📚 RAG: {len(rag_summary)} chars")
        except Exception as e:
            logger.warning(f"RAG failed: {e}")
        
        # 1b. Query uploaded documents separately for focused retrieval
        try:
            doc_results = rag.query_documents(
                query=query,
                workflow_id=workflow_id,
                n_results=5
            )
            if doc_results:
                doc_texts = []
                for doc in doc_results:
                    title = doc['metadata'].get('title', 'Unknown')
                    content = doc['document'][:500]
                    doc_texts.append(f"[{title}]: {content}")
                
                doc_context = "\n\n".join(doc_texts)
                parts.append(f"[Reference Documents]\n{doc_context}")
                
                if emitter:
                    emit_progress(deps, "gather_context", f"📄 Found {len(doc_results)} relevant document chunks")
                
                logger.info(f"📄 Documents: {len(doc_results)} chunks")
        except Exception as e:
            logger.warning(f"Document query failed: {e}")
    
    # 2. Web Search (if enabled) - Results are stored in RAG
    if state.get("web_search_enabled"):
        try:
            from .web_search import search_and_synthesize
            
            def on_search_progress(msg: str):
                if emitter:
                    emit_progress(deps, "gather_context", f"🌐 {msg}")
            
            llm_model = deps.get("llm", DEFAULTS["llm"])
            base_url = deps.get("base_url", DEFAULTS["base_url"])
            
            search_result = await search_and_synthesize(
                query=query,
                context=context.get("rag", ""),
                llm_model=llm_model,
                base_url=base_url,
                on_progress=on_search_progress
            )
            
            if search_result["results"]:
                context["web"] = search_result["formatted_text"]
                parts.append(f"[Web Research]\n{search_result['formatted_text']}")
                
                # Store web results in RAG for future queries
                if rag and rag.enabled:
                    try:
                        web_results_for_rag = [
                            {
                                "url": r.url,
                                "title": r.title,
                                "content": r.content,
                                "score": r.score,
                                "source": r.source,
                                "enriched": "crawl4ai" in r.source.lower(),
                                "query_used": r.query_used
                            }
                            for r in search_result["results"]
                        ]
                        rag.add_web_search_batch(
                            query=query,
                            results=web_results_for_rag,
                            stage_name=stage_name,
                            workflow_id=workflow_id,
                            metadata={
                                "queries_used": ", ".join(search_result["queries_used"]),
                                "relevance_score": search_result["synthesis"].relevance_score
                            }
                        )
                        
                        enriched_count = sum(1 for r in web_results_for_rag if r["enriched"])
                        if emitter:
                            emit_progress(deps, "gather_context", 
                                f"💾 Stored {len(web_results_for_rag)} web results in RAG ({enriched_count} enriched)")
                        
                        logger.info(f"💾 Stored {len(web_results_for_rag)} web results in RAG ({enriched_count} enriched)")
                    except Exception as e:
                        logger.warning(f"Failed to store web results in RAG: {e}")
                
                if emitter:
                    queries_count = len(search_result["queries_used"])
                    results_count = len(search_result["results"])
                    relevance = search_result["synthesis"].relevance_score
                    emitter.web_search("gather_context", f"{queries_count} queries", results_count)
                    emit_progress(deps, "gather_context", 
                        f"📊 Web search: {results_count} results, {relevance:.0%} relevance")
                
                logger.info(f"🌐 Web: {len(search_result['results'])} results from {len(search_result['queries_used'])} queries")
                
            if search_result.get("error"):
                logger.warning(f"Web search had errors: {search_result['error']}")
                
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            if emitter:
                emit_progress(deps, "gather_context", f"⚠️ Web search failed: {e}")
    
    # 3. Existing Outputs
    output_manager = deps.get("output_manager")
    if output_manager and state.get("stage_name"):
        try:
            stage_dir = output_manager.get_stage_dir(state["stage_name"])
            
            plots_dir = stage_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                if plot_files:
                    context["plots"] = [str(p) for p in plot_files]
                    parts.append(f"[Existing Plots: {len(plot_files)}]")
                    
                    if emitter:
                        emit_progress(deps, "gather_context", f"📊 Found {len(plot_files)} existing plots")
            
            console_files = list(stage_dir.glob("console_output_*.txt"))
            if console_files:
                latest = sorted(console_files)[-1]
                with open(latest) as f:
                    output_text = f.read()[:1000]
                context["outputs"] = output_text
                parts.append(f"[Previous Output]\n{output_text}")
                
        except Exception as e:
            logger.warning(f"Output check failed: {e}")
    
    context["combined"] = "\n\n".join(parts) if parts else "No prior context."
    
    if emitter:
        emitter.stage_end("gather_context", success=True)
    
    logger.info(f"📦 Context assembled: {len(context['combined'])} chars")
    
    return {"context": context}


# =============================================================================
# NODE 2: PLAN AND DECIDE
# =============================================================================

class PlanDecision(BaseModel):
    """Combined plan and action decision."""
    plan: str = Field(description="Step-by-step analysis plan")
    action: str = Field(description="'answer' if context sufficient, 'execute' if code needed")
    reasoning: str = Field(description="Why this action")


def plan_and_decide(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Create analysis plan and decide: answer from context or execute code.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("plan_and_decide", "Creating analysis plan")
    
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    has_data = bool(state.get("data_path"))
    
    llm = get_llm(deps).with_structured_output(PlanDecision)
    
    prompt = f"""Create an analysis plan and decide the best action.

User Query: {query}

Available Context:
{context[:2000]}

Data File Available: {has_data}

Rules:
- action="answer" ONLY if context fully answers the query (no new analysis needed)
- action="execute" if ANY calculation, visualization, or new analysis is needed
- action="execute" if user asks to redo, repeat, or create something new
- When uncertain, choose "execute"

Create a specific, actionable plan."""

    if emitter:
        emit_progress(deps, "plan_and_decide", "🧠 Analyzing query and context...")

    result = llm.invoke([
        SystemMessage(content="You are a data scientist. Plan the analysis and decide the action."),
        HumanMessage(content=prompt)
    ])
    
    if emitter:
        emit_progress(deps, "plan_and_decide", f"📋 Plan created: {result.action}")
        emit_progress(deps, "plan_and_decide", f"Strategy: {result.reasoning[:100]}...")
        emitter.stage_end("plan_and_decide", success=True)
    
    logger.info(f"🎯 Action: {result.action} ({result.reasoning[:50]}...)")
    
    return {
        "plan": result.plan,
        "action": result.action
    }


# =============================================================================
# NODE 3: EXECUTE (Code Gen + Run + Fix Loop)
# =============================================================================

class GeneratedCode(BaseModel):
    """Generated Python code."""
    code: str = Field(description="Complete Python code")
    reasoning: str = Field(description="Approach explanation")


async def execute(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Generate code, execute it, and fix errors (up to max_retries).
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("execute", "Generating and executing code")
    
    query = get_query(state)
    data_path = Path(state.get("data_path", ""))
    
    if not data_path.exists():
        if emitter:
            emitter.stage_end("execute", success=False)
        return {"error": "Data file not found", "code": "", "output": None}
    
    output_manager = deps.get("output_manager")
    stage_dir = output_manager.get_stage_dir(state.get("stage_name", "analysis"))
    exec_dir = stage_dir / "execution"
    exec_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(data_path, exec_dir / data_path.name)
    
    context = state.get("context", {})
    context_text = context.get("combined", "")[:2000]
    
    code_llm = get_llm(deps, "code_llm").with_structured_output(GeneratedCode)
    
    if emitter:
        emit_progress(deps, "execute", "✍️ Generating code...")
    
    code_prompt = f"""Write Python code to analyze '{data_path.name}'.

Plan:
{state.get("plan", "")[:800]}

Context:
{context_text}

Requirements:
- Load: pd.read_csv('{data_path.name}')
- Use matplotlib with 'Agg' backend
- Save plots with descriptive names (plt.savefig) use a dpi of 300
- Print key findings to stdout
- Raise exceptions on errors with clear messages
- Be sure to raise errors if any issues occur
- Write clear, documented code
- Break tasks into atomic modular functions that are composed in a main() function to produce the final output
"""

    result = code_llm.invoke([
        SystemMessage(content="Write complete, executable Python code."),
        HumanMessage(content=code_prompt)
    ])
    
    code = result.code
    
    if emitter:
        emitter.code_generated("execute", code[:300])
        emit_progress(deps, "execute", f"📝 Code generated ({len(code)} chars)")
    
    logger.info(f"✍️ Code generated ({len(code)} chars)")
    
    executor = deps.get("executor")
    max_retries = deps.get("max_retries", DEFAULTS["max_retries"])
    output = None
    error = ""
    
    for attempt in range(max_retries + 1):
        if emitter:
            emit_progress(deps, "execute", f"⚡ Execution attempt {attempt + 1}/{max_retries + 1}")
        
        logger.info(f"⚡ Attempt {attempt + 1}/{max_retries + 1}")
        
        output = await executor.execute_with_output_manager(
            code=code,
            stage_name=state.get("stage_name", "analysis"),
            output_manager=output_manager,
            code_filename=f"code_v{attempt + 1}.py"
        )
        
        if output.success:
            if emitter:
                emitter.code_output("execute", output.stdout[:300] if output.stdout else "No output")
                emit_progress(deps, "execute", "✅ Execution succeeded!")
            logger.info("✅ Execution succeeded")
            error = ""
            break
        
        error = output.stderr or output.error or "Unknown error"
        
        if emitter:
            emitter.code_output("execute", error[:300], is_error=True)
            emit_progress(deps, "execute", f"❌ Attempt {attempt + 1} failed")
        
        logger.warning(f"❌ Failed: {error[:100]}...")
        
        if attempt < max_retries:
            if emitter:
                emit_progress(deps, "execute", "🔧 Attempting to fix code...")
            
            fix_prompt = f"""Fix this code error:

Code:
```python
{code}
```

Error:
{error}

Output so far:
{output.stdout[:500] if output.stdout else 'None'}

Return the complete fixed code."""

            try:
                fix_result = code_llm.invoke([
                    SystemMessage(content="Fix the Python code error."),
                    HumanMessage(content=fix_prompt)
                ])
                code = fix_result.code
                
                if emitter:
                    emit_progress(deps, "execute", "🔧 Code fixed, retrying...")
                
                logger.info("🔧 Code fixed, retrying...")
            except Exception as e:
                logger.error(f"Fix failed: {e}")
                break
    
    # Store in RAG if successful
    rag = deps.get("rag")
    if output and output.success and rag and rag.enabled:
        rag.add_code_execution(
            code=code,
            stdout=output.stdout,
            stderr=output.stderr or "",
            stage_name=state.get("stage_name", "analysis"),
            workflow_id=state.get("workflow_id", "default"),
            success=True
        )
    
    if emitter:
        emitter.stage_end("execute", success=(output and output.success))
    
    return {
        "code": code,
        "output": output,
        "error": error
    }


# =============================================================================
# NODE 4: SUMMARIZE (Analyze Plots + Create Summary)
# =============================================================================

async def summarize(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Analyze any plots with vision LLM, then create comprehensive summary.
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("summarize", "Analyzing results and creating summary")
    
    query = get_query(state)

    rag = deps.get("rag")

    output = state.get("output")
    stdout = output.stdout if output and hasattr(output, 'stdout') else ""
    context = state.get("context", {})
    
    plot_analyses = []
    output_manager = deps.get("output_manager")
    plot_cache = deps.get("plot_cache")
    
    if output_manager and state.get("stage_name"):
        try:
            stage_dir = output_manager.get_stage_dir(state["stage_name"])
            plots_dir = stage_dir / "plots"
            
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                
                if plot_files:
                    if emitter:
                        emit_progress(deps, "summarize", f"🔍 Analyzing {len(plot_files)} plots...")
                    
                    vision_llm = get_llm(deps, "vision_llm")
                    
                    for plot_path in plot_files:
                        if plot_cache:
                            cached = plot_cache.get(str(plot_path))
                            if cached:
                                plot_analyses.append(cached)
                                if emitter:
                                    emit_progress(deps, "summarize", f"⚡ Cache hit: {plot_path.name}")
                                continue
                        
                        try:
                            if emitter:
                                emit_progress(deps, "summarize", f"🔍 Analyzing: {plot_path.name}")
                            
                            with open(plot_path, "rb") as f:
                                img_data = base64.b64encode(f.read()).decode()
                            
                            response = vision_llm.invoke([
                                SystemMessage(content="Describe key insights from this visualization concisely."),
                                HumanMessage(content=[
                                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_data}"},
                                    {"type": "text", "text": "Analyze this plot."}
                                ])
                            ])
                            
                            analysis = {
                                "plot": plot_path.name,
                                "analysis": response.content
                            }
                            plot_analyses.append(analysis)
                            
                            if rag and rag.enabled:
                                rag.add_plot_analysis(
                                    plot_name=plot_path.name,
                                    plot_path=str(plot_path),
                                    analysis=response.content,
                                    stage_name=state.get("stage_name", "analysis"),
                                    workflow_id=state.get("workflow_id", "default")
                                )
                            
                            if plot_cache:
                                plot_cache.set(str(plot_path), analysis)
                            
                            if emitter:
                                emitter.plot_analyzed("summarize", plot_path.name, response.content[:100])
                            
                            logger.info(f"🔍 Analyzed: {plot_path.name}")
                            
                        except Exception as e:
                            logger.warning(f"Plot analysis failed for {plot_path}: {e}")
                            
        except Exception as e:
            logger.warning(f"Plot gathering failed: {e}")
    
    if emitter:
        emit_progress(deps, "summarize", "📝 Generating comprehensive summary...")
    
    summary_parts = []
    
    if stdout:
        summary_parts.append(f"Code Output:\n{stdout[:2000]}")
    
    if plot_analyses:
        plots_text = "\n".join(f"- {p['plot']}: {p['analysis'][:300]}" for p in plot_analyses if 'plot' in p and 'analysis' in p)
        summary_parts.append(f"Plot Analyses:\n{plots_text}")
    
    if context.get("rag"):
        summary_parts.append(f"Previous Context:\n{context['rag'][:500]}")
    
    if context.get("web"):
        summary_parts.append(f"Research:\n{context['web'][:300]}")
    
    results_context = "\n\n".join(summary_parts) if summary_parts else "No results available."
    
    llm = get_llm(deps)
    
    summary_prompt = f"""Provide a comprehensive answer to the user's query.

Query: {query}

Plan:
{state.get("plan", "")[:500]}

Results:
{results_context}

Be thorough, specific, and actionable. Reference specific findings from the results."""

    response = llm.invoke([
        SystemMessage(content="You are a data scientist providing analysis results."),
        HumanMessage(content=summary_prompt)
    ])
    
    if rag and rag.enabled:
        rag.add_summary(
            summary=response.content,
            stage_name=state.get("stage_name", "analysis"),
            workflow_id=state.get("workflow_id", "default")
        )
    
    if emitter:
        emitter.stage_end("summarize", success=True)
    
    logger.info("✅ Summary complete")
    
    return {"summary": response.content}


# =============================================================================
# SIMPLE ANSWER NODE (when no execution needed)
# =============================================================================

def answer_from_context(state: State, runtime: Runtime[Deps]) -> dict:
    """
    Answer directly from gathered context (no code execution).
    """
    deps = runtime.context
    emitter = get_emitter(deps)
    
    if emitter:
        emitter.stage_start("answer", "Answering from existing context")
    
    query = get_query(state)
    context = state.get("context", {}).get("combined", "")
    
    llm = get_llm(deps)
    
    if emitter:
        emit_progress(deps, "answer", "💬 Generating answer from context...")
    
    prompt = f"""Answer the user's query using the available context.

Query: {query}

Context:
{context[:3000]}

Plan:
{state.get("plan", "")}

Provide a clear, helpful answer based on the context."""

    response = llm.invoke([
        SystemMessage(content="Answer based on the provided context."),
        HumanMessage(content=prompt)
    ])
    
    if emitter:
        emitter.stage_end("answer", success=True)
    
    return {"summary": response.content}


# =============================================================================
# ROUTING
# =============================================================================

def route_action(state: State, runtime: Runtime[Deps]) -> str:
    """Route based on plan decision."""
    action = state.get("action", "execute")
    
    if action == "answer":
        logger.info("➡️ Routing to answer")
        return "answer"
    
    logger.info("➡️ Routing to execute")
    return "execute"