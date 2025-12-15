"""
Node functions for RAG retrieval workflow.

Nodes:
- plan_queries: Generate initial specific + broad queries
- rewrite_query: Transform query to match RAG semantics
- execute_query: Run query against RAG
- grade_context: Grade relevance and extract subtopics
- synthesize: Combine all relevant context
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from langgraph.types import RuntimeValue

from .state import (
    RetrievalState, 
    RetrievalConfig, 
    QueryTask, 
    RetrievedChunk,
    Deps,
)
from .models import QueryPlan, RewrittenQuery, ContextGrade, SynthesisResult


# =============================================================================
# HELPERS
# =============================================================================

def get_deps(runtime: Runtime[Deps]) -> Deps:
    """Extract deps from runtime context."""
    return runtime.context if hasattr(runtime, 'context') else {}


def get_llm(deps: Deps) -> ChatOllama:
    """Get configured LLM from dependencies."""
    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    return ChatOllama(
        model=config.llm_model,
        temperature=0,
        base_url=config.base_url,
    )


def get_rag(deps: Deps):
    """Get RAG instance from dependencies."""
    return deps.get("rag")


def emit_progress(deps: Deps, message: str):
    """Emit progress if callback available."""
    callback = deps.get("on_progress")
    if callback:
        callback(message)
    logger.debug(f"RAG Retrieval: {message}")


# =============================================================================
# NODE: PLAN QUERIES
# =============================================================================

async def plan_queries(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Generate initial queries - both specific and broad.
    
    For "how can I build a linear regression":
    - Specific: "linear regression python", "sklearn LinearRegression"
    - Broad: "regression models", "supervised learning", "statistical modeling"
    """
    deps = get_deps(runtime)
    emit_progress(deps, "📋 Planning search queries...")
    
    llm = get_llm(deps).with_structured_output(QueryPlan)
    original_query = state["original_query"]
    
    prompt = f"""Generate search queries for a RAG knowledge base.

USER QUERY: {original_query}

Generate two types of queries:

1. SPECIFIC QUERIES (2-3):
   - Target the exact user request
   - Use technical terms, library names, method names
   - Example: "linear regression" → "sklearn LinearRegression", "OLS statsmodels"

2. BROAD QUERIES (2-3):
   - Capture related concepts and parent topics
   - Help find context that might be indirectly relevant
   - Example: "linear regression" → "regression models", "supervised learning"

The knowledge base contains:
- Code examples and implementations
- Documentation and tutorials  
- Analysis summaries and results
- PDF documents and web articles

Generate queries optimized for semantic search."""

    try:
        result = llm.invoke([
            SystemMessage(content="Generate targeted RAG search queries."),
            HumanMessage(content=prompt),
        ])
        
        # Build todo list with priorities
        todo: List[QueryTask] = []
        
        # Specific queries first (higher priority)
        for q in result.specific_queries:
            todo.append(QueryTask(
                query=q,
                query_type="specific",
                priority=2,
            ))
        
        # Then broad queries
        for q in result.broad_queries:
            todo.append(QueryTask(
                query=q,
                query_type="broad", 
                priority=1,
            ))
        
        emit_progress(deps, f"📋 Planned {len(todo)} queries: {[t.query for t in todo]}")
        
        return {
            "query_todo": todo,
            "iteration": state.get("iteration", 0) + 1,
        }
        
    except Exception as e:
        logger.error(f"Query planning failed: {e}")
        # Fallback: use original query
        return {
            "query_todo": [QueryTask(
                query=original_query,
                query_type="specific",
                priority=2,
            )],
            "iteration": state.get("iteration", 0) + 1,
        }


# =============================================================================
# NODE: REWRITE QUERY
# =============================================================================

async def rewrite_query(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Transform current query to match RAG document semantics.
    
    Takes into account:
    - Document types in the RAG (code, docs, summaries)
    - Technical terminology variations
    - Common phrasings in stored documents
    """
    deps = get_deps(runtime)
    todo = state.get("query_todo", [])
    if not todo:
        return {"should_stop": True, "error": "No queries to process"}
    
    # Pop highest priority query
    todo = sorted(todo, key=lambda x: -x.priority)
    current = todo[0]
    remaining = todo[1:]
    
    emit_progress(deps, f"✏️ Rewriting query: {current.query}")
    
    llm = get_llm(deps).with_structured_output(RewrittenQuery)
    
    # Get document type hints from RAG if available
    rag = get_rag(deps)
    doc_types_hint = ""
    if rag and hasattr(rag, 'get_stats'):
        try:
            stats = rag.get_stats()
            types = stats.get("type_breakdown", {})
            if types:
                doc_types_hint = f"\nDocument types in RAG: {', '.join(types.keys())}"
        except:
            pass
    
    prompt = f"""Rewrite this search query to match RAG document semantics.

ORIGINAL QUERY: {current.query}
QUERY TYPE: {current.query_type}
{doc_types_hint}

The RAG contains various document types:
- Code: Function definitions, implementations, examples
- Documentation: Tutorials, guides, API references
- Summaries: Analysis results, findings
- PDFs/URLs: Research papers, articles

REWRITING RULES:
1. Transform natural language to technical terms
   "how do I build" → "implementation", "example", "tutorial"
2. Add library/framework names when relevant
   "linear regression" → "sklearn LinearRegression", "statsmodels OLS"
3. Include common variations
   "ML model" → "machine learning", "classifier", "predictor"
4. Match document structure patterns
   Questions → keywords: "how to X" → "X tutorial", "X example"

Also provide 2-3 alternative phrasings and key terms to search."""

    try:
        result = llm.invoke([
            SystemMessage(content="Rewrite queries to match document semantics."),
            HumanMessage(content=prompt),
        ])
        
        emit_progress(deps, f"✏️ Rewritten: '{current.query}' → '{result.rewritten}'")
        
        return {
            "current_query": current,
            "rewritten_query": result.rewritten,
            "query_todo": remaining,
        }
        
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        # Fallback: use original
        return {
            "current_query": current,
            "rewritten_query": current.query,
            "query_todo": remaining,
        }


# =============================================================================
# NODE: EXECUTE QUERY
# =============================================================================

async def execute_query(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Execute the rewritten query against RAG.
    
    Returns chunks with metadata for grading.
    """
    deps = get_deps(runtime)
    rewritten = state.get("rewritten_query")
    current = state.get("current_query")
    todo = state.get("query_todo", [])
    queries: List[str] = [current.query]
    if todo:
        todo = sorted(todo, key=lambda x: -x.priority)
        queries = [q.query for q in todo[-3:]]
        state["query_todo"] = todo[:-3]
    

    if not rewritten or not current:
        return {"error": "No query to execute"}
    
    emit_progress(deps, f"🔍 Searching RAG: {rewritten}")
    
    rag = get_rag(deps)
    if not rag:
        return {"error": "RAG not available"}
    
    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    workflow_id = state.get("workflow_id")
    
    try:
        # Query RAG
        results = []
        for q in queries:
            results.extend(rag.query_relevant_context(
            query=q,
            n_results=config.max_chunks_per_query,
            workflow_id=workflow_id,
        ))
        
        # Convert to RetrievedChunk objects
        chunks: List[RetrievedChunk] = []

        
        if results:
            docs = [res.get('document') for res in results if res.get('document')]
            metas = [res.get('metadata', {}) for res in results if res.get('document')]
            distances = [res.get('distance', 1.0) for res in results if res.get('document')]
            
            for doc, meta, dist in zip(docs, metas, distances):
                # Convert distance to relevance score (lower distance = higher relevance)
                relevance = max(0, 1 - dist)
                
                chunk = RetrievedChunk(
                    content=doc,
                    source_type=meta.get("type", "unknown"),
                    metadata=meta,
                    relevance_score=relevance,
                    query_used=rewritten,
                )
                chunks.append(chunk)
        
        # Add to retrieved chunks (dedupe by content)
        existing = state.get("retrieved_chunks", [])
        existing_content = {c.content for c in existing}
        
        new_chunks = [c for c in chunks if c.content not in existing_content]
        all_chunks = existing + new_chunks
        
        # Track completed queries
        completed = state.get("completed_queries", [])
        for q in queries:
            if q not in completed:
                completed = completed + [q]
        
        
        emit_progress(deps, f"🔍 Found {len(new_chunks)} new chunks ({len(chunks)} total)")
        
        return {
            "retrieved_chunks": all_chunks,
            "completed_queries": completed,
        }
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return {"error": f"Query failed: {e}"}


# =============================================================================
# NODE: GRADE CONTEXT
# =============================================================================

async def grade_context(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Grade retrieved context for relevance.
    Extract subtopics for further exploration.
    """
    deps = get_deps(runtime)
    retrieved = state.get("retrieved_chunks", [])
    current = state.get("current_query")
    original_query = state["original_query"]
    
    if not retrieved:
        emit_progress(deps, "⚠️ No chunks to grade")
        return {
            "last_grade_score": 0.0,
            "consecutive_low_relevance": state.get("consecutive_low_relevance", 0) + 1,
        }
    
    # Get chunks from current query that haven't been graded yet
    relevant_existing = state.get("relevant_chunks", [])
    relevant_content = {c.content for c in relevant_existing}
    
    # Only grade chunks from current query
    current_query = state.get("rewritten_query", "")
    chunks_to_grade = [
        c for c in retrieved 
        if c.query_used == current_query and c.content not in relevant_content
    ]
    
    if not chunks_to_grade:
        return {
            "last_grade_score": 0.0,
            "consecutive_low_relevance": state.get("consecutive_low_relevance", 0) + 1,
        }
    
    emit_progress(deps, f"📊 Grading {len(chunks_to_grade)} chunks...")
    
    llm = get_llm(deps).with_structured_output(ContextGrade)
    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    
    # Format chunks for grading
    chunks_text = "\n\n---\n\n".join([
        f"[{c.source_type}] (score: {c.relevance_score:.2f})\n{c.content}"
        for c in chunks_to_grade  # Grade top 5 at a time
    ])
    
    prompt = f"""Grade the relevance of this retrieved context.

ORIGINAL USER QUERY: {original_query}
CURRENT SEARCH QUERY: {current_query}

CURRENT RELEVANT CONTEXT: 
{chr(10).join([f'- {c.content[:100]}...' for c in relevant_existing[-5:]])}

RETRIEVED CONTEXT:
{chunks_text}

GRADING CRITERIA:
1. Is this context relevant to answering the user's query?
2. What is the overall relevance score (0-1)?
3. What new subtopics are mentioned that we should explore?
   - Look for related concepts, methods, libraries
   - Example: if searching "linear regression" and context mentions "regularization", "ridge regression", "feature selection" - those are subtopics
4. What aspects of the query are NOT yet covered?
5. Should we continue searching?

If enough relevant context is found, we may stop further searching."""

    try:
        result = llm.invoke([
            SystemMessage(
                content=(
                    "You are an expert researcher." 
                    " You are excellent at understanding the semantic meaning of a users query and identifying"
                    " wether the associated context is relevant."
                    " You are also skilled at extracting subtopics from technical documents. That will help guide further searches. "
                    " Grade RETRIEVED CONTEXT relevance and extract subtopics.")
                    ),
            HumanMessage(content=prompt),
        ])
        
        # Filter chunks by relevance
        
        
        new_relevant = [c for c in chunks_to_grade if c.relevance_score >= 0.3]
        
        all_relevant = relevant_existing + new_relevant
        
        # Track subtopics
        discovered = state.get("discovered_subtopics", [])
        explored = state.get("explored_subtopics", [])
        
        new_subtopics = [
            s for s in result.extracted_subtopics 
            if s not in discovered and s not in explored
        ]
        all_discovered = discovered + new_subtopics
        
        # Add subtopic queries to todo if we should continue
        todo = state.get("query_todo", [])
        if result.should_continue and new_subtopics:
            for subtopic in new_subtopics[:config.max_subtopics_to_explore]:
                todo.append(QueryTask(
                    query=subtopic,
                    query_type="subtopic",
                    priority=1,
                    parent_query=current_query,
                ))
        
        # Track consecutive low relevance
        consecutive_low = state.get("consecutive_low_relevance", 0)
        if result.relevance_score < config.relevance_threshold:
            consecutive_low += 1
        else:
            consecutive_low = 0
        
        
        emit_progress(
            deps, 
            f"📊 Grade: {result.relevance_score:.2f}, "
            f"subtopics: {new_subtopics}, "
            f"relevant chunks: {len(new_relevant)}"
        )
        
        return {
            "relevant_chunks": all_relevant,
            "last_grade_score": result.relevance_score,
            "discovered_subtopics": all_discovered,
            "query_todo": todo,
            "consecutive_low_relevance": consecutive_low,
            "total_relevant_found": len(all_relevant),
            "iteration": state.get("iteration", 0) + 1,
        }
        
    except Exception as e:
        logger.error(f"Context grading failed: {e}")
        return {
            "last_grade_score": 0.5,  # Assume moderate relevance
            "consecutive_low_relevance": state.get("consecutive_low_relevance", 0),
            "iteration": state.get("iteration", 0) + 1,
        }


# =============================================================================
# NODE: SYNTHESIZE
# =============================================================================

async def synthesize(state: RetrievalState, runtime: Runtime[Deps]) -> Dict[str, Any]:
    """
    Combine all relevant chunks into synthesized context.
    
    Groups by document type and creates coherent summary.
    """
    deps = get_deps(runtime)
    relevant = state.get("relevant_chunks", [])
    original_query = state["original_query"]
    
    if not relevant:
        emit_progress(deps, "⚠️ No relevant context to synthesize")
        return {
            "synthesized_context": "",
            "key_findings": [],
            "should_stop": True,
        }
    
    emit_progress(deps, f"🔄 Synthesizing {len(relevant)} relevant chunks...")
    
    llm = get_llm(deps).with_structured_output(SynthesisResult)
    config: RetrievalConfig = deps.get("config", RetrievalConfig())
    
    # Group by document type
    by_type: Dict[str, List[RetrievedChunk]] = {}
    for chunk in relevant:
        doc_type = chunk.source_type
        if doc_type not in by_type:
            by_type[doc_type] = []
        by_type[doc_type].append(chunk)
    
    # Format grouped context
    grouped_text = ""
    for doc_type, chunks in by_type.items():
        grouped_text += f"\n\n## {doc_type.upper()} SOURCES:\n"
        for c in chunks[:5]:  # Limit per type
            grouped_text += f"\n{c.content[:800]}\n---"
    
    # Truncate if too long
    if len(grouped_text) > config.max_context_chars:
        grouped_text = grouped_text[:config.max_context_chars] + "\n[truncated]"
    
    prompt = f"""Synthesize this retrieved context into a coherent response.

ORIGINAL QUERY: {original_query}

RETRIEVED CONTEXT (grouped by source type):
{grouped_text}

SYNTHESIS GOALS:
1. Combine information from different sources coherently
2. Highlight the most relevant information for the query
3. Note which document types contributed (code examples, documentation, etc.)
4. Identify key findings that directly address the query
5. Assess how well the combined context covers the query

Create a synthesized context that would help answer the user's query."""

    try:
        result = llm.invoke([
            SystemMessage(content="Synthesize RAG context into coherent response."),
            HumanMessage(content=prompt),
        ])
        
        emit_progress(
            deps,
            f"✅ Synthesized context ({len(result.combined_context)} chars), "
            f"confidence: {result.confidence:.2f}"
        )
        
        return {
            "synthesized_context": result.combined_context,
            "key_findings": result.key_findings,
            "document_types_found": result.document_types_found,
            "should_stop": True,
        }
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        # Fallback: concatenate relevant chunks
        fallback_context = "\n\n".join([c.content for c in relevant[:10]])
        return {
            "synthesized_context": fallback_context[:config.max_context_chars],
            "key_findings": [],
            "document_types_found": list(by_type.keys()),
            "should_stop": True,
        }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_continue(state: RetrievalState) -> str:
    """
    Determine next action after grading.
    
    Returns:
        "rewrite_query" - More queries to process
        "synthesize" - Done searching, combine results
    """
    config_max = state.get("max_iterations", 10)
    iteration = state.get("iteration", 0)
    todo = state.get("query_todo", [])
    consecutive_low = state.get("consecutive_low_relevance", 0)
    total_relevant = state.get("total_relevant_found", 0)
    
    # Stop conditions
    if state.get("should_stop"):
        return "synthesize"
    
    if iteration >= config_max:
        logger.info(f"Max iterations ({config_max}) reached")
        return "synthesize"
    
    if consecutive_low >= 3:
        logger.info("Too many low-relevance results, stopping")
        return "synthesize"
    
    if not todo and total_relevant > 0:
        logger.info("No more queries and have relevant results")
        return "synthesize"
    
    if todo:
        return "rewrite_query"
    
    # No todo and no relevant results - try broader search
    if total_relevant == 0 and iteration < 3:
        return "rewrite_query"  # Will handle empty todo in rewrite
    
    return "synthesize"