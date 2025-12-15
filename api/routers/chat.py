"""
Chat Router - Streaming chat with RAG context.

Provides:
- POST /chat/stream - Stream chat with RAG retrieval
- POST /chat/query-rag - Query RAG directly
"""

import asyncio
import numpy as np
import json
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Annotated
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

from schemas import (
    ChatRequest,
    ChatChunk,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGChunk,
    ErrorResponse,
)
from dependencies import RAGManager, settings
from auth import (
    get_current_user_optional,
    get_current_active_user,
    User,
)


router = APIRouter(prefix="/chat", tags=["Chat"])


# =============================================================================
# STREAMING CHAT
# =============================================================================


@router.post(
    "/stream",
    summary="Stream chat with RAG",
    description="Stream a chat response with automatic RAG context retrieval.",
)
async def stream_chat(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """
    Stream chat response with RAG context.

    Uses Server-Sent Events (SSE) for real-time token streaming.
    The agent automatically retrieves relevant context from RAG.

    Requires authentication.
    """
    rag = await RAGManager.get_rag(request.workflow_id)

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Initialize LLM with streaming
            llm = ChatOllama(
                model=request.model or settings.DEFAULT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                streaming=True,
            )

            # Build message history
            messages = []
            if request.history:
                for msg in request.history:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))

            # Retrieve RAG context
            rag_context = ""
            if rag and rag.enabled:
                try:
                    # Emit context retrieval event
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving context...'})}\n\n"

                    rag_context = rag.get_context_summary(
                        query=request.message,
                        workflow_id=request.workflow_id,
                        max_tokens=2000,
                    )

                    if rag_context:
                        yield f"data: {json.dumps({'type': 'context', 'content': f'Found {len(rag_context)} chars of context'})}\n\n"

                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'content': 'Context retrieval failed'})}\n\n"

            # Build system message with context
            system_content = (
                "You are a helpful data science assistant. "
                "Answer questions based on the provided context and your knowledge. "
                "Be accurate, cite sources when available, and provide detailed explanations."
            )

            if rag_context:
                system_content += f"\n\nRelevant Context:\n{rag_context}"

            messages.insert(0, SystemMessage(content=system_content))
            messages.append(HumanMessage(content=request.message))

            # Stream response
            yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

            full_response = ""
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Store the interaction in RAG for future reference
            if rag and rag.enabled and full_response:
                try:
                    rag.add_summary(
                        summary=f"Q: {request.message}\nA: {full_response[:500]}...",
                        stage_name="chat",
                        workflow_id=request.workflow_id,
                        metadata={"timestamp": datetime.now().isoformat()},
                    )
                except Exception as e:
                    logger.warning(f"Failed to store chat in RAG: {e}")

            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# AGENT CHAT (with tool use)
# =============================================================================


@router.post(
    "/agent/stream",
    summary="Stream agent chat with tools",
    description="Stream chat with an agent that can use RAG retrieval and storage tools.",
)
async def stream_agent_chat(request: ChatRequest):
    """
    Stream agent chat with tool capabilities.

    The agent can:
    - retrieve_context: Query RAG for relevant information
    - store_knowledge: Save useful information to RAG
    """
    rag = await RAGManager.get_rag(request.workflow_id)

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            llm = ChatOllama(
                model=request.model or settings.DEFAULT_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                streaming=True,
                temperature=0.0,
                # format='json',
                
            )

            # Define tools
            @tool
            def retrieve_context(query: str) -> list[str]:
                """Search the knowledge base and retrieve relevant documents.
                
                Use this tool to find information before answering user questions. The knowledge base
                contains prior conversations, uploaded documents, and analysis results.
                
                Args:
                    query: A focused search query. Use specific terms, entities, or concepts.
                        Examples: "Q3 revenue trends", "customer churn analysis", "pandas merge syntax"
                
                Returns:
                    List of relevant documents, each formatted as:
                    "# Source: [document_title] - Relevance: [0.0-1.0]
                    
                    [document content]"
                    
                    Higher relevance scores (closer to 1.0) indicate better matches.
                    Returns up to 20 results, ordered by relevance.
                
                Usage Tips:
                    - Call multiple times with different queries to gather comprehensive context
                    - Try variations: exact terms, synonyms, broader/narrower concepts
                    - Check relevance scores - scores below 0.3 may indicate off-topic results
                    - If results seem incomplete, reformulate your query and search again"""
                
                if not rag or not rag.enabled:
                    logger.warning("RAG not available for retrieve_context tool")
                    return ["RAG not available"]
                
                logger.info(f"Tool retrieve_context called with query: {query}")
                results = rag.query_relevant_context(
                    query=query,
                    workflow_id=request.workflow_id,
                    n_results=20,
                )
                processed_results = np.unique([
                    f"# Source: {r.get('metadata',{}).get('title', 'N/A')} - Relevance: {1-r.get('distance', 1): .2f}\n\n{r['document']}" 
                    for r in results
                    if (1 - r.get("distance", 1)) >= 0.3
                ]).tolist()
                return processed_results
            
            @tool
            def store_knowledge(document: str) -> str:
                """Store important information in the knowledge base for future retrieval.
    
                Use this tool to persist valuable information that may be needed in future conversations.
                Stored documents become searchable via retrieve_context tool.
                
                Args:
                    document: Well-structured text to store. Should include:
                            - Clear title/header describing the content
                            - Key facts, findings, or insights
                            - Context about when/why this information matters
                            - Relevant entities, metrics, or terminology for searchability
                
                What to Store:
                    ✓ User-provided facts: "Company has 500 employees in 3 offices"
                    ✓ Analysis results: "Churn rate increased 15% in Q3 due to pricing changes"
                    ✓ Decisions made: "User prefers matplotlib over seaborn for visualizations"
                    ✓ Important context: "Dataset contains 2019-2024 sales data, missing Q2 2021"
                    ✓ Definitions: "NPS = Net Promoter Score, calculated as % promoters - % detractors"
                    
                What NOT to Store:
                    ✗ Transient chat messages or greetings
                    ✗ Information already in uploaded documents
                    ✗ Redundant content that's already stored
                
                Format Example:
                    "# User Preference: Visualization Style
                    
                    User prefers clean, minimal visualizations with:
                    - Matplotlib as primary library
                    - Seaborn only for statistical plots
                    - Color palette: blues and grays
                    - Always include axis labels and titles"
                
                Returns:
                    Confirmation message if successful, error message if RAG unavailable.
                """

                if not rag or not rag.enabled:
                    logger.warning("RAG not available for store_knowledge tool")
                    return "RAG not available"
                logger.info(f"Tool store_knowledge called")
                rag.add_summary(
                    summary=document,
                    stage_name="chat",
                    workflow_id=request.workflow_id,
                    metadata={"timestamp": datetime.now().isoformat()},
                )
                return "Knowledge stored successfully"

            # Create agent
            try:
                from langchain.agents import create_agent
                from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
                
                agent = create_agent(
                    model=llm,
                    tools=[retrieve_context, store_knowledge],
                )
                logger.info(f"Using LangGraph agent for chat streaming have {len(request.history or [])} history messages")
                # Build messages
                messages = [
                    SystemMessage(
                        content=(
                        "You are an expert data science communicator with access to a vast knowledge base. "
                        
                        "TOOL USAGE RULES:\n"
                        "- Use the structured tool calling format provided by the system\n"
                        "- DO NOT output <function-call> XML tags\n"
                        "- DO NOT output raw JSON like {\"name\": \"tool_name\"}\n"
                        "- Let the system handle tool invocation\n\n"
                        "CRITICAL: Your workflow for EVERY user query:\n\n"
                        "CRITICAL: You MUST use retrieve_context tool before answering ANY question.\n"
                        "1. PLAN YOUR QUERIES\n"
                        "   - Break down the user's question into key concepts\n"
                        "   - Identify multiple search angles: direct terms, synonyms, related concepts\n"
                        "   - Consider temporal aspects (recent vs historical)\n\n"
                        
                        "2. EXECUTE SEARCHES (call retrieve_context multiple times)\n"
                        "   - Start broad: query main topic\n"
                        "   - Get specific: query each key entity, metric, or concept\n"
                        "   - Explore related: query connected topics that context reveals\n"
                        "   - Continue until you get redundant results or have comprehensive coverage\n\n"
                        
                        "3. SYNTHESIZE & ANSWER\n"
                        "   - Combine all retrieved context\n"
                        "   - Cite specific sources when making claims\n"
                        "   - Note if information seems incomplete\n\n"
                        
                        "4. STORE NEW KNOWLEDGE\n"
                        "   - If user provides new information, use store_knowledge\n\n"
                        
                        "Example: For 'What were Q3 sales?', query:\n"
                        "- 'Q3 sales'\n"
                        "- 'third quarter revenue'\n"
                        "- 'Q3 2024 financial results'\n"
                        "- 'quarterly sales comparison'\n\n"

                        "Always retrieve context BEFORE answering. Multiple searches are expected.\n"
                        "Do NOT answer from your own knowledge - always retrieve context first."
                        )
                    )
                ]
                if request.history:
                    for msg in request.history:
                        if msg.role == "user":
                            messages.append(HumanMessage(content=msg.content))
                        elif msg.role == "assistant":
                            messages.append(AIMessage(content=msg.content))

                messages.append(HumanMessage(content=request.message))

                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                # Stream agent response
                async for event in agent.astream(
                    {"messages": messages},
                    stream_mode="messages",
                ):
                    message, metadata = event
                    # logger.info(f"Event node: {metadata.get('langgraph_node')}, message type: {type(message).__name__}")
                    # Handle tool calls
                    if metadata.get("langgraph_node") == "tools":
                        tool_name = getattr(message, "name", "tool")
                        yield f"data: {json.dumps({'type': 'tool', 'content': f'Using {tool_name}...'})}\n\n"

                    # Handle model output
                    elif metadata.get("langgraph_node") == "model":
                        if hasattr(message, "content_blocks") and message.content_blocks and message.content_blocks[-1]['type'] == "text":
                            content = message.content_blocks[-1]['text']
                            if "<function-call>" in content or "</function-call>" in content:
                                logger.warning(f"Filtered XML function call from output: {content[:100]}")
                                continue  # Skip this - don't stream it
                            
                            # Also filter JSON-only responses that look like tool calls
                            if content.strip().startswith('{"name":') and '"arguments"' in content:
                                logger.warning(f"Filtered JSON tool call from output: {content[:100]}")
                                continue  # 
                            yield f"data: {json.dumps({'type': 'token', 'content': message.content_blocks[-1]['text']})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

            except ImportError:
                # Fallback to simple streaming if langgraph agent not available
                logger.warning("LangGraph agent not available, using simple streaming")

                # Get context first
                context = ""
                if rag and rag.enabled:
                    context = rag.get_context_summary(
                        query=request.message,
                        workflow_id=request.workflow_id,
                        max_tokens=2000,
                    )

                system = SystemMessage(
                    content=f"""You are a helpful assistant.
                
Context from knowledge base:
{context if context else 'No relevant context found.'}

Answer the user's question based on the context and your knowledge."""
                )

                messages = [system, HumanMessage(content=request.message)]

                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                async for chunk in llm.astream(messages):
                    if chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            import traceback

            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# RAG QUERY
# =============================================================================


@router.post(
    "/query-rag",
    response_model=RAGQueryResponse,
    summary="Query RAG directly",
    description="Search the RAG knowledge base directly without chat.",
)
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    """Direct RAG query without chat context."""
    workflow_id = request.workflow_id or "default"
    rag = await RAGManager.get_rag(workflow_id)

    if not rag or not rag.enabled:
        raise HTTPException(status_code=503, detail="RAG not available")

    try:
        results = rag.query_relevant_context(
            query=request.query,
            workflow_id=workflow_id,
            doc_types=getattr(request, 'doc_types', None),
            n_results=request.n_results,
        )

        chunks = []
        for r in results:
            relevance = (
                1.0 - r.get("distance", 0.5) if r.get("distance") is not None else 0.5
            )
            chunks.append(
                RAGChunk(
                    content=r["document"],
                    metadata=r["metadata"],
                    relevance_score=relevance,
                )
            )

        return RAGQueryResponse(
            query=request.query,
            results=chunks,
            total_found=len(chunks),
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/rag-stats/{workflow_id}",
    summary="Get RAG statistics",
)
async def get_rag_stats(workflow_id: str) -> dict:
    """Get statistics about the RAG knowledge base."""
    rag = await RAGManager.get_rag(workflow_id)

    if not rag or not rag.enabled:
        return {"enabled": False, "error": "RAG not available"}

    try:
        return rag.get_stats()
    except Exception as e:
        return {"enabled": True, "error": str(e)}
