"""
Web Search - Agent-based research using DuckDuckGo.

Features:
- LLM agent crafts optimized search queries
- Multiple search iterations for comprehensive results
- Query refinement based on initial results
- Progress callbacks for real-time updates
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict
from loguru import logger

from pydantic import BaseModel, Field

# DuckDuckGo search
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed. Install with: pip install duckduckgo-search")

# Crawl4ai for deep page crawling
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("crawl4ai not installed")

# LangChain for agent
try:
    from langchain_ollama import ChatOllama
    from langchain.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain not available for search agent")


@dataclass
class SearchResult:
    url: str
    title: str
    content: str
    score: float = 0.0
    source: str = "duckduckgo"
    query_used: str = ""  # Track which query found this


@dataclass
class WebSearchContext:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# PYDANTIC MODELS FOR AGENT
# =============================================================================

class SearchQueries(BaseModel):
    """Generated search queries from the agent."""
    queries: List[str] = Field(
        description="List of 2-4 optimized search queries",
        min_length=1,
        max_length=5
    )
    reasoning: str = Field(
        description="Brief explanation of query strategy"
    )


class RefinedQueries(BaseModel):
    """Refined queries based on initial results."""
    queries: List[str] = Field(
        description="1-2 refined follow-up queries to fill gaps",
        min_length=0,
        max_length=3
    )
    sufficient: bool = Field(
        description="Whether current results are sufficient"
    )
    reasoning: str = Field(
        description="What information is missing or why results are sufficient"
    )


class SearchSynthesis(BaseModel):
    """Synthesized search results."""
    summary: str = Field(
        description="Concise summary of key findings"
    )
    key_insights: List[str] = Field(
        description="3-5 key insights from the search"
    )
    relevance_score: float = Field(
        description="How relevant the results are to the original query (0-1)",
        ge=0.0,
        le=1.0
    )


# =============================================================================
# SEARCH AGENT
# =============================================================================

class SearchAgent:
    """
    Agent that intelligently searches the web.
    
    Uses an LLM to:
    1. Generate optimized search queries
    2. Evaluate results and refine queries
    3. Synthesize findings into actionable insights
    """
    
    def __init__(
        self,
        llm_model: str = "qwen3:30b",
        base_url: str = "http://100.91.155.118:11434",
        max_iterations: int = 2,
        results_per_query: int = 3
    ):
        self.llm_model = llm_model
        self.base_url = base_url
        self.max_iterations = max_iterations
        self.results_per_query = results_per_query
        
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOllama(
                model=llm_model,
                temperature=0,
                base_url=base_url
            )
        else:
            self.llm = None
            logger.warning("LangChain not available - using basic search")
    
    def generate_queries(
        self,
        user_query: str,
        context: str = "",
        on_progress: Optional[Callable[[str], None]] = None
    ) -> SearchQueries:
        """Generate optimized search queries for the user's question."""
        if not self.llm:
            # Fallback: return the original query
            return SearchQueries(
                queries=[f"{user_query} python data science"],
                reasoning="LLM not available, using basic query"
            )
        
        if on_progress:
            on_progress("🧠 Analyzing query to generate search terms...")
        
        structured_llm = self.llm.with_structured_output(SearchQueries)
        
        prompt = f"""Generate optimized web search queries for a data science research task.

User's Question: {user_query}

Additional Context:
{context[:500] if context else "None provided"}

Guidelines:
- Create 2-4 specific, targeted search queries
- Include technical terms and library names when relevant
- One query should be broad, others more specific
- Add "python" or specific library names for code-related queries
- Focus on methodology, best practices, and implementation

Examples of good queries:
- "pandas groupby aggregation best practices"
- "correlation analysis python statsmodels"
- "feature importance random forest sklearn tutorial"
"""
        
        try:
            result = structured_llm.invoke([
                SystemMessage(content="You are a research assistant. Generate effective web search queries."),
                HumanMessage(content=prompt)
            ])
            
            if on_progress:
                on_progress(f"📝 Generated {len(result.queries)} search queries")
            
            logger.info(f"🔍 Generated queries: {result.queries}")
            return result
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return SearchQueries(
                queries=[user_query],
                reasoning=f"Fallback due to error: {e}"
            )
    
    def refine_queries(
        self,
        original_query: str,
        current_results: List[SearchResult],
        queries_used: List[str],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> RefinedQueries:
        """Evaluate results and generate refined queries if needed."""
        if not self.llm:
            return RefinedQueries(
                queries=[],
                sufficient=True,
                reasoning="LLM not available"
            )
        
        if on_progress:
            on_progress("🔄 Evaluating search results...")
        
        # Summarize current results
        results_summary = "\n".join([
            f"- {r.title}: {r.content[:150]}..."
            for r in current_results[:5]
        ])
        
        structured_llm = self.llm.with_structured_output(RefinedQueries)
        
        prompt = f"""Evaluate search results and determine if more searches are needed.

Original Question: {original_query}

Queries Already Used:
{chr(10).join(f'- {q}' for q in queries_used)}

Current Results:
{results_summary}

Determine:
1. Are the results sufficient to answer the question?
2. What information is missing?
3. If insufficient, what 1-2 refined queries would help?

Consider:
- Code examples / tutorials
- Best practices / methodology
- Specific library documentation
- Common pitfalls / troubleshooting
"""
        
        try:
            result = structured_llm.invoke([
                SystemMessage(content="Evaluate search results and suggest refinements."),
                HumanMessage(content=prompt)
            ])
            
            if on_progress:
                if result.sufficient:
                    on_progress("✅ Results are sufficient")
                else:
                    on_progress(f"🔄 Need {len(result.queries)} more queries: {result.reasoning[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return RefinedQueries(
                queries=[],
                sufficient=True,
                reasoning=f"Error during refinement: {e}"
            )
    
    def synthesize_results(
        self,
        original_query: str,
        results: List[SearchResult],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> SearchSynthesis:
        """Synthesize search results into actionable insights."""
        if not self.llm or not results:
            return SearchSynthesis(
                summary="No results to synthesize",
                key_insights=[],
                relevance_score=0.0
            )
        
        if on_progress:
            on_progress("🧩 Synthesizing search results...")
        
        # Format results for LLM
        results_text = "\n\n".join([
            f"[{i+1}] {r.title}\nURL: {r.url}\n{r.content[:300]}"
            for i, r in enumerate(results[:8])
        ])
        
        structured_llm = self.llm.with_structured_output(SearchSynthesis)
        
        prompt = f"""Synthesize these search results into actionable insights.

Original Question: {original_query}

Search Results:
{results_text}

Provide:
1. A concise summary of key findings
2. 3-5 specific, actionable insights
3. A relevance score (0-1) for how well results answer the question
"""
        
        try:
            result = structured_llm.invoke([
                SystemMessage(content="Synthesize web search results into insights."),
                HumanMessage(content=prompt)
            ])
            
            if on_progress:
                on_progress(f"✅ Synthesis complete (relevance: {result.relevance_score:.0%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return SearchSynthesis(
                summary=f"Synthesis error: {e}",
                key_insights=[],
                relevance_score=0.5
            )
    
    async def search(
        self,
        query: str,
        context: str = "",
        on_progress: Optional[Callable[[str], None]] = None,
        enrich_with_crawl: bool = True,
        max_crawl_urls: int = 3
    ) -> WebSearchContext:
        """
        Run an agent-guided web search.
        
        1. Generate optimized queries
        2. Execute searches
        3. Evaluate and refine if needed
        4. Optionally enrich top results with full page content via crawl4ai
        5. Return comprehensive results
        """
        ctx = WebSearchContext(query=query)
        
        if not DDGS_AVAILABLE:
            ctx.error = "DuckDuckGo not available"
            return ctx
        
        try:
            # Step 1: Generate initial queries
            query_result = self.generate_queries(query, context, on_progress)
            
            # Step 2: Execute initial searches
            for search_query in query_result.queries:
                if on_progress:
                    on_progress(f"🔍 Searching: {search_query[:50]}...")
                
                results = search_duckduckgo(
                    search_query,
                    max_results=self.results_per_query
                )
                
                # Tag results with the query that found them
                for r in results:
                    r.query_used = search_query
                
                ctx.results.extend(results)
                ctx.queries_used.append(search_query)
                
                if on_progress:
                    on_progress(f"   Found {len(results)} results")
            
            # Step 3: Evaluate and refine (if LLM available)
            for iteration in range(self.max_iterations - 1):
                if not self.llm:
                    break
                
                refinement = self.refine_queries(
                    query,
                    ctx.results,
                    ctx.queries_used,
                    on_progress
                )
                
                if refinement.sufficient or not refinement.queries:
                    break
                
                # Execute refined queries
                for search_query in refinement.queries:
                    if on_progress:
                        on_progress(f"🔍 Refined search: {search_query[:50]}...")
                    
                    results = search_duckduckgo(
                        search_query,
                        max_results=self.results_per_query
                    )
                    
                    for r in results:
                        r.query_used = search_query
                    
                    ctx.results.extend(results)
                    ctx.queries_used.append(search_query)
            
            # Deduplicate results by URL
            seen_urls = set()
            unique_results = []
            for r in ctx.results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)
            ctx.results = unique_results
            
            # Step 4: Enrich top results with crawl4ai
            if enrich_with_crawl and CRAWL4AI_AVAILABLE and ctx.results:
                ctx.results = await self._enrich_results_with_crawl(
                    ctx.results,
                    query,
                    max_urls=max_crawl_urls,
                    on_progress=on_progress
                )
            
            if on_progress:
                on_progress(f"✅ Search complete: {len(ctx.results)} unique results from {len(ctx.queries_used)} queries")
            
            logger.info(f"🌐 Agent search complete: {len(ctx.results)} results")
            
        except Exception as e:
            ctx.error = str(e)
            logger.error(f"Agent search failed: {e}")
        
        return ctx
    
    async def _enrich_results_with_crawl(
        self,
        results: List[SearchResult],
        query: str,
        max_urls: int = 3,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[SearchResult]:
        """
        Enrich search results by crawling top URLs for full content.
        
        Uses crawl4ai to extract detailed content from the most promising results.
        """
        if not CRAWL4AI_AVAILABLE:
            return results
        
        if on_progress:
            on_progress(f"🕷️ Selecting best URLs to crawl for detailed content...")
        
        # Use LLM to select best URLs to crawl
        urls_to_crawl = await self._select_urls_to_crawl(results, query, max_urls, on_progress)
        
        if not urls_to_crawl:
            # Fallback to score-based selection
            urls_to_crawl = self._fallback_url_selection(results, max_urls)
        
        if not urls_to_crawl:
            return results
        
        if on_progress:
            on_progress(f"🕷️ Crawling {len(urls_to_crawl)} selected URLs...")
        
        # Crawl URLs concurrently
        try:
            browser_cfg = BrowserConfig(
                headless=True,
                verbose=False
            )
            crawler_cfg = CrawlerRunConfig(
                word_count_threshold=50,
                excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style'],
                remove_overlay_elements=True
            )
            
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                enriched_count = 0
                
                for result in urls_to_crawl:
                    try:
                        if on_progress:
                            on_progress(f"   🕷️ Crawling: {result.title[:40]}...")
                        
                        crawl_result = await asyncio.wait_for(
                            crawler.arun(url=result.url, config=crawler_cfg),
                            timeout=15
                        )
                        
                        if crawl_result.success and crawl_result.markdown:
                            # Extract relevant content
                            enriched_content = extract_relevant(
                                crawl_result.markdown,
                                query,
                                max_len=2000
                            )
                            
                            if enriched_content and len(enriched_content) > len(result.content):
                                # Update result with richer content
                                result.content = enriched_content
                                result.source = "duckduckgo+crawl4ai"
                                result.score += 0.2  # Boost score for enriched results
                                enriched_count += 1
                                
                                if on_progress:
                                    on_progress(f"   ✅ Enriched: {result.title[:40]} (+{len(enriched_content)} chars)")
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout crawling: {result.url}")
                        if on_progress:
                            on_progress(f"   ⏱️ Timeout: {result.title[:40]}")
                    except Exception as e:
                        logger.warning(f"Error crawling {result.url}: {e}")
                
                if on_progress:
                    on_progress(f"🕷️ Enriched {enriched_count}/{len(urls_to_crawl)} results with detailed content")
                
        except Exception as e:
            logger.error(f"Crawl enrichment failed: {e}")
            if on_progress:
                on_progress(f"⚠️ Crawl enrichment error: {str(e)[:50]}")
        
        return results
    
    async def _select_urls_to_crawl(
        self,
        results: List[SearchResult],
        query: str,
        max_urls: int,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[SearchResult]:
        """Use LLM to intelligently select which URLs to crawl for more detail."""
        if not self.llm or not results:
            return []
        
        # Filter out non-crawlable URLs
        crawlable = [r for r in results if self._is_crawlable_url(r.url)]
        
        if len(crawlable) <= max_urls:
            return crawlable
        
        # Create selection model
        class UrlSelection(BaseModel):
            selected_indices: List[int] = Field(
                description=f"Indices of the {max_urls} most valuable URLs to crawl (0-indexed)"
            )
            reasoning: str = Field(
                description="Brief explanation of selection"
            )
        
        # Format results for LLM
        results_text = "\n".join([
            f"[{i}] {r.title}\n    URL: {r.url}\n    Snippet: {r.content[:150]}..."
            for i, r in enumerate(crawlable[:10])  # Max 10 candidates
        ])
        
        prompt = f"""Select the {max_urls} most valuable URLs to crawl for detailed content.

Original Query: {query}

Available Results:
{results_text}

Selection Criteria:
- Prioritize official documentation and tutorials
- Prefer pages with code examples
- Choose authoritative sources (official docs, reputable blogs)
- Avoid aggregator sites, forums with short answers
- Prefer pages that likely have detailed methodology explanations

Return the indices of the {max_urls} best URLs to crawl."""

        try:
            structured_llm = self.llm.with_structured_output(UrlSelection)
            selection = structured_llm.invoke([
                SystemMessage(content="Select the most valuable URLs to crawl for detailed content."),
                HumanMessage(content=prompt)
            ])
            
            if on_progress:
                on_progress(f"🎯 Selected URLs: {selection.reasoning[:60]}...")
            
            # Return selected results
            selected = []
            for idx in selection.selected_indices[:max_urls]:
                if 0 <= idx < len(crawlable):
                    selected.append(crawlable[idx])
            
            return selected
            
        except Exception as e:
            logger.warning(f"URL selection failed: {e}")
            return []
    
    def _fallback_url_selection(
        self,
        results: List[SearchResult],
        max_urls: int
    ) -> List[SearchResult]:
        """Fallback URL selection based on heuristics."""
        crawlable = [r for r in results if self._is_crawlable_url(r.url)]
        
        # Sort by score and heuristic boosts
        def url_priority(r: SearchResult) -> float:
            score = r.score
            url = r.url.lower()
            
            # Boost official docs
            if any(d in url for d in ['docs.', 'documentation', '.readthedocs.', 'scikit-learn.org', 'pandas.pydata.org']):
                score += 0.3
            # Boost tutorials
            if 'tutorial' in url or 'guide' in url:
                score += 0.2
            # Boost educational
            if any(d in url for d in ['towardsdatascience.com', 'realpython.com', 'geeksforgeeks.org']):
                score += 0.15
            # Penalize Q&A (usually short answers)
            if 'stackoverflow.com' in url or 'quora.com' in url:
                score -= 0.1
            
            return score
        
        sorted_results = sorted(crawlable, key=url_priority, reverse=True)
        return sorted_results[:max_urls]
    
    def _is_crawlable_url(self, url: str) -> bool:
        """Check if URL is suitable for crawling."""
        if not url.startswith('http'):
            return False
        
        blocked = [
            'youtube.com', 'twitter.com', 'facebook.com', 'linkedin.com',
            'instagram.com', 'tiktok.com', 'reddit.com/r/',  # Reddit threads are messy
            '.pdf', '.zip', '.exe', '.dmg',  # Binary files
            'login', 'signin', 'signup',  # Auth pages
        ]
        
        return not any(b in url.lower() for b in blocked)


# =============================================================================
# CORE SEARCH FUNCTIONS
# =============================================================================


# Curated documentation sources for fallback
DOC_SOURCES = {
    "sklearn": "https://scikit-learn.org/stable/user_guide.html",
    "pandas": "https://pandas.pydata.org/docs/user_guide/index.html",
    "seaborn": "https://seaborn.pydata.org/tutorial.html",
    "statsmodels": "https://www.statsmodels.org/stable/user-guide.html",
    "matplotlib": "https://matplotlib.org/stable/tutorials/index.html",
}


def search_duckduckgo(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    on_progress: Optional[Callable[[str], None]] = None
) -> List[SearchResult]:
    """
    Search DuckDuckGo for relevant results.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        region: Region code (wt-wt = worldwide)
        on_progress: Optional callback for progress updates
        
    Returns:
        List of SearchResult objects
    """
    if not DDGS_AVAILABLE:
        logger.error("DuckDuckGo search not available")
        return []
    
    results = []
    
    try:
        if on_progress:
            on_progress(f"Searching DuckDuckGo: {query}")
        
        with DDGS() as ddgs:
            # Text search
            search_results = list(ddgs.text(
                query,
                region=region,
                max_results=max_results
            ))
            
            for i, r in enumerate(search_results):
                result = SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    score=1.0 - (i * 0.1),  # Decreasing score by rank
                    source="duckduckgo"
                )
                results.append(result)
                
                if on_progress:
                    on_progress(f"Found: {result.title[:50]}...")
        
        logger.info(f"🔍 DuckDuckGo: Found {len(results)} results for '{query}'")
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
    
    return results


async def search_duckduckgo_async(
    query: str,
    max_results: int = 5,
    on_progress: Optional[Callable[[str], None]] = None
) -> List[SearchResult]:
    """Async wrapper for DuckDuckGo search."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: search_duckduckgo(query, max_results, on_progress=on_progress)
    )


def select_doc_sources(query: str, max_sources: int = 2) -> List[str]:
    """Select relevant documentation sources based on query keywords."""
    q = query.lower()
    selected = []
    
    if any(k in q for k in ["model", "classify", "regress", "cluster", "predict", "sklearn"]):
        selected.append(DOC_SOURCES["sklearn"])
    if any(k in q for k in ["dataframe", "pandas", "merge", "groupby", "clean"]):
        selected.append(DOC_SOURCES["pandas"])
    if any(k in q for k in ["plot", "visual", "chart", "graph", "heatmap", "seaborn"]):
        selected.append(DOC_SOURCES["seaborn"])
    if any(k in q for k in ["statistic", "hypothesis", "anova", "ttest", "regression"]):
        selected.append(DOC_SOURCES["statsmodels"])
    
    if not selected:
        selected = [DOC_SOURCES["pandas"], DOC_SOURCES["sklearn"]]
    
    return selected[:max_sources]


def extract_relevant(markdown: str, query: str, max_len: int = 1500) -> str:
    """Extract relevant sections from markdown."""
    if not markdown:
        return ""
    
    sections = markdown.split("\n\n")
    query_words = set(query.lower().split())
    
    scored = []
    for section in sections:
        if len(section.strip()) < 30:
            continue
        
        s_lower = section.lower()
        score = sum(1 for w in query_words if w in s_lower)
        
        if "```" in section or "def " in section:
            score += 3
        if "example" in s_lower:
            score += 2
            
        scored.append((score, section))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    parts, length = [], 0
    for _, section in scored:
        if length + len(section) > max_len:
            break
        parts.append(section)
        length += len(section)
    
    return "\n\n".join(parts)


async def crawl_url(
    url: str,
    query: str,
    on_progress: Optional[Callable[[str], None]] = None
) -> Optional[SearchResult]:
    """Crawl a single URL and extract relevant content."""
    if not CRAWL4AI_AVAILABLE:
        return None
    
    try:
        if on_progress:
            on_progress(f"Crawling: {url}")
        
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        crawler_cfg = CrawlerRunConfig(word_count_threshold=100)
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=crawler_cfg),
                timeout=20
            )
            
            if result.success and result.markdown:
                content = extract_relevant(result.markdown, query)
                if content:
                    return SearchResult(
                        url=url,
                        title=result.metadata.get("title", url),
                        content=content,
                        score=len(content) / 1000,
                        source="crawl4ai"
                    )
                    
    except asyncio.TimeoutError:
        logger.warning(f"Timeout crawling: {url}")
    except Exception as e:
        logger.warning(f"Error crawling {url}: {e}")
    
    return None


async def search_analytics_methods(
    query: str,
    max_results: int = 5,
    use_agent: bool = True,
    context: str = "",
    llm_model: str = "qwen3:30b",
    base_url: str = "http://100.91.155.118:11434",
    enrich_with_crawl: bool = True,
    max_crawl_urls: int = 3,
    on_progress: Optional[Callable[[str], None]] = None
) -> WebSearchContext:
    """
    Search for analytics methodology guidance using an intelligent agent.
    
    Args:
        query: Search query
        max_results: Maximum number of results per query
        use_agent: Whether to use LLM agent for query optimization
        context: Additional context for query generation
        llm_model: LLM model for agent
        base_url: LLM base URL
        enrich_with_crawl: Whether to crawl top URLs for full content
        max_crawl_urls: Maximum number of URLs to crawl
        on_progress: Optional callback for progress updates
        
    Returns:
        WebSearchContext with results
    """
    if on_progress:
        on_progress(f"🌐 Starting web search: {query[:50]}...")
    
    # Use agent-based search if available and enabled
    if use_agent and LANGCHAIN_AVAILABLE and DDGS_AVAILABLE:
        agent = SearchAgent(
            llm_model=llm_model,
            base_url=base_url,
            max_iterations=2,
            results_per_query=max_results
        )
        
        return await agent.search(
            query,
            context,
            on_progress,
            enrich_with_crawl=enrich_with_crawl,
            max_crawl_urls=max_crawl_urls
        )
    
    # Fallback to basic search
    ctx = WebSearchContext(query=query)
    
    if not DDGS_AVAILABLE:
        ctx.error = "DuckDuckGo not available"
        return ctx
    
    try:
        results = await search_duckduckgo_async(
            f"{query} python data science",
            max_results=max_results,
            on_progress=on_progress
        )
        ctx.results.extend(results)
        ctx.queries_used.append(query)
        
        # Even without agent, try to enrich with crawl4ai
        if enrich_with_crawl and CRAWL4AI_AVAILABLE and results:
            if on_progress:
                on_progress(f"🕷️ Enriching results with crawl4ai...")
            
            for result in results[:max_crawl_urls]:
                enriched = await crawl_url(result.url, query, on_progress)
                if enriched and len(enriched.content) > len(result.content):
                    result.content = enriched.content
                    result.source = "duckduckgo+crawl4ai"
        
        if on_progress:
            on_progress(f"✅ Found {len(results)} results")
            
    except Exception as e:
        ctx.error = str(e)
        logger.error(f"Search failed: {e}")
    
    return ctx


async def quick_search(
    query: str,
    max_results: int = 3,
    on_progress: Optional[Callable[[str], None]] = None
) -> List[SearchResult]:
    """
    Quick DuckDuckGo-only search for fast results.
    
    Use this for inline searches during workflow execution.
    """
    if not DDGS_AVAILABLE:
        logger.warning("DuckDuckGo not available for quick search")
        return []
    
    return await search_duckduckgo_async(query, max_results, on_progress)


def format_search_results(
    results: List[SearchResult],
    max_chars: int = 2000,
    include_queries: bool = True
) -> str:
    """Format search results into a text summary."""
    if not results:
        return "No search results found."
    
    parts = ["=== Web Search Results ===\n"]
    
    # Group by query if available
    if include_queries:
        queries_used = set(r.query_used for r in results if r.query_used)
        if queries_used:
            parts.append(f"Queries used: {', '.join(queries_used)}\n")
    
    total_len = len(parts[0])
    
    for i, r in enumerate(results, 1):
        entry = f"\n[{i}] {r.title}\nURL: {r.url}\n{r.content[:400]}...\n"
        
        if total_len + len(entry) > max_chars:
            break
            
        parts.append(entry)
        total_len += len(entry)
    
    return "\n".join(parts)


async def search_and_synthesize(
    query: str,
    context: str = "",
    llm_model: str = "qwen3:30b",
    base_url: str = "http://100.91.155.118:11434",
    enrich_with_crawl: bool = True,
    max_crawl_urls: int = 3,
    on_progress: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Perform agent-based search and synthesize results.
    
    Args:
        query: Search query
        context: Additional context for query generation
        llm_model: LLM model for agent
        base_url: LLM base URL
        enrich_with_crawl: Whether to use crawl4ai to get full page content
        max_crawl_urls: Maximum URLs to crawl for enrichment
        on_progress: Optional callback for progress updates
    
    Returns a dictionary with:
    - results: List of SearchResult
    - synthesis: SearchSynthesis object
    - queries_used: List of queries executed
    - formatted_text: Ready-to-use text summary
    """
    agent = SearchAgent(
        llm_model=llm_model,
        base_url=base_url
    )
    
    # Run search with optional crawl enrichment
    ctx = await agent.search(
        query,
        context,
        on_progress,
        enrich_with_crawl=enrich_with_crawl,
        max_crawl_urls=max_crawl_urls
    )
    
    # Synthesize results
    synthesis = agent.synthesize_results(query, ctx.results, on_progress)
    
    # Format for use in prompts
    formatted_parts = [
        f"=== Web Research: {query} ===\n",
        f"Queries executed: {len(ctx.queries_used)}",
        f"Results found: {len(ctx.results)}",
    ]
    
    # Note if results were enriched
    enriched_count = sum(1 for r in ctx.results if r.source == "duckduckgo+crawl4ai")
    if enriched_count > 0:
        formatted_parts.append(f"Enriched with full content: {enriched_count}")
    
    formatted_parts.extend([
        f"Relevance: {synthesis.relevance_score:.0%}\n",
        f"Summary: {synthesis.summary}\n",
        "Key Insights:"
    ])
    
    for insight in synthesis.key_insights:
        formatted_parts.append(f"• {insight}")
    
    if ctx.results:
        formatted_parts.append("\nTop Sources:")
        for r in ctx.results[:3]:
            source_note = " (enriched)" if r.source == "duckduckgo+crawl4ai" else ""
            formatted_parts.append(f"• {r.title}{source_note} ({r.url})")
    
    return {
        "results": ctx.results,
        "synthesis": synthesis,
        "queries_used": ctx.queries_used,
        "formatted_text": "\n".join(formatted_parts),
        "error": ctx.error
    }