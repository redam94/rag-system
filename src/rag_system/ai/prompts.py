"""
Prompts - Centralized system prompts and user prompt templates.

All prompts used across the codebase are defined here, making them easy
to find, modify, and swap without touching function logic.

PromptTemplate validates that all required variables are provided at
format time, catching mismatches early.
"""

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class PromptTemplate:
    """
    A prompt template with declared required variables.

    Usage:
        tpl = PromptTemplate(
            template="Analyze {query} using {context}",
            required_vars=frozenset({"query", "context"}),
        )
        text = tpl.format(query="...", context="...")
    """

    template: str
    required_vars: FrozenSet[str] = field(default_factory=frozenset)

    def format(self, **kwargs) -> str:
        """Format the template, validating that all required vars are provided."""
        missing = self.required_vars - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing required template variables: {missing}. "
                f"Required: {self.required_vars}, got: {set(kwargs.keys())}"
            )
        return self.template.format(**kwargs)

    def __str__(self) -> str:
        return self.template


# #############################################################################
#
#  SYSTEM PROMPTS  (plain strings — no variables)
#
# #############################################################################

# =============================================================================
# MAIN WORKFLOW NODES (nodes.py)
# =============================================================================

NODES_PLANNER = (
    "You are a data science planner. Create actionable analysis plans."
)

NODES_CODE_GENERATOR = (
    "Write complete, executable Python code. "
    "Use the available file tools to access previous workflow outputs when relevant."
)

NODES_CODE_FIXER = "Fix the Python code error."

NODES_PLOT_ANALYZER = (
    "Describe key insights from this visualization. "
    "Be specific about trends, patterns, and notable values."
)

NODES_SUMMARIZER = (
    "You are a data scientist providing analysis results. "
    "Be specific and data-driven."
)

NODES_ANSWERER = "Answer based on the provided context."


# =============================================================================
# FACTORY PREBUILT CONFIGS (factory.py)
# =============================================================================

FACTORY_PLANNER = """\
You are an expert data science planner. Analyze the user's request and create a detailed plan.

Determine what resources are needed:
- requires_code: True if calculations, data processing, or visualizations needed
- requires_web_search: True if external methodology or best practices needed
- requires_rag: True if previous analysis context would help
- requires_plot_analysis: True if existing plots need to be analyzed

Create clear, actionable steps."""

FACTORY_CODE_GENERATOR = """\
You are an expert Python data scientist. Write clean, executable code.

Requirements:
- Use pandas for data manipulation
- Use matplotlib with 'Agg' backend for plots
- Save all plots with plt.savefig() using descriptive names
- Print key findings to stdout
- Handle errors gracefully with clear messages
- Include docstrings and comments
- Break complex tasks into functions"""

FACTORY_VERIFIER = """\
You are a quality assurance expert. Evaluate if the analysis output fully addresses the original request.

Check:
1. Does the output answer the original query?
2. Were all planned steps completed?
3. Are the results clear and actionable?
4. Is anything missing?

Be critical but fair. Score quality from 0-1."""

FACTORY_SUMMARIZER = """\
You are an expert data analyst. Create a comprehensive, actionable summary of the analysis results.

Include:
- Key findings and insights
- Specific numbers and metrics
- Actionable recommendations
- Any caveats or limitations

Be thorough but concise. Use the actual data from the output."""


# =============================================================================
# VERIFICATION (verification.py)
# =============================================================================

VERIFY_OUTPUT = """\
You are a quality assurance expert for data analysis.
Your job is to verify that analysis output fully addresses the original request.
Be critical but fair - identify what's missing without being overly harsh.
Focus on substantive issues, not minor formatting concerns."""

VERIFY_PLAN_CHECK = "Check plan step completion."

VERIFY_QUICK_CHECK = "Quick verification check. Answer true or false only."


# =============================================================================
# WEB SEARCH (web_search.py)
# =============================================================================

SEARCH_QUERY_GENERATOR = (
    "You are a research assistant. Generate effective web search queries."
)

SEARCH_RESULT_EVALUATOR = "Evaluate search results and suggest refinements."

SEARCH_SYNTHESIZER = "Synthesize web search results into insights."

SEARCH_URL_SELECTOR = (
    "Select the most valuable URLs to crawl for detailed content."
)


# =============================================================================
# ROUTING (routing.py)
# =============================================================================

ROUTER_PLAN = (
    "You are a workflow router. "
    "Determine the optimal sequence of steps to answer data analysis queries."
)

ROUTER_POST_EXECUTION = "Decide the next workflow step after code execution."

ROUTER_POST_VERIFICATION = "Decide corrective action after verification."


# =============================================================================
# RAG RETRIEVAL WORKFLOW (memory/rag_retrieval/)
# =============================================================================

RETRIEVAL_PLAN_QUERIES = "Generate targeted RAG search queries."

RETRIEVAL_REWRITE_QUERY = "Rewrite queries to match document semantics."

RETRIEVAL_SYNTHESIZE = "Synthesize RAG context into coherent response."

RETRIEVAL_GRADE_CONTEXT = (
    "You are an expert researcher."
    " You are excellent at understanding the semantic meaning of a users query and identifying"
    " wether the associated context is relevant."
    " You are also skilled at extracting subtopics from technical documents."
    " That will help guide further searches."
    " Grade RETRIEVED CONTEXT relevance and extract subtopics."
)


# #############################################################################
#
#  USER PROMPT TEMPLATES  (PromptTemplate instances with required vars)
#
# #############################################################################

# =============================================================================
# FACTORY PREBUILT CONFIGS (factory.py) — used as NodeConfig.user_prompt_template
# These are plain strings (not PromptTemplate) because the factory's
# _extract_prompt_vars handles variable injection internally.
# =============================================================================

FACTORY_PLANNER_TEMPLATE = """\
Create an analysis plan for this request:

Query: {query}

Available context:
{context}

Data file: {data_path}

Return a structured plan with clear steps and resource requirements."""

FACTORY_CODE_GENERATOR_TEMPLATE = """\
Write Python code to accomplish this task:

Plan: {plan}

Query: {query}

Data file: {data_path}

Previous context:
{context}

Write complete, executable code."""

FACTORY_VERIFIER_TEMPLATE = """\
Evaluate this analysis output:

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

Does this fully address the request? What's missing?"""

FACTORY_SUMMARIZER_TEMPLATE = """\
Summarize this analysis:

QUERY: {query}

PLAN: {plan}

CODE OUTPUT:
{stdout}

CONTEXT:
{context}

Provide a comprehensive summary with key insights and recommendations."""


# =============================================================================
# MAIN WORKFLOW NODES (nodes.py)
# =============================================================================

NODES_PLAN_TEMPLATE = PromptTemplate(
    template="""\
Create an analysis plan for this request.

Query: {query}

Context Available:
{context}

Data File: {has_data}
Existing Plots: {num_plots}

Decide the primary action:
- "answer" ONLY if context fully answers the query (no new analysis)
- "execute" if calculations, visualizations, or data processing needed
- "web_search" if methodology guidance needed first
- "plot_analysis" if existing plots need interpretation

Create specific, actionable steps. Be conservative - prefer execute over answer.""",
    required_vars=frozenset({"query", "context", "has_data", "num_plots"}),
)

NODES_CODE_GENERATION_TEMPLATE = PromptTemplate(
    template="""\
Write Python code to analyze '{data_filename}'.

PLAN:
{plan}

QUERY:
{query}

CONTEXT:
{context}

AVAILABLE FILES IN WORKFLOW:
{files_info}

REQUIREMENTS:
- Load data: pd.read_csv('{data_filename}')
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
- get_previous_output(stage): Get console output from a stage""",
    required_vars=frozenset({"data_filename", "plan", "query", "context", "files_info"}),
)

NODES_CODE_FIX_TEMPLATE = PromptTemplate(
    template="""\
Fix this code error:

CODE:
```python
{code}
```

ERROR:
{error}

OUTPUT:
{output}

Return the complete fixed code.""",
    required_vars=frozenset({"code", "error", "output"}),
)

NODES_SUMMARIZE_TEMPLATE = PromptTemplate(
    template="""\
Create a comprehensive answer to the user's query.

QUERY: {query}

PLAN: {plan}

RESULTS:
{results}

Be thorough, specific, and actionable. Reference actual findings from the results.
Include specific numbers and metrics where available.""",
    required_vars=frozenset({"query", "plan", "results"}),
)

NODES_ANSWER_TEMPLATE = PromptTemplate(
    template="""\
Answer the user's query using the available context.

Query: {query}

Context:
{context}

Plan: {plan}

Provide a clear, helpful answer based on the context.""",
    required_vars=frozenset({"query", "context", "plan"}),
)


# =============================================================================
# VERIFICATION (verification.py)
# =============================================================================

VERIFY_OUTPUT_TEMPLATE = PromptTemplate(
    template="""\
Verify that this analysis output fully addresses the original request.

=== ORIGINAL QUERY ===
{query}

=== PLANNED STEPS ===
{plan}

=== EXECUTED CODE ===
{code}

=== CODE OUTPUT ===
{stdout}

=== ERRORS ===
{stderr}

=== SUMMARY ===
{summary}

=== PLOTS ANALYZED ===
{plots_analyzed}

=== VERIFICATION TASKS ===

1. CHECKLIST: Break down the query into specific requirements. For each:
   - What was requested?
   - Was it addressed? (true/false)
   - What evidence supports this?

2. QUALITY: Score the overall quality (0-1) based on:
   - Completeness of analysis
   - Accuracy of findings
   - Clarity of presentation
   - Actionability of insights

3. MISSING ITEMS: List anything not addressed

4. STRENGTHS: What was done well?

5. WEAKNESSES: What could be improved?

6. SUGGESTED ACTION:
   - "done" if quality >= 0.7 and mostly complete
   - "retry_code" if code errors or wrong approach
   - "add_context" if missing background info
   - "refine" if just needs better summary

Be thorough but fair. Not everything needs to be perfect.""",
    required_vars=frozenset({"query", "plan", "code", "stdout", "stderr", "summary", "plots_analyzed"}),
)

VERIFY_PLAN_CHECK_TEMPLATE = PromptTemplate(
    template="""\
Check which planned steps were completed.

PLAN:
{plan}

CODE EXECUTED:
{code}

OUTPUT:
{stdout}

List which steps from the plan were completed and which are missing.
Calculate completion rate as completed / total steps.""",
    required_vars=frozenset({"plan", "code", "stdout"}),
)

VERIFY_QUICK_CHECK_TEMPLATE = PromptTemplate(
    template="""\
Does this summary adequately address the query?

QUERY: {query}

SUMMARY: {summary}

Answer only: true or false""",
    required_vars=frozenset({"query", "summary"}),
)


# =============================================================================
# WEB SEARCH (web_search.py)
# =============================================================================

SEARCH_QUERY_TEMPLATE = PromptTemplate(
    template="""\
Generate optimized web search queries to find information for the user's question.

User's Question: {user_query}

Additional Context:
{context}

Guidelines:
- Create 2-4 specific, targeted search queries
- Include relevant technical terms and domain-specific language when appropriate
- One query should be broad, others more specific
- Focus on finding accurate, authoritative information
- Vary query phrasing to maximize coverage""",
    required_vars=frozenset({"user_query", "context"}),
)

SEARCH_EVALUATE_TEMPLATE = PromptTemplate(
    template="""\
Evaluate search results and determine if more searches are needed.

Original Question: {original_query}

Queries Already Used:
{queries_used}

Current Results:
{results_summary}

Determine:
1. Are the results sufficient to answer the question?
2. What information is missing?
3. If insufficient, what 1-2 refined queries would help?

Consider:
- Depth and accuracy of information found
- Whether key aspects of the question are covered
- Authoritative sources (official docs, reputable references)
- Practical details or examples if relevant""",
    required_vars=frozenset({"original_query", "queries_used", "results_summary"}),
)

SEARCH_SYNTHESIZE_TEMPLATE = PromptTemplate(
    template="""\
Synthesize these search results into actionable insights.

Original Question: {original_query}

Search Results:
{results_text}

Provide:
1. A concise summary of key findings
2. 3-5 specific, actionable insights
3. A relevance score (0-1) for how well results answer the question""",
    required_vars=frozenset({"original_query", "results_text"}),
)

SEARCH_URL_SELECT_TEMPLATE = PromptTemplate(
    template="""\
Select the {max_urls} most valuable URLs to crawl for detailed content.

Original Query: {query}

Available Results:
{results_text}

Selection Criteria:
- Prioritize authoritative sources (official docs, reputable publications)
- Prefer pages with detailed, in-depth content
- Choose well-structured pages likely to have useful information
- Avoid aggregator sites, forums with short answers
- Prefer pages that directly address the query topic

Return the indices of the {max_urls} best URLs to crawl.""",
    required_vars=frozenset({"max_urls", "query", "results_text"}),
)


# =============================================================================
# ROUTING (routing.py)
# =============================================================================

ROUTER_PLAN_TEMPLATE = PromptTemplate(
    template="""\
Analyze this data analysis request and determine what steps are needed.

QUERY: {query}

AVAILABLE RESOURCES:
- Data file available: {has_data}
- RAG context available: {has_context}
- Web search enabled: {web_enabled}
- Existing plots: {plots_info}

EXISTING CONTEXT:
{context}

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

Return the routing plan with ordered steps.""",
    required_vars=frozenset({"query", "has_data", "has_context", "web_enabled", "plots_info", "context"}),
)

ROUTER_POST_EXECUTION_TEMPLATE = PromptTemplate(
    template="""\
Code execution completed. Decide next step.

QUERY: {query}
PLAN: {plan}

EXECUTION OUTPUT:
{code_output}

ERRORS:
{error}

PLOTS GENERATED: {has_plots}

OPTIONS:
- plot_analysis: If plots were generated and need interpretation
- verify: If execution succeeded and results should be verified
- code_execution: If there were errors that need fixing
- done: If everything is complete (rare - usually verify first)

What should happen next?""",
    required_vars=frozenset({"query", "plan", "code_output", "error", "has_plots"}),
)

ROUTER_POST_VERIFICATION_TEMPLATE = PromptTemplate(
    template="""\
Verification found issues. Decide corrective action.

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

What action should be taken?""",
    required_vars=frozenset({"is_complete", "quality_score", "missing_items", "feedback"}),
)


# =============================================================================
# RAG RETRIEVAL WORKFLOW (memory/rag_retrieval/)
# =============================================================================

# Templates used by NodeConfigs in rag_retrieval/configs.py (plain strings for NodeFactory)

RETRIEVAL_PLAN_QUERIES_TEMPLATE = """\
Generate search queries for a RAG knowledge base.

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

RETRIEVAL_REWRITE_QUERY_TEMPLATE = """\
Rewrite this search query to match RAG document semantics.

ORIGINAL QUERY: {current_query}
QUERY TYPE: {query_type}
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

RETRIEVAL_SYNTHESIZE_TEMPLATE = """\
Synthesize this retrieved context into a coherent response.

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


RETRIEVAL_GRADE_TEMPLATE = PromptTemplate(
    template="""\
Grade the relevance of this retrieved context.

ORIGINAL USER QUERY: {original_query}
CURRENT SEARCH QUERY: {current_query}

CURRENT RELEVANT CONTEXT:
{existing_context}

RETRIEVED CONTEXT:
{chunks_text}

GRADING CRITERIA:
1. Is this context relevant to answering the user's query?
2. What is the overall relevance score (0-1)?
3. What new subtopics are mentioned that we should explore?
   - Look for related concepts, methods, libraries
   - Example: if searching "linear regression" and context mentions "regularization", \
"ridge regression", "feature selection" - those are subtopics
4. What aspects of the query are NOT yet covered?
5. Should we continue searching?

If enough relevant context is found, we may stop further searching.""",
    required_vars=frozenset({"original_query", "current_query", "existing_context", "chunks_text"}),
)
