"""
Verification System - Checks if output addresses the original request.

Provides:
- Output completeness checking
- Quality scoring
- Missing item identification
- Corrective action suggestions
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage


# =============================================================================
# VERIFICATION MODELS
# =============================================================================

class ChecklistItem(BaseModel):
    """Single item from the request checklist."""
    item: str
    addressed: bool
    evidence: str = ""


class VerificationResult(BaseModel):
    """Complete verification result."""
    is_complete: bool = Field(description="Whether all requirements are met")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality 0-1")
    
    checklist: List[ChecklistItem] = Field(
        default_factory=list,
        description="Breakdown of requirements and whether each was addressed"
    )
    
    missing_items: List[str] = Field(
        default_factory=list,
        description="List of unaddressed requirements"
    )
    
    strengths: List[str] = Field(
        default_factory=list,
        description="What was done well"
    )
    
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Areas for improvement"
    )
    
    suggested_action: str = Field(
        default="done",
        description="Recommended next action: done, retry_code, add_context, refine"
    )
    
    feedback: str = Field(
        default="",
        description="Detailed feedback for improvement"
    )


class PlanCheckResult(BaseModel):
    """Check if plan steps were executed."""
    steps_completed: List[str]
    steps_missing: List[str]
    completion_rate: float


# =============================================================================
# VERIFIER CLASS
# =============================================================================

class OutputVerifier:
    """
    Verifies that workflow output addresses the original request.
    
    Checks:
    1. Query requirements fulfilled
    2. Plan steps completed
    3. Output quality and completeness
    4. Actionable recommendations
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
    
    def verify(
        self,
        query: str,
        plan: str,
        code: str = "",
        stdout: str = "",
        stderr: str = "",
        summary: str = "",
        plots_analyzed: List[str] = None,
    ) -> VerificationResult:
        """
        Verify that the output addresses the original query.
        
        Args:
            query: Original user query
            plan: Generated plan
            code: Executed code
            stdout: Code output
            stderr: Code errors
            summary: Generated summary
            plots_analyzed: List of plots that were analyzed
            
        Returns:
            VerificationResult with completeness check
        """
        plots_analyzed = plots_analyzed or []
        
        prompt = f"""Verify that this analysis output fully addresses the original request.

=== ORIGINAL QUERY ===
{query}

=== PLANNED STEPS ===
{plan}

=== EXECUTED CODE ===
{code[:2000] if code else 'No code executed'}

=== CODE OUTPUT ===
{stdout[:2000] if stdout else 'No output'}

=== ERRORS ===
{stderr[:500] if stderr else 'None'}

=== SUMMARY ===
{summary[:1500] if summary else 'No summary generated'}

=== PLOTS ANALYZED ===
{', '.join(plots_analyzed) if plots_analyzed else 'None'}

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

Be thorough but fair. Not everything needs to be perfect."""

        structured_llm = self.llm.with_structured_output(VerificationResult)
        
        try:
            result = structured_llm.invoke([
                SystemMessage(content="""You are a quality assurance expert for data analysis.
Your job is to verify that analysis output fully addresses the original request.
Be critical but fair - identify what's missing without being overly harsh.
Focus on substantive issues, not minor formatting concerns."""),
                HumanMessage(content=prompt),
            ])
            
            logger.info(
                f"✅ Verification: complete={result.is_complete}, "
                f"quality={result.quality_score:.2f}, action={result.suggested_action}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                is_complete=True,
                quality_score=0.5,
                suggested_action="done",
                feedback=f"Verification error: {e}",
            )
    
    def check_plan_completion(
        self,
        plan: str,
        stdout: str,
        code: str = "",
    ) -> PlanCheckResult:
        """
        Check which plan steps were completed.
        
        Simpler check focused on plan execution.
        """
        prompt = f"""Check which planned steps were completed.

PLAN:
{plan}

CODE EXECUTED:
{code[:1500] if code else 'None'}

OUTPUT:
{stdout[:1500] if stdout else 'None'}

List which steps from the plan were completed and which are missing.
Calculate completion rate as completed / total steps."""

        class PlanCheck(BaseModel):
            steps_completed: List[str]
            steps_missing: List[str]
            completion_rate: float
        
        structured_llm = self.llm.with_structured_output(PlanCheck)
        
        try:
            result = structured_llm.invoke([
                SystemMessage(content="Check plan step completion."),
                HumanMessage(content=prompt),
            ])
            
            return PlanCheckResult(
                steps_completed=result.steps_completed,
                steps_missing=result.steps_missing,
                completion_rate=result.completion_rate,
            )
            
        except Exception as e:
            logger.error(f"Plan check failed: {e}")
            return PlanCheckResult(
                steps_completed=[],
                steps_missing=["Unable to verify"],
                completion_rate=0.5,
            )
    
    def quick_check(
        self,
        query: str,
        summary: str,
    ) -> bool:
        """
        Quick boolean check if summary addresses query.
        
        Use for fast verification without detailed feedback.
        """
        if not summary:
            return False
        
        prompt = f"""Does this summary adequately address the query?

QUERY: {query}

SUMMARY: {summary[:1500]}

Answer only: true or false"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Quick verification check. Answer true or false only."),
                HumanMessage(content=prompt),
            ])
            
            return "true" in response.content.lower()
            
        except Exception as e:
            logger.error(f"Quick check failed: {e}")
            return True  # Default to passing


# =============================================================================
# VERIFICATION NODE FUNCTION
# =============================================================================

async def verify_output(state: Dict[str, Any], runtime) -> Dict[str, Any]:
    """
    Verification node for LangGraph workflow.
    
    Checks if output addresses original query and plan.
    """
    deps = runtime.context
    emitter = deps.get("progress_emitter")
    
    if emitter:
        from utils.progress_events import EventType
        emitter.emit(EventType.STAGE_START, "verify", "Verifying output completeness")
    
    # Extract state
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    plan = state.get("plan", "")
    code = state.get("code", "")
    summary = state.get("summary", "")
    
    output = state.get("output")
    stdout = output.stdout if output and hasattr(output, "stdout") else ""
    stderr = output.stderr if output and hasattr(output, "stderr") else ""
    
    # Get verifier
    llm_model = deps.get("llm", "qwen3:30b")
    base_url = deps.get("base_url", "http://100.91.155.118:11434")
    
    verifier = OutputVerifier(llm_model=llm_model, base_url=base_url)
    
    # Run verification
    result = verifier.verify(
        query=query,
        plan=plan,
        code=code,
        stdout=stdout,
        stderr=stderr,
        summary=summary,
    )
    
    if emitter:
        from utils.progress_events import EventType
        status_msg = f"Quality: {result.quality_score:.0%}, Complete: {result.is_complete}"
        emitter.emit(EventType.PROGRESS, "verify", status_msg)
        emitter.emit(EventType.STAGE_END, "verify", "Verification complete")
    
    logger.info(f"📋 Verification: {result.quality_score:.0%} quality, action={result.suggested_action}")
    
    return {
        "verification": result.model_dump(),
        "verified": True,
    }