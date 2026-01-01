"""Grading of generated traces for quality control."""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Trace, GradeResult
from synkro.prompts.templates import BATCHED_GRADER_PROMPT
from synkro.schemas import SingleGrade
from synkro.parsers import parse_batched_grades


class Grader:
    """
    Grades generated traces for quality and policy compliance.

    Uses an LLM to evaluate each trace against strict criteria:
    - Policy compliance
    - Proper citations
    - Complete reasoning
    - Actionable recommendations

    Examples:
        >>> grader = Grader()
        >>> result = await grader.grade(trace, policy.text)
        >>> if result.passed:
        ...     print("Trace passes quality checks!")
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O):
        """
        Initialize the grader.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (recommend stronger model for grading)
        """
        self.llm = llm or LLM(model=model)

    async def grade(self, trace: Trace, policy_text: str) -> GradeResult:
        """
        Grade a single trace.

        Args:
            trace: The trace to grade
            policy_text: The policy text to grade against

        Returns:
            GradeResult with pass/fail and feedback
        """
        prompt = f"""You are a strict evaluator. Grade this response.

A response PASSES only if ALL are true:
1. Policy Compliant - Every recommendation follows the policy exactly
2. Fully Supported - Every claim backed by specific policy section
3. Properly Cited - All relevant policy sections referenced
4. Complete Reasoning - Chain of thought has no gaps
5. Actionable & Specific - Recommendations are concrete, not vague

SCENARIO:
{trace.scenario.description}

POLICY:
{policy_text}

RESPONSE TO GRADE:
{trace.assistant_message}

Grade this response."""

        try:
            # Use structured output for reliable grading
            parsed = await self.llm.generate_structured(prompt, SingleGrade)
            return GradeResult(
                passed=parsed.passed,
                issues=(
                    parsed.policy_violations
                    + parsed.missing_citations
                    + parsed.incomplete_reasoning
                    + parsed.vague_recommendations
                ),
                feedback=parsed.feedback,
            )
        except Exception:
            # Fallback: assume fail if we can't parse
            return GradeResult(
                passed=False,
                issues=["Unable to parse grade response"],
                feedback="Grading failed - unable to parse response",
            )

    async def grade_batch(
        self, traces: list[Trace], policy_text: str
    ) -> list[GradeResult]:
        """
        Grade multiple traces.

        Args:
            traces: List of traces to grade
            policy_text: The policy text to grade against

        Returns:
            List of GradeResults in same order as input
        """
        results = []

        for trace in traces:
            result = await self.grade(trace, policy_text)
            results.append(result)

        return results

    async def grade_batch_parallel(
        self, traces: list[Trace], policy_text: str
    ) -> list[GradeResult]:
        """
        Grade multiple traces in parallel.

        More efficient for large batches but uses more API calls concurrently.

        Args:
            traces: List of traces to grade
            policy_text: The policy text to grade against

        Returns:
            List of GradeResults in same order as input
        """
        import asyncio

        tasks = [self.grade(trace, policy_text) for trace in traces]
        return await asyncio.gather(*tasks)

