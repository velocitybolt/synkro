"""Refinement of failed traces based on grader feedback."""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Trace, GradeResult, Message
from synkro.prompts.templates import BATCHED_REFINER_PROMPT, SYSTEM_PROMPT
from synkro.parsers import parse_single_response, extract_content


class Refiner:
    """
    Refines traces that failed grading.

    Takes failed traces and their grader feedback and generates
    improved versions that address the issues.

    Examples:
        >>> refiner = Refiner()
        >>> improved = await refiner.refine(failed_trace, grade_result, policy.text)
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O_MINI):
        """
        Initialize the refiner.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model)
        self.prompt_template = BATCHED_REFINER_PROMPT

    async def refine(
        self, trace: Trace, grade: GradeResult, policy_text: str
    ) -> Trace:
        """
        Refine a failed trace based on grader feedback.

        Args:
            trace: The trace that failed grading
            grade: The grade result with feedback
            policy_text: The policy text

        Returns:
            New trace with improved response
        """
        prompt = self._build_prompt(trace, grade, policy_text)

        response = await self.llm.generate(prompt)
        parsed = parse_single_response(response)

        if parsed and len(parsed.messages) >= 3:
            messages = [
                Message(role=m.role, content=m.content) for m in parsed.messages
            ]
        else:
            # Fallback: construct from response
            content = extract_content(response)
            messages = [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=f"Scenario: {trace.scenario.description}\n\nContext: {trace.scenario.context}",
                ),
                Message(role="assistant", content=content),
            ]

        return Trace(messages=messages, scenario=trace.scenario)

    def _build_prompt(
        self, trace: Trace, grade: GradeResult, policy_text: str
    ) -> str:
        """Build the refinement prompt."""
        return f"""You are improving a response that failed quality checks.

SCENARIO:
{trace.scenario.description}

CONTEXT:
{trace.scenario.context}

ORIGINAL RESPONSE:
{trace.assistant_message}

GRADER FEEDBACK:
Issues: {', '.join(grade.issues) if grade.issues else 'None listed'}
Summary: {grade.feedback}

POLICY:
{policy_text}

Generate an IMPROVED response that fixes all the issues. Output a JSON object:
{{
  "messages": [
    {{"role": "system", "content": "<system prompt>"}},
    {{"role": "user", "content": "<the scenario>"}},
    {{"role": "assistant", "content": "<your IMPROVED response>"}}
  ]
}}

The improved response must:
- Fix all policy violations
- Add missing citations
- Complete reasoning with no gaps
- Make recommendations specific and actionable
- Keep what was correct from the original

Respond with ONLY the JSON object."""

    async def refine_batch(
        self,
        traces: list[Trace],
        grades: list[GradeResult],
        policy_text: str,
    ) -> list[Trace]:
        """
        Refine multiple failed traces.

        Args:
            traces: List of traces that failed grading
            grades: Corresponding grade results
            policy_text: The policy text

        Returns:
            List of refined traces
        """
        refined = []

        for trace, grade in zip(traces, grades):
            if not grade.passed:
                improved = await self.refine(trace, grade, policy_text)
                refined.append(improved)
            else:
                refined.append(trace)

        return refined

