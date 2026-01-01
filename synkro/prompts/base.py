"""Customizable prompt classes for building your own generation pipelines."""

from pydantic import BaseModel, Field
from synkro.prompts.templates import (
    SYSTEM_PROMPT,
    SCENARIO_GENERATOR_PROMPT,
    BATCHED_RESPONSE_PROMPT,
    BATCHED_GRADER_PROMPT,
    BATCHED_REFINER_PROMPT,
    POLICY_PLANNING_PROMPT,
)


class SystemPrompt(BaseModel):
    """The system prompt that defines the expert's role and behavior."""

    template: str = Field(default=SYSTEM_PROMPT)

    def render(self, **kwargs) -> str:
        """Render the prompt with any custom variables."""
        return self.template.format(**kwargs) if kwargs else self.template


class ScenarioPrompt(BaseModel):
    """Prompt for generating scenarios from policy documents."""

    template: str = Field(default=SCENARIO_GENERATOR_PROMPT)

    def render(self, policy: str, count: int, category: str | None = None) -> str:
        """
        Render the scenario generation prompt.

        Args:
            policy: The policy text
            count: Number of scenarios to generate
            category: Optional category to focus scenarios on
        """
        prompt = f"{self.template}\n\nPOLICY:\n{policy}\n\nGenerate exactly {count} scenarios."
        if category:
            prompt += f"\n\nFocus on scenarios related to: {category}"
        return prompt


class ResponsePrompt(BaseModel):
    """Prompt for generating responses to scenarios."""

    template: str = Field(default=BATCHED_RESPONSE_PROMPT)
    system_prompt: str = Field(default=SYSTEM_PROMPT)

    def render(self, scenarios: list[dict], policy: str) -> str:
        """
        Render the response generation prompt.

        Args:
            scenarios: List of scenario dicts with 'description' and 'context'
            policy: The policy text for grounding responses
        """
        scenarios_text = "\n\n".join(
            f"SCENARIO {i}:\n{s['description']}\n\nCONTEXT:\n{s['context']}"
            for i, s in enumerate(scenarios)
        )

        return f"""{self.template}

SYSTEM PROMPT TO USE:
{self.system_prompt}

POLICY:
{policy}

SCENARIOS:
{scenarios_text}"""


class GradePrompt(BaseModel):
    """Prompt for grading response quality."""

    template: str = Field(default=BATCHED_GRADER_PROMPT)

    def render(self, responses: list[dict], policy: str) -> str:
        """
        Render the grading prompt.

        Args:
            responses: List of response dicts with messages
            policy: The policy text to grade against
        """
        responses_text = "\n\n".join(
            f"RESPONSE {i}:\n{r.get('assistant_message', r.get('messages', [{}])[-1].get('content', ''))}"
            for i, r in enumerate(responses)
        )

        return f"""{self.template}

POLICY:
{policy}

RESPONSES TO GRADE:
{responses_text}"""


class RefinePrompt(BaseModel):
    """Prompt for refining failed responses."""

    template: str = Field(default=BATCHED_REFINER_PROMPT)
    system_prompt: str = Field(default=SYSTEM_PROMPT)

    def render(self, failed_items: list[dict], policy: str) -> str:
        """
        Render the refinement prompt.

        Args:
            failed_items: List of dicts with 'scenario', 'response', and 'feedback'
            policy: The policy text
        """
        items_text = "\n\n".join(
            f"""SCENARIO {i}:
{item['scenario']}

ORIGINAL RESPONSE:
{item['response']}

GRADER FEEDBACK:
- Policy Violations: {item.get('policy_violations', [])}
- Missing Citations: {item.get('missing_citations', [])}
- Incomplete Reasoning: {item.get('incomplete_reasoning', [])}
- Vague Recommendations: {item.get('vague_recommendations', [])}
- Summary: {item.get('feedback', '')}"""
            for i, item in enumerate(failed_items)
        )

        return f"""{self.template}

SYSTEM PROMPT TO USE:
{self.system_prompt}

POLICY:
{policy}

ITEMS TO REFINE:
{items_text}"""


class PlanPrompt(BaseModel):
    """Prompt for planning generation categories."""

    template: str = Field(default=POLICY_PLANNING_PROMPT)

    def render(self, policy: str, target_traces: int) -> str:
        """
        Render the planning prompt.

        Args:
            policy: The policy text to analyze
            target_traces: Target number of traces to generate
        """
        return f"""{self.template}

POLICY/DOMAIN SPECIFICATION:
{policy}

TARGET TRACES: {target_traces}

Respond with a JSON object containing:
- "categories": array of category objects with "name", "description", and "traces"
- "reasoning": explanation of your analysis and category choices"""

