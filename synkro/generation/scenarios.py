"""Scenario generation from policy documents."""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Scenario, Category
from synkro.prompts.templates import SCENARIO_GENERATOR_PROMPT, CATEGORY_SCENARIO_PROMPT
from synkro.schemas import ScenariosArray


class ScenarioGenerator:
    """
    Generates realistic scenarios from policy documents.

    Creates diverse scenarios that test different aspects of policy
    understanding and compliance.

    Examples:
        >>> gen = ScenarioGenerator()
        >>> scenarios = await gen.generate(policy.text, count=50)
        >>> for s in scenarios:
        ...     print(s.description)
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O_MINI):
        """
        Initialize the scenario generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model)
        self.prompt_template = SCENARIO_GENERATOR_PROMPT

    async def generate(
        self,
        policy_text: str,
        count: int,
        category: Category | None = None,
    ) -> list[Scenario]:
        """
        Generate scenarios from the policy.

        Args:
            policy_text: The policy text
            count: Number of scenarios to generate
            category: Optional category to focus on

        Returns:
            List of generated scenarios
        """
        if category:
            prompt = self._build_category_prompt(policy_text, count, category)
        else:
            prompt = self._build_general_prompt(policy_text, count)

        # Use structured output for reliable scenario generation
        parsed = await self.llm.generate_structured(prompt, ScenariosArray)
        return [
            Scenario(
                description=s.scenario,
                context=s.context,
                category=category.name if category else None,
            )
            for s in parsed.scenarios[:count]
        ]

    def _build_general_prompt(self, policy_text: str, count: int) -> str:
        """Build prompt for general scenario generation."""
        return f"""{self.prompt_template}

POLICY:
{policy_text}

Generate exactly {count} diverse scenarios."""

    def _build_category_prompt(
        self, policy_text: str, count: int, category: Category
    ) -> str:
        """Build prompt for category-specific scenario generation."""
        return f"""{CATEGORY_SCENARIO_PROMPT}

Category: {category.name}
Description: {category.description}

POLICY:
{policy_text}

Generate exactly {count} scenarios for the "{category.name}" category."""

