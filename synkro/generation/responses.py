"""Response generation for scenarios."""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Scenario, Trace, Message
from synkro.prompts.templates import BATCHED_RESPONSE_PROMPT, SYSTEM_PROMPT
from synkro.schemas import SingleResponse
from synkro.parsers import parse_batched_responses, extract_content


class ResponseGenerator:
    """
    Generates expert responses for scenarios.

    Creates comprehensive, policy-grounded responses that demonstrate
    deep domain understanding.

    Examples:
        >>> gen = ResponseGenerator()
        >>> traces = await gen.generate(policy.text, scenarios)
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O_MINI):
        """
        Initialize the response generator.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model)

    async def generate(
        self,
        policy_text: str,
        scenarios: list[Scenario],
    ) -> list[Trace]:
        """
        Generate responses for scenarios.

        Args:
            policy_text: The policy text
            scenarios: List of scenarios to respond to

        Returns:
            List of traces with generated responses
        """
        traces = []

        # Generate responses one at a time for better quality
        for scenario in scenarios:
            trace = await self._generate_single(policy_text, scenario)
            traces.append(trace)

        return traces

    async def _generate_single(
        self,
        policy_text: str,
        scenario: Scenario,
    ) -> Trace:
        """Generate a single trace for one scenario."""
        prompt = f"""You are a domain expert generating a training example.

Given the scenario and policy below, create a complete training example.

The assistant response must:
- Start with <reasoning> tags showing your thought process
- Cite specific policy sections that apply
- Give specific, actionable recommendations
- Address all aspects of the scenario
- Acknowledge edge cases and complications

SCENARIO:
{scenario.description}

CONTEXT:
{scenario.context}

POLICY:
{policy_text}

Generate exactly 3 messages: system, user, and assistant."""

        # Use structured output for reliable JSON
        parsed = await self.llm.generate_structured(prompt, SingleResponse)
        messages = [
            Message(role=m.role, content=m.content) for m in parsed.messages
        ]

        return Trace(messages=messages, scenario=scenario)

    async def generate_batch(
        self,
        policy_text: str,
        scenarios: list[Scenario],
        batch_size: int = 10,
    ) -> list[Trace]:
        """
        Generate responses in batches.

        More efficient than single generation for large numbers of scenarios.

        Args:
            policy_text: The policy text
            scenarios: List of scenarios to respond to
            batch_size: Number of scenarios per batch

        Returns:
            List of traces with generated responses
        """
        traces = []

        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i : i + batch_size]
            batch_traces = await self._generate_batch(policy_text, batch)
            traces.extend(batch_traces)

        return traces

    async def _generate_batch(
        self,
        policy_text: str,
        scenarios: list[Scenario],
    ) -> list[Trace]:
        """Generate traces for a batch of scenarios."""
        scenarios_text = "\n\n".join(
            f"SCENARIO {i}:\n{s.description}\n\nCONTEXT:\n{s.context}"
            for i, s in enumerate(scenarios)
        )

        prompt = f"""{BATCHED_RESPONSE_PROMPT}

SYSTEM PROMPT TO USE:
{SYSTEM_PROMPT}

POLICY:
{policy_text}

SCENARIOS:
{scenarios_text}"""

        response = await self.llm.generate(prompt)
        from synkro.schemas import ScenarioOutput

        scenario_outputs = [
            ScenarioOutput(scenario=s.description, context=s.context) for s in scenarios
        ]
        parsed = parse_batched_responses(response, len(scenarios), scenario_outputs)

        traces = []
        for i, p in enumerate(parsed):
            scenario = scenarios[min(p["index"], len(scenarios) - 1)]
            messages = [
                Message(role=m.role, content=m.content) for m in p["messages"]
            ]
            traces.append(Trace(messages=messages, scenario=scenario))

        return traces

