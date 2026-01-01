"""Main Generator class orchestrating the full trace generation pipeline."""

import asyncio
from enum import Enum

from synkro.llm.client import LLM
from synkro.llm.rate_limits import auto_workers
from synkro.models import Model, OpenAI
from synkro.types.dataset_type import DatasetType
from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.modes.config import get_mode_config
from synkro.errors import handle_error
from synkro.factory import ComponentFactory
from synkro.reporting import ProgressReporter, RichReporter
from synkro.pipeline.runner import GenerationPipeline


class Generator:
    """
    Main orchestrator for generating training datasets.

    The Generator handles the full pipeline:
    1. Plan: Analyze policy and create category distribution
    2. Generate: Create scenarios and responses
    3. Grade: Evaluate response quality
    4. Refine: Fix failed responses
    5. Return: Dataset of passing traces

    Examples:
        >>> generator = Generator()
        >>> dataset = generator.generate(policy, traces=20)

        >>> # QA dataset
        >>> generator = Generator(dataset_type=DatasetType.QA)
        >>> dataset = generator.generate(policy)
        
        >>> # Silent mode (no console output)
        >>> from synkro.reporting import SilentReporter
        >>> generator = Generator(reporter=SilentReporter())
        >>> dataset = generator.generate(policy)
    """

    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.SFT,
        generation_model: Model = OpenAI.GPT_4O_MINI,
        grading_model: Model = OpenAI.GPT_4O,
        max_iterations: int = 1,
        skip_grading: bool = False,
        reporter: ProgressReporter | None = None,
    ):
        """
        Initialize the Generator.

        Args:
            dataset_type: Type of dataset to generate (QA or SFT)
            generation_model: Model for scenarios/responses (default: gpt-4o-mini)
            grading_model: Model for grading (default: gpt-4o, recommend stronger)
            max_iterations: Max refinement iterations per trace (default: 1, no retries)
            skip_grading: Skip grading phase for faster generation (default: False)
            reporter: Progress reporter (default: RichReporter for console output)
        """
        self.dataset_type = dataset_type
        self.mode_config = get_mode_config(dataset_type)
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        
        # Store model info for reporting
        self.generation_model = generation_model
        self.grading_model = grading_model
        
        # Create LLM clients
        self.generation_llm = LLM(model=generation_model)
        self.grading_llm = LLM(model=grading_model)
        
        # Create factory for component creation
        self.factory = ComponentFactory(
            generation_llm=self.generation_llm,
            grading_llm=self.grading_llm,
            mode_config=self.mode_config,
        )
        
        # Reporter for progress output
        self.reporter = reporter or RichReporter()
        
        # Auto-scale workers based on provider
        model_str = generation_model.value if isinstance(generation_model, Enum) else str(generation_model)
        self.workers = auto_workers(model_str)
        
        # Create pipeline
        self.pipeline = GenerationPipeline(
            factory=self.factory,
            reporter=self.reporter,
            workers=self.workers,
            max_iterations=max_iterations,
            skip_grading=skip_grading,
        )

    @handle_error
    def generate(self, policy: Policy | str, traces: int = 20) -> Dataset:
        """
        Generate a training dataset from a policy.

        Args:
            policy: Policy object or text string
            traces: Target number of traces to generate (default: 20)

        Returns:
            Dataset with generated traces
        """
        if isinstance(policy, str):
            policy = Policy(text=policy)

        # Validate policy has enough content
        policy.validate_length()

        return asyncio.run(self._generate_async(policy, traces))

    async def _generate_async(self, policy: Policy, traces: int) -> Dataset:
        """Async implementation of generation pipeline."""
        model_str = self.generation_model.value if isinstance(self.generation_model, Enum) else str(self.generation_model)
        
        return await self.pipeline.run(
            policy=policy,
            traces=traces,
            model=model_str,
            dataset_type=self.dataset_type.value,
        )

    async def generate_async(self, policy: Policy | str, traces: int = 20) -> Dataset:
        """
        Async version of generate for use in async contexts.

        Args:
            policy: Policy object or text string
            traces: Target number of traces to generate (default: 20)

        Returns:
            Dataset with generated traces
        """
        if isinstance(policy, str):
            policy = Policy(text=policy)

        return await self._generate_async(policy, traces)
