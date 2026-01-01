"""
Synkro - Generate high-quality training datasets from any document.

Modular Usage (recommended):
    >>> from synkro.pipelines import create_pipeline
    >>> from synkro.models.openai import OpenAI
    >>> from synkro.types import DatasetType
    >>>
    >>> pipeline = create_pipeline(
    ...     model=OpenAI.GPT_5_MINI,
    ...     dataset_type=DatasetType.SFT,
    ... )
    >>> dataset = pipeline.generate("policy text", traces=50)
    >>> dataset.save("training.jsonl")

Simple Usage:
    >>> import synkro
    >>> dataset = synkro.generate("Your policy text...")
    >>> dataset.save("training.jsonl")

Silent Mode (for embedding/testing):
    >>> from synkro import SilentReporter, create_pipeline
    >>> pipeline = create_pipeline(reporter=SilentReporter())
    >>> dataset = pipeline.generate("policy text")  # No console output
"""

from synkro.pipelines import create_pipeline
from synkro.models import OpenAI, Anthropic, Google
from synkro.types import DatasetType, Message, Scenario, Trace, GradeResult, Plan, Category
from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.llm.client import LLM
from synkro.generation.generator import Generator
from synkro.generation.scenarios import ScenarioGenerator
from synkro.generation.responses import ResponseGenerator
from synkro.generation.planner import Planner
from synkro.quality.grader import Grader
from synkro.quality.refiner import Refiner
from synkro.formatters.sft import SFTFormatter
from synkro.formatters.qa import QAFormatter
from synkro.prompts import SystemPrompt, ScenarioPrompt, ResponsePrompt, GradePrompt
from synkro.reporting import ProgressReporter, RichReporter, SilentReporter

__version__ = "0.3.2"

__all__ = [
    # Pipeline creation
    "create_pipeline",
    # Quick function
    "generate",
    # Dataset type enum
    "DatasetType",
    # Core classes
    "Policy",
    "Dataset",
    "Trace",
    "Scenario",
    "Message",
    "GradeResult",
    "Plan",
    "Category",
    # Generation
    "Generator",
    "ScenarioGenerator",
    "ResponseGenerator",
    "Planner",
    # Quality
    "Grader",
    "Refiner",
    # LLM
    "LLM",
    # Prompts
    "SystemPrompt",
    "ScenarioPrompt",
    "ResponsePrompt",
    "GradePrompt",
    # Formatters
    "SFTFormatter",
    "QAFormatter",
    # Reporters
    "ProgressReporter",
    "RichReporter",
    "SilentReporter",
    # Model enums (OpenAI, Anthropic, Google supported)
    "OpenAI",
    "Anthropic",
    "Google",
]


def generate(
    policy: str | Policy,
    traces: int = 20,
    dataset_type: DatasetType = DatasetType.SFT,
    generation_model: OpenAI | Anthropic | Google | str = OpenAI.GPT_5_MINI,
    grading_model: OpenAI | Anthropic | Google | str = OpenAI.GPT_52,
    max_iterations: int = 3,
    skip_grading: bool = False,
    reporter: ProgressReporter | None = None,
) -> Dataset:
    """
    Generate training traces from a policy document.

    This is a convenience function. For more control, use create_pipeline().

    Args:
        policy: Policy text or Policy object
        traces: Number of traces to generate (default: 20)
        dataset_type: Type of dataset - SFT (default) or QA
        generation_model: Model for generating (default: gpt-5-mini)
        grading_model: Model for grading (default: gpt-5.2)
        max_iterations: Max refinement iterations per trace (default: 3)
        skip_grading: Skip grading phase for faster generation (default: False)
        reporter: Progress reporter (default: RichReporter for console output)

    Returns:
        Dataset object with generated traces

    Example:
        >>> import synkro
        >>> dataset = synkro.generate("All expenses over $50 require approval")
        >>> dataset.save("training.jsonl")
        
        >>> # Silent mode
        >>> from synkro import SilentReporter
        >>> dataset = synkro.generate(policy, reporter=SilentReporter())
    """
    if isinstance(policy, str):
        policy = Policy(text=policy)

    generator = Generator(
        dataset_type=dataset_type,
        generation_model=generation_model,
        grading_model=grading_model,
        max_iterations=max_iterations,
        skip_grading=skip_grading,
        reporter=reporter,
    )

    return generator.generate(policy, traces=traces)
