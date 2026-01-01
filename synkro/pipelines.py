"""Pipeline creation utilities.

Usage:
    from synkro.pipelines import create_pipeline
    from synkro.models.openai import OpenAI
    from synkro.types import DatasetType

    pipeline = create_pipeline(
        model=OpenAI.GPT_5_MINI,
        dataset_type=DatasetType.SFT,
    )
    dataset = pipeline.generate("policy text", traces=50)
"""

from synkro.generation.generator import Generator
from synkro.types import DatasetType
from synkro.models import Model, OpenAI
from synkro.reporting import ProgressReporter


def create_pipeline(
    model: Model = OpenAI.GPT_5_MINI,
    dataset_type: DatasetType = DatasetType.SFT,
    grading_model: Model = OpenAI.GPT_52,
    max_iterations: int = 3,
    skip_grading: bool = False,
    reporter: ProgressReporter | None = None,
) -> Generator:
    """
    Create a pipeline for generating training datasets.

    Args:
        model: Model enum for generation (default: OpenAI.GPT_5_MINI)
        dataset_type: Type of dataset - DatasetType.QA or SFT (default: SFT)
        grading_model: Model enum for grading (default: OpenAI.GPT_52)
        max_iterations: Max refinement iterations per trace (default: 3)
        skip_grading: Skip grading phase for faster generation (default: False)
        reporter: Progress reporter (default: RichReporter for console output)

    Returns:
        Generator instance ready to use

    Example:
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
        
        >>> # Silent mode for embedding
        >>> from synkro.reporting import SilentReporter
        >>> pipeline = create_pipeline(reporter=SilentReporter())
    """
    return Generator(
        dataset_type=dataset_type,
        generation_model=model,
        grading_model=grading_model,
        max_iterations=max_iterations,
        skip_grading=skip_grading,
        reporter=reporter,
    )


__all__ = ["create_pipeline"]
