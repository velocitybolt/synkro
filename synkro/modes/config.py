"""Mode configuration that bundles prompts, schema, and formatter per dataset type."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from synkro.types.dataset_type import DatasetType


@dataclass
class ModeConfig:
    """
    Configuration bundle for a dataset type.

    Defines all the prompts, schemas, and formatters needed
    for generating a specific type of dataset.
    """

    # Prompts
    scenario_prompt: str
    """Prompt for generating scenarios/questions"""

    response_prompt: str
    """Prompt for generating responses/answers"""

    grade_prompt: str
    """Prompt for grading quality"""

    refine_prompt: str
    """Prompt for refining failed responses"""

    # Output configuration
    output_description: str
    """Human-readable description of output format"""


def get_mode_config(dataset_type: "DatasetType") -> ModeConfig:
    """
    Get the mode configuration for a dataset type.

    Args:
        dataset_type: The type of dataset to generate

    Returns:
        ModeConfig with appropriate prompts and settings

    Example:
        >>> from synkro import DatasetType
        >>> config = get_mode_config(DatasetType.QA)
    """
    from synkro.types.dataset_type import DatasetType
    from synkro.modes.qa import QA_CONFIG
    from synkro.modes.sft import SFT_CONFIG

    configs = {
        DatasetType.QA: QA_CONFIG,
        DatasetType.SFT: SFT_CONFIG,
    }

    if dataset_type not in configs:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return configs[dataset_type]

