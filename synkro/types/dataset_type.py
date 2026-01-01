"""Dataset type enum for steering generation pipeline."""

from enum import Enum


class DatasetType(str, Enum):
    """
    Type of dataset to generate.

    The dataset type determines:
    - Prompts used for scenario and response generation
    - Grading criteria
    - Output format and schema

    Examples:
        >>> from synkro import DatasetType
        >>> synkro.generate(policy, dataset_type=DatasetType.QA)
        >>> synkro.generate(policy, dataset_type=DatasetType.SFT)
    """

    QA = "qa"
    """Question-Answer pairs: {question, answer, context}"""

    SFT = "sft"
    """Supervised Fine-Tuning: {messages: [system, user, assistant]}"""

