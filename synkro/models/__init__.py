"""Model enums for supported LLM providers.

Supported providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet/Haiku)
- Google (Gemini 2.5 Flash/Pro)

Usage:
    # Per-provider import (recommended)
    from synkro.models.openai import OpenAI
    from synkro.models.anthropic import Anthropic
    from synkro.models.google import Google

    # Convenience import (all at once)
    from synkro.models import OpenAI, Anthropic, Google
"""

from enum import Enum
from typing import Union

from synkro.models.openai import OpenAI
from synkro.models.anthropic import Anthropic
from synkro.models.google import Google

# Union type for any model
Model = Union[OpenAI, Anthropic, Google, str]


def get_model_string(model: Model) -> str:
    """Convert a model enum or string to its string value."""
    if isinstance(model, Enum):
        return model.value
    return model


__all__ = [
    "OpenAI",
    "Anthropic",
    "Google",
    "Model",
    "get_model_string",
]

