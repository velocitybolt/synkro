"""OpenAI models."""

from enum import Enum


class OpenAI(str, Enum):
    """OpenAI models."""

    # GPT-5 series (latest)
    GPT_52 = "gpt-5.2"
    """Flagship: High-speed, human-like dialogue, agentic tool-calling"""

    GPT_5_MINI = "gpt-5-mini"
    """Mid-tier: Balanced cost and intelligence, primary workhorse"""

    GPT_5_NANO = "gpt-5-nano"
    """Edge: Extremely low latency, high-volume basic tasks"""

    # GPT-4 series (legacy)
    GPT_41 = "gpt-4.1"
    """Legacy flagship: Smartest non-reasoning model from previous gen"""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Reasoning models
    O3 = "o3"
    O3_MINI = "o3-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"

