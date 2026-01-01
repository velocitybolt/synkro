"""Anthropic Claude models."""

from enum import Enum


class Anthropic(str, Enum):
    """Anthropic Claude models."""

    # Claude 4.5 (latest)
    CLAUDE_45_OPUS = "claude-opus-4-5-20250601"
    """Premium: State-of-the-art for coding and autonomous agents"""

    CLAUDE_45_SONNET = "claude-sonnet-4-5-20250601"
    """Standard: Default model for most users, faster and more context-aware"""

    CLAUDE_45_HAIKU = "claude-haiku-4-5-20250601"
    """Light: High-speed, cost-effective, matches Claude 3 Opus intelligence"""

    # Claude 4 (previous gen)
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"

    # Claude 3.5 (legacy)
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"

