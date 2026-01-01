"""LLM client wrapper for multiple providers via LiteLLM."""

from synkro.llm.client import LLM
from synkro.llm.rate_limits import auto_workers, get_provider

__all__ = ["LLM", "auto_workers", "get_provider"]

