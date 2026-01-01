"""Automatic worker scaling based on provider rate limits."""

# Known rate limits per provider (requests per minute)
PROVIDER_RATE_LIMITS = {
    "openai": 60,  # Tier 1 default, scales with tier
    "anthropic": 60,  # Standard limit
    "google": 60,  # Gemini API
    "gemini": 60,  # Gemini API (alternative prefix)
}

# Target 80% of rate limit to avoid hitting caps
UTILIZATION_TARGET = 0.8

# Default workers per provider (pre-computed for convenience)
DEFAULT_WORKERS = {
    "openai": 15,  # ~60 RPM / 3 calls = 20, use 15 to be safe
    "anthropic": 10,  # ~60 RPM, more conservative
    "google": 15,  # Gemini
    "gemini": 15,  # Gemini
}


def get_provider(model: str) -> str:
    """
    Extract provider name from model string.

    Args:
        model: Model string like "gpt-4o" or "ollama/llama3.1:8b"

    Returns:
        Provider name
    """
    # Check for explicit prefix
    if "/" in model:
        return model.split("/")[0]

    # Infer from model name
    if model.startswith("gpt") or model.startswith("o1"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"

    return "openai"  # Default


def auto_workers(model: str) -> int:
    """
    Determine optimal worker count based on model's provider.

    This calculates a safe default that won't hit rate limits,
    accounting for the fact that each trace needs ~3 LLM calls
    (generate, grade, maybe refine).

    Args:
        model: Model string

    Returns:
        Recommended worker count

    Example:
        >>> auto_workers("gpt-4o")
        15
        >>> auto_workers("gemini/gemini-2.5-flash")
        15
    """
    provider = get_provider(model)
    rpm = PROVIDER_RATE_LIMITS.get(provider, 60)

    # Workers = RPM * utilization / avg_calls_per_trace
    # Each trace needs ~3 calls (generate, grade, maybe refine)
    avg_calls_per_trace = 3

    workers = int((rpm * UTILIZATION_TARGET) / avg_calls_per_trace)

    # Clamp to reasonable bounds
    return max(5, min(workers, 100))


def get_default_workers(model: str) -> int:
    """
    Quick lookup for worker count.

    Uses pre-computed defaults for common providers.

    Args:
        model: Model string

    Returns:
        Default worker count for the provider
    """
    provider = get_provider(model)
    return DEFAULT_WORKERS.get(provider, 10)

