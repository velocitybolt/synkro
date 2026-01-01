"""Friendly error messages with fix suggestions."""

from rich.console import Console
from rich.panel import Panel

console = Console()


class SynkroError(Exception):
    """Base exception for Synkro errors."""
    
    def __init__(self, message: str, suggestion: str = ""):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)
    
    def print_friendly(self):
        """Print a user-friendly error message."""
        content = f"[red bold]{self.message}[/red bold]"
        if self.suggestion:
            content += f"\n\n[dim]{self.suggestion}[/dim]"
        console.print(Panel(content, title="[red]Error[/red]", border_style="red"))


class APIKeyError(SynkroError):
    """Raised when API key is missing or invalid."""
    
    def __init__(self, provider: str = "OpenAI"):
        provider_info = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys",
            },
            "anthropic": {
                "env_var": "ANTHROPIC_API_KEY",
                "url": "https://console.anthropic.com/settings/keys",
            },
            "google": {
                "env_var": "GEMINI_API_KEY",
                "url": "https://aistudio.google.com/app/apikey",
            },
        }
        
        info = provider_info.get(provider.lower(), provider_info["openai"])
        
        message = f"{provider} API key not found or invalid"
        suggestion = f"""To fix this, either:

  1. Set environment variable:
     export {info['env_var']}="your-key-here"

  2. Pass directly:
     synkro.generate(..., generation_model=LLM(api_key="..."))

Get an API key at: {info['url']}"""
        
        super().__init__(message, suggestion)


class FileNotFoundError(SynkroError):
    """Raised when a policy file cannot be found."""
    
    def __init__(self, filepath: str, similar_files: list[str] | None = None):
        message = f"Could not find file: {filepath}"
        
        suggestion_parts = []
        if similar_files:
            suggestion_parts.append("Did you mean one of these?")
            for f in similar_files[:3]:
                suggestion_parts.append(f"  → {f}")
            suggestion_parts.append("")
        
        suggestion_parts.append("Or pass text directly:")
        suggestion_parts.append('  synkro.generate("Your policy text here...")')
        
        super().__init__(message, "\n".join(suggestion_parts))


class RateLimitError(SynkroError):
    """Raised when hitting API rate limits."""
    
    def __init__(self, provider: str = "OpenAI", retry_after: int | None = None):
        message = f"Rate limited by {provider}"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        
        suggestion = """Tip: Reduce the number of traces or wait and retry:

  synkro.generate(..., traces=10)

Or use a different provider:
  synkro.generate(..., generation_model=Google.GEMINI_25_FLASH)"""
        
        super().__init__(message, suggestion)


class PolicyTooShortError(SynkroError):
    """Raised when policy text is too short to generate meaningful data."""
    
    def __init__(self, word_count: int):
        message = f"Policy too short ({word_count} words)"
        suggestion = """Your policy needs more content to generate diverse training data.

Minimum recommended: 50+ words with clear rules or guidelines.

Example:
  synkro.generate('''
  All expenses over $50 require manager approval.
  Expenses over $500 require VP approval.
  Receipts are required for all purchases over $25.
  ''')"""
        
        super().__init__(message, suggestion)


class ModelNotFoundError(SynkroError):
    """Raised when specified model doesn't exist."""
    
    def __init__(self, model: str):
        message = f"Model not found: {model}"
        suggestion = """Available models:

  OpenAI:    OpenAI.GPT_4O_MINI, OpenAI.GPT_4O
  Anthropic: Anthropic.CLAUDE_35_SONNET, Anthropic.CLAUDE_35_HAIKU
  Google:    Google.GEMINI_25_FLASH, Google.GEMINI_25_PRO

Or pass any model string: "gpt-4o-mini", "claude-3-5-sonnet-20241022", etc."""
        
        super().__init__(message, suggestion)


def _detect_provider(error_str: str) -> str:
    """Detect the provider from the error message."""
    error_lower = error_str.lower()
    if "gemini" in error_lower or "googleapis" in error_lower or "google" in error_lower:
        return "Google"
    if "anthropic" in error_lower or "claude" in error_lower:
        return "Anthropic"
    return "OpenAI"


def handle_error(func):
    """Decorator to catch and display friendly errors."""
    import sys
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SynkroError as e:
            e.print_friendly()
            sys.exit(1)
        except Exception as e:
            # Try to convert common errors to friendly ones
            error_str = str(e)
            error_lower = error_str.lower()
            
            if "api key" in error_lower or "authentication" in error_lower or "unauthorized" in error_lower:
                provider = _detect_provider(error_str)
                friendly = APIKeyError(provider)
                friendly.print_friendly()
                sys.exit(1)
            
            if "rate limit" in error_lower or "429" in error_lower:
                provider = _detect_provider(error_str)
                friendly = RateLimitError(provider)
                friendly.print_friendly()
                sys.exit(1)
            
            if "not found" in error_lower and "model" in error_lower:
                friendly = ModelNotFoundError(str(e))
                friendly.print_friendly()
                sys.exit(1)
            
            # Re-raise unknown errors
            raise
    
    return wrapper

