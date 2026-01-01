"""Type-safe LLM wrapper using LiteLLM."""

from typing import TypeVar, Type, overload

import litellm
from litellm import acompletion, supports_response_schema
from pydantic import BaseModel

# Configure litellm
litellm.suppress_debug_info = True
litellm.enable_json_schema_validation=True

from synkro.models import OpenAI, Model, get_model_string


T = TypeVar("T", bound=BaseModel)


class LLM:
    """
    Type-safe LLM wrapper using LiteLLM for universal provider support.

    Supports structured outputs via native JSON mode for reliable responses.

    Supported providers: OpenAI, Anthropic, Google (Gemini)

    Examples:
        >>> from synkro import LLM, OpenAI, Anthropic, Google

        # Use OpenAI
        >>> llm = LLM(model=OpenAI.GPT_4O_MINI)
        >>> response = await llm.generate("Hello!")

        # Use Anthropic
        >>> llm = LLM(model=Anthropic.CLAUDE_35_SONNET)

        # Use Google Gemini
        >>> llm = LLM(model=Google.GEMINI_25_FLASH)

        # Structured output
        >>> class Output(BaseModel):
        ...     answer: str
        ...     confidence: float
        >>> result = await llm.generate_structured("What is 2+2?", Output)
        >>> result.answer
        '4'
    """

    def __init__(
        self,
        model: Model = OpenAI.GPT_4O_MINI,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model to use (enum or string)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate (default: None = model's max)
            api_key: Optional API key override
        """
        self.model = get_model_string(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Generate a text response.

        Args:
            prompt: The user prompt
            system: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "api_key": self._api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await acompletion(**kwargs)
        return response.choices[0].message.content

    async def generate_batch(
        self, prompts: list[str], system: str | None = None
    ) -> list[str]:
        """
        Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            system: Optional system prompt for all

        Returns:
            List of generated responses
        """
        import asyncio

        tasks = [self.generate(p, system) for p in prompts]
        return await asyncio.gather(*tasks)

    @overload
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
    ) -> T: ...

    @overload
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[list[T]],
        system: str | None = None,
    ) -> list[T]: ...

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T] | Type[list[T]],
        system: str | None = None,
    ) -> T | list[T]:
        """
        Generate a structured response matching a Pydantic model.

        Uses LiteLLM's native JSON mode with response_format for
        reliable structured outputs.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class for the response
            system: Optional system prompt

        Returns:
            Parsed response matching the model

        Example:
            >>> class Analysis(BaseModel):
            ...     sentiment: str
            ...     score: float
            >>> result = await llm.generate_structured(
            ...     "Analyze: I love this product!",
            ...     Analysis
            ... )
            >>> result.sentiment
            'positive'
        """
        # Check if model supports structured outputs
        if not supports_response_schema(model=self.model, custom_llm_provider=None):
            raise ValueError(
                f"Model '{self.model}' does not support structured outputs (response_format). "
                f"Use a model that supports JSON schema like GPT-4o, Gemini 1.5+, or Claude 3.5+."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use LiteLLM's native response_format with Pydantic model
        kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": response_model,
            "temperature": self.temperature,
            "api_key": self._api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await acompletion(**kwargs)
        return response_model.model_validate_json(response.choices[0].message.content)

    async def generate_chat(
        self, messages: list[dict], response_model: Type[T] | None = None
    ) -> str | T:
        """
        Generate a response for a full conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Optional Pydantic model for structured output

        Returns:
            Generated response (string or structured)
        """
        if response_model:
            # Check if model supports structured outputs
            if not supports_response_schema(model=self.model, custom_llm_provider=None):
                raise ValueError(
                    f"Model '{self.model}' does not support structured outputs (response_format). "
                    f"Use a model that supports JSON schema like GPT-4o, Gemini 1.5+, or Claude 3.5+."
                )

            # Use LiteLLM's native response_format with Pydantic model
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format": response_model,
                "temperature": self.temperature,
                "api_key": self._api_key,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            response = await acompletion(**kwargs)
            return response_model.model_validate_json(response.choices[0].message.content)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "api_key": self._api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await acompletion(**kwargs)
        return response.choices[0].message.content

