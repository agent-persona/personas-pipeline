from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from synthesis.config import settings
from synthesis.provider_registry import (
    get_provider_spec,
    normalize_provider,
    validate_provider_model,
)


@dataclass
class LLMResult:
    """Raw result from an LLM call."""

    tool_input: dict
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on Claude pricing."""
        # Haiku 4.5: $1/M input, $5/M output
        # Sonnet 4.6: $3/M input, $15/M output
        # Opus 4.6: $15/M input, $75/M output
        if "opus" in self.model:
            return (self.input_tokens * 15 + self.output_tokens * 75) / 1_000_000
        if "haiku" in self.model:
            return (self.input_tokens * 1 + self.output_tokens * 5) / 1_000_000
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


class ModelBackend(Protocol):
    """Protocol for LLM backends that produce persona JSON via tool use."""

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult: ...


def _gemini_api_key() -> str:
    selector = settings.gemini_api_key_selector.strip().lower()
    if selector == "turkey":
        return settings.gemini_api_key_turkey
    return settings.gemini_api_key_max


def provider_supports_anthropic_sdk(provider: str | None = None) -> bool:
    return get_provider_spec(provider or settings.model_provider).transport == "anthropic"


def build_anthropic_compatible_client_from_settings(
    provider: str | None = None,
) -> AsyncAnthropic:
    normalized = normalize_provider(provider or settings.model_provider)
    if normalized == "anthropic":
        api_key = settings.anthropic_api_key
        base_url = settings.anthropic_base_url or None
    elif normalized == "zai":
        api_key = settings.z_ai_glm_api_key
        base_url = settings.z_ai_anthropic_base_url
    elif normalized == "minimax":
        api_key = settings.minimax_api_key
        base_url = settings.minimax_anthropic_base_url
    else:
        raise ValueError(
            f"Provider '{normalized}' is not Anthropic-compatible for this runtime",
        )

    if not api_key:
        raise RuntimeError(f"API key missing for provider '{normalized}'")
    if normalized == "anthropic" and not base_url:
        os.environ.pop("ANTHROPIC_BASE_URL", None)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)


class AnthropicBackend:
    """Anthropic Claude backend using tool-use forcing."""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k

        response = await self.client.messages.create(**kwargs)

        # Extract the tool use block
        tool_block = next(
            block for block in response.content if block.type == "tool_use"
        )

        return LLMResult(
            tool_input=tool_block.input,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )


class OpenAICompatibleBackend:
    """OpenAI-compatible backend for providers exposing chat completions + tools."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult:
        oai_messages = [{"role": "system", "content": system}, *messages]
        oai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        kwargs: dict = dict(
            model=self.model,
            messages=oai_messages,
            tools=[oai_tool],
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
            max_tokens=4096,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}

        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            arguments = message.tool_calls[0].function.arguments
            tool_input = json.loads(arguments)
        elif message.content:
            tool_input = json.loads(message.content)
        else:
            raise ValueError("Model returned neither tool calls nor JSON content")

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return LLMResult(
            tool_input=tool_input,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )


def build_backend_from_settings(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> ModelBackend:
    provider = normalize_provider(settings.model_provider)
    selected_model = model or settings.default_model
    validate_provider_model(provider, selected_model, label="Synthesis")

    if provider_supports_anthropic_sdk(provider):
        client = build_anthropic_compatible_client_from_settings(provider)
        return AnthropicBackend(
            client=client,
            model=selected_model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    if provider == "gemini":
        api_key = _gemini_api_key()
        base_url = settings.gemini_base_url
    elif provider == "kimi":
        api_key = settings.kimi_api_key
        base_url = settings.kimi_base_url
    else:
        raise ValueError(f"Unsupported model provider '{provider}'")

    if not api_key:
        raise RuntimeError(f"API key missing for provider '{provider}'")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAICompatibleBackend(
        client=client,
        model=selected_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
