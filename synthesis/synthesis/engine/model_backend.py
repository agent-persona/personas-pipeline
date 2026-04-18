from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    AsyncOpenAI = None  # type: ignore[assignment]


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
        if "gpt-5-nano" in self.model:
            return (self.input_tokens * 0.05 + self.output_tokens * 0.4) / 1_000_000
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


class ModelBackend(Protocol):
    """Protocol for LLM backends that produce persona JSON via tool use."""

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult: ...


class AnthropicBackend:
    """Anthropic Claude backend using tool-use forcing."""

    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )

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


class OpenAIJsonBackend:
    """OpenAI Responses backend that returns validated persona JSON."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult:
        if AsyncOpenAI is None:  # pragma: no cover - import guard
            raise RuntimeError("openai package not installed")

        response = await self.client.responses.create(
            model=self.model,
            instructions=(
                f"{system}\n\n"
                "Return only valid JSON. No markdown. No prose. "
                "The JSON must match the supplied response schema."
            ),
            input=messages,
            max_output_tokens=8000,
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": tool["name"],
                    "description": tool["description"],
                    "schema": tool["input_schema"],
                    "strict": False,
                },
            },
        )

        text = response.output_text or "{}"
        tool_input = _extract_first_json_object(text)
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model,
        )


def _extract_first_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    parsed, _ = decoder.raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Top-level JSON value is not an object", text, start)
    return parsed
