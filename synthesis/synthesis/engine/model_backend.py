from __future__ import annotations

import json
import re
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
        use_cache: bool = False,
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
        use_cache: bool = False,
    ) -> LLMResult:
        # When use_cache=True the static system content is embedded in the
        # first user content block (with cache_control markers), so we skip
        # the system param to avoid duplication.
        effective_system = "" if use_cache else system

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=effective_system,
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


class OpenAIBackend:
    """OpenAI-compatible backend (Ollama, vLLM, etc.) using JSON prompting.

    Falls back to asking the model to return JSON directly since many local
    models don't support forced tool calling.
    """

    def __init__(self, client, model: str) -> None:
        self.client = client  # openai.AsyncOpenAI
        self.model = model

    async def generate(
        self,
        system: str,
        messages: list[dict],
        tool: dict,
    ) -> LLMResult:
        schema = tool.get("input_schema", {})
        schema_str = json.dumps(schema, indent=2)

        json_system = (
            f"{system}\n\n"
            "CRITICAL: You MUST respond with ONLY a valid JSON object matching this schema. "
            "No markdown fences, no explanation, no extra text.\n\n"
            f"JSON SCHEMA:\n{schema_str}"
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": json_system},
                *messages,
            ],
        )

        content = response.choices[0].message.content or ""
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        # Try direct parse first
        try:
            tool_input = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract the largest valid JSON object
            # Find the outermost { ... } and attempt repair
            start = content.find("{")
            if start == -1:
                raise ValueError(f"No JSON object found in response: {content[:200]}")

            # Try progressively shorter substrings to find valid JSON
            best = None
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            best = json.loads(content[start : i + 1])
                            break
                        except json.JSONDecodeError:
                            continue

            if best is None:
                # Last resort: truncate at last complete key-value and close
                truncated = content[start:]
                # Find last complete string value
                last_quote = truncated.rfind('"')
                if last_quote > 0:
                    # Walk back to find a clean cut point
                    for cut in range(last_quote, 0, -1):
                        candidate = truncated[:cut].rstrip(", \n\t")
                        # Close any open arrays/objects
                        opens = candidate.count("[") - candidate.count("]")
                        candidate += "]" * max(0, opens)
                        opens = candidate.count("{") - candidate.count("}")
                        candidate += "}" * max(0, opens)
                        try:
                            best = json.loads(candidate)
                            break
                        except json.JSONDecodeError:
                            continue

            if best is None:
                raise ValueError(f"Could not parse JSON from response: {content[:500]}")
            tool_input = best

        return LLMResult(
            tool_input=tool_input,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=self.model,
        )
