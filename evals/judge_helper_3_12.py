"""Local provider and judge helpers for experiment 3.12."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from synthesis.config import settings
from synthesis.engine.model_backend import AnthropicBackend, LLMResult

OPENAI_MODEL = "gpt-5-nano"
ANTHROPIC_JUDGE_MODEL = "claude-sonnet-4-20250514"

CONFIDENCE_LABELS = ("HIGH", "MEDIUM", "LOW", "MADE_UP")
ENTAILMENT_LABELS = ("entailed", "neutral", "contradicted")


class ClaimAssessment(BaseModel):
    field_path: str
    claim: str
    confidence: Literal["HIGH", "MEDIUM", "LOW", "MADE_UP"]
    rationale: str = ""


class SelfCritiqueReport(BaseModel):
    cluster_id: str
    persona_name: str
    assessments: list[ClaimAssessment] = Field(min_length=1)


class EntailmentJudgment(BaseModel):
    field_path: str
    claim: str
    label: Literal["entailed", "neutral", "contradicted"]
    rationale: str = ""


class EntailmentReport(BaseModel):
    cluster_id: str
    persona_name: str
    judgments: list[EntailmentJudgment] = Field(min_length=1)


class OpenAIJsonBackend:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float | None = None,
        max_completion_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def with_temperature(self, temperature: float | None):
        return OpenAIJsonBackend(
            client=self.client,
            model=self.model,
            temperature=temperature,
            max_completion_tokens=self.max_completion_tokens,
        )

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        tool_schema = json.dumps(tool["input_schema"], indent=2, default=str)
        prompt_messages = [{"role": "system", "content": system}, *messages]
        prompt_messages.append(
            {
                "role": "user",
                "content": (
                    "Return a single JSON object that conforms to the schema below.\n\n"
                    f"SCHEMA:\n{tool_schema}"
                ),
            }
        )
        payload = {
            "model": self.model,
            "messages": prompt_messages,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = await self.client.chat.completions.create(**payload)
        choice = response.choices[0].message
        text = choice.content or "{}"
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text)
        tool_input = json.loads(text)
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
        )


class FallbackGenerateBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.temperature = getattr(primary, "temperature", None)
        self.model = getattr(primary, "model", "")
        self.client = getattr(primary, "client", None)

    def with_temperature(self, temperature: float | None):
        if hasattr(self.primary, "with_temperature"):
            primary = self.primary.with_temperature(temperature)
        else:
            try:
                primary = self.primary.__class__(
                    client=self.primary.client,
                    model=self.primary.model,
                    temperature=temperature,
                    top_p=getattr(self.primary, "top_p", None),
                )
            except TypeError:
                primary = self.primary.__class__(
                    client=self.primary.client,
                    model=self.primary.model,
                )
        fallback = self.fallback.with_temperature(temperature) if self.fallback is not None else None
        return FallbackGenerateBackend(primary=primary, fallback=fallback)

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        try:
            return await self.primary.generate(system=system, messages=messages, tool=tool)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.generate(system=system, messages=messages, tool=tool)


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_generation_backend(
    *,
    anthropic_model: str,
    temperature: float | None = None,
    openai_model: str = OPENAI_MODEL,
    prefer_anthropic: bool = True,
) -> tuple[object, str, str]:
    """Build a generation backend with Anthropic primary and OpenAI fallback."""
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if prefer_anthropic and settings.anthropic_api_key:
        primary = AnthropicBackend(
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=anthropic_model,
        )
        fallback = (
            OpenAIJsonBackend(
                client=openai_client,
                model=openai_model,
                temperature=None,
                max_completion_tokens=4096,
            )
            if openai_client is not None
            else None
        )
        provider = "anthropic->openai" if fallback is not None else "anthropic"
        return FallbackGenerateBackend(primary=primary, fallback=fallback), provider, anthropic_model

    if not prefer_anthropic and openai_client is not None:
        return (
            OpenAIJsonBackend(
                client=openai_client,
                model=openai_model,
                temperature=None,
                max_completion_tokens=4096,
            ),
            "openai",
            openai_model,
        )

    if settings.anthropic_api_key:
        primary = AnthropicBackend(
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=anthropic_model,
        )
        return primary, "anthropic", anthropic_model

    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")

    return (
        OpenAIJsonBackend(
            client=openai_client,
            model=openai_model,
            temperature=None,
            max_completion_tokens=4096,
        ),
        "openai",
        openai_model,
    )
