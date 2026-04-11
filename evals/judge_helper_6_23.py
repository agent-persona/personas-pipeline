"""Local judge helper for experiment 6.23."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

DEFAULT_DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)

_JUDGE_SYSTEM_PROMPT = """You are an expert persona evaluator.

Score a customer persona on a 1-5 scale for:
- grounded
- distinctive
- coherent
- actionable
- voice_fidelity

Return JSON only:
{
  "grounded": <1-5>,
  "distinctive": <1-5>,
  "coherent": <1-5>,
  "actionable": <1-5>,
  "voice_fidelity": <1-5>,
  "overall": <1-5>,
  "rationale": "short justification"
}
"""


@dataclass
class JudgeScore:
    overall: float
    dimensions: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    judge_model: str = ""


class JudgeBackend:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIJudgeBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            max_output_tokens=1024,
            reasoning={"effort": "minimal"},
        )
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI judge returned empty output")
        return text


class FallbackJudgeBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def score(self, system: str, prompt: str) -> str:
        try:
            return await self.primary.score(system=system, prompt=prompt)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.score(system=system, prompt=prompt)


class LLMJudge:
    def __init__(self, backend: JudgeBackend, model: str) -> None:
        self.backend = backend
        self.model = model

    async def score_persona(self, persona: dict) -> JudgeScore:
        prompt = "Score this persona.\n\nPERSONA:\n" + json.dumps(
            persona,
            indent=2,
            default=str,
        )
        response = await self.backend.score(_JUDGE_SYSTEM_PROMPT, prompt)
        return _parse_response(response, self.model)


def _parse_response(text: str, model: str) -> JudgeScore:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return JudgeScore(
                overall=float("nan"),
                dimensions={dimension: float("nan") for dimension in DEFAULT_DIMENSIONS},
                rationale=cleaned[:240],
                judge_model=model,
            )
        data = json.loads(match.group())
    return JudgeScore(
        overall=float(data.get("overall", float("nan"))),
        dimensions={
            dimension: float(data.get(dimension, float("nan")))
            for dimension in DEFAULT_DIMENSIONS
        },
        rationale=str(data.get("rationale", "")),
        judge_model=model,
    )


def build_judge() -> tuple[LLMJudge, str]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = "claude-sonnet-4-20250514"
    if anthropic_key:
        primary = JudgeBackend(client=AsyncAnthropic(api_key=anthropic_key), model=model)
        fallback = OpenAIJudgeBackend(client=AsyncOpenAI(api_key=openai_key), model="gpt-5-nano") if openai_key else None
        return LLMJudge(backend=FallbackJudgeBackend(primary=primary, fallback=fallback), model=model), model
    if openai_key:
        return LLMJudge(backend=OpenAIJudgeBackend(client=AsyncOpenAI(api_key=openai_key), model="gpt-5-nano"), model="gpt-5-nano"), "gpt-5-nano"
    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")
