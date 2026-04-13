"""Local judge helper for experiment 1.22."""

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


class AnthropicJudgeBackend:
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
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or ""


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


def _clean_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _weighted_ratio(persona: dict, specs: list[tuple[str, float, int]]) -> float:
    total_weight = sum(weight for _, weight, _ in specs)
    if total_weight <= 0:
        return 0.0
    total = 0.0
    for field, weight, cap in specs:
        values = _clean_list(persona.get(field, []))
        total += weight * min(len(values) / max(cap, 1), 1.0)
    return total / total_weight


def _score_to_json(persona: dict) -> str:
    grounded_ratio = _weighted_ratio(
        persona,
        [
            ("goals", 1.7, 4),
            ("pains", 1.3, 4),
            ("motivations", 1.0, 3),
            ("objections", 0.8, 3),
            ("sample_quotes", 0.8, 4),
        ],
    )
    distinctive_ratio = _weighted_ratio(
        persona,
        [
            ("sample_quotes", 1.8, 4),
            ("vocabulary", 1.2, 6),
            ("goals", 0.6, 4),
            ("journey_stages", 0.4, 2),
        ],
    )
    coherent_ratio = _weighted_ratio(
        persona,
        [
            ("goals", 1.2, 4),
            ("pains", 1.0, 4),
            ("motivations", 1.0, 3),
            ("objections", 1.0, 3),
            ("journey_stages", 0.8, 2),
            ("decision_triggers", 0.8, 3),
        ],
    )
    actionable_ratio = _weighted_ratio(
        persona,
        [
            ("goals", 1.7, 4),
            ("pains", 1.4, 4),
            ("decision_triggers", 1.3, 3),
            ("sample_quotes", 0.6, 4),
        ],
    )
    voice_ratio = _weighted_ratio(
        persona,
        [
            ("sample_quotes", 1.8, 4),
            ("vocabulary", 1.4, 6),
            ("channels", 0.4, 4),
            ("journey_stages", 0.4, 2),
        ],
    )

    dimensions = {
        "grounded": round(1 + 4 * grounded_ratio, 2),
        "distinctive": round(1 + 4 * distinctive_ratio, 2),
        "coherent": round(1 + 4 * coherent_ratio, 2),
        "actionable": round(1 + 4 * actionable_ratio, 2),
        "voice_fidelity": round(1 + 4 * voice_ratio, 2),
    }
    overall = round(sum(dimensions.values()) / len(dimensions), 2)
    rationale = (
        "heuristic judge: "
        f"grounded={dimensions['grounded']:.2f}, distinctive={dimensions['distinctive']:.2f}, "
        f"coherent={dimensions['coherent']:.2f}, actionable={dimensions['actionable']:.2f}, "
        f"voice_fidelity={dimensions['voice_fidelity']:.2f}"
    )
    return json.dumps({**dimensions, "overall": overall, "rationale": rationale})


class HeuristicJudgeBackend:
    async def score(self, system: str, prompt: str) -> str:
        match = re.search(r"PERSONA:\n(\{.*\})\s*$", prompt, re.DOTALL)
        if match:
            try:
                persona = json.loads(match.group(1))
                return _score_to_json(persona)
            except json.JSONDecodeError:
                pass
        return json.dumps(
            {
                "grounded": 1.0,
                "distinctive": 1.0,
                "coherent": 1.0,
                "actionable": 1.0,
                "voice_fidelity": 1.0,
                "overall": 1.0,
                "rationale": "heuristic judge fallback",
            }
        )


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


class LLMJudge:
    def __init__(self, backend, model: str) -> None:
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


def build_judge(provider: str, anthropic_key: str, openai_key: str):
    if provider == "heuristic":
        return LLMJudge(HeuristicJudgeBackend(), "heuristic"), "heuristic"
    if provider == "anthropic":
        judge_model = "claude-sonnet-4-20250514"
        model_label = (
            f"{judge_model} (+fallback gpt-5-nano)"
            if openai_key
            else judge_model
        )
        primary = AnthropicJudgeBackend(
            client=AsyncAnthropic(api_key=anthropic_key),
            model=judge_model,
        )
        fallback = (
            OpenAIJudgeBackend(client=AsyncOpenAI(api_key=openai_key), model="gpt-5-nano")
            if openai_key
            else None
        )
        return LLMJudge(FallbackJudgeBackend(primary=primary, fallback=fallback), model_label), model_label
    if provider == "openai":
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return (
            LLMJudge(OpenAIJudgeBackend(client=AsyncOpenAI(api_key=openai_key), model="gpt-5-nano"), "gpt-5-nano"),
            "gpt-5-nano",
        )
    raise RuntimeError(f"unknown provider: {provider}")
