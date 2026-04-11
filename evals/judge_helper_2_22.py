"""Local judge helper for experiment 2.22."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

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
