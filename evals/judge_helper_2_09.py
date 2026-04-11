"""Local judge helper for experiment 2.09 — best-of-N selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from anthropic import AsyncAnthropic

CalibrationMode = Literal["none", "few_shot"]

DEFAULT_DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)


@dataclass
class JudgeScore:
    overall: float
    dimensions: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    judge_model: str = ""


class JudgeBackend(Protocol):
    async def score(self, system: str, prompt: str) -> str: ...


class AnthropicJudgeBackend:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


_JUDGE_SYSTEM_PROMPT = """\
You are an expert persona evaluator. You score synthesized customer personas \
on a 1-5 scale across multiple quality dimensions.

Scoring scale:
  1 = Very poor — fails this dimension entirely
  2 = Weak — major gaps, mostly generic or inconsistent
  3 = Acceptable — meets minimum bar but unremarkable
  4 = Good — solid quality with minor issues
  5 = Excellent — publication-ready, specific, grounded, distinctive

Dimensions to score:
- **grounded**: Are claims traceable to source evidence? Do record IDs exist and make sense?
- **distinctive**: Does this feel like a real individual, or a generic average?
- **coherent**: Are demographics, firmographics, vocabulary, and quotes internally consistent?
- **actionable**: Are goals/pains specific enough to drive product decisions?
- **voice_fidelity**: Do sample quotes sound like one consistent speaker?

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "grounded": <1-5>,
  "distinctive": <1-5>,
  "coherent": <1-5>,
  "actionable": <1-5>,
  "voice_fidelity": <1-5>,
  "overall": <1-5>,
  "rationale": "<brief justification>"
}
"""

_ANCHOR_SCORE_1 = """\
=== CALIBRATION EXAMPLE — Score: 1 (Very Poor) ===
{
  "name": "Tech User",
  "summary": "A person who uses technology.",
  "goals": ["Be more productive", "Save time"],
  "pains": ["Things are hard"],
  "source_evidence": [{"claim": "Uses technology", "record_ids": ["rec_001"], "confidence": 0.5}]
}
SCORES: grounded=1, distinctive=1, coherent=1, actionable=1, voice_fidelity=1, overall=1
WHY: Completely generic. No real grounding. Unusable."""

_ANCHOR_SCORE_3 = """\
=== CALIBRATION EXAMPLE — Score: 3 (Acceptable) ===
{
  "name": "Marketing Manager Maria",
  "summary": "A mid-level marketing manager at a SaaS company focused on lead generation.",
  "goals": ["Increase lead conversion rate", "Build better attribution models"],
  "pains": ["Difficulty tracking campaign performance across channels"],
  "vocabulary": ["MQL", "pipeline", "attribution"],
  "sample_quotes": ["I need to show the CEO that marketing drives revenue."],
  "source_evidence": [
    {"claim": "Focused on lead conversion", "record_ids": ["rec_012", "rec_034"], "confidence": 0.7},
    {"claim": "Manual reporting pain", "record_ids": ["rec_034"], "confidence": 0.8}
  ]
}
SCORES: grounded=3, distinctive=3, coherent=3, actionable=3, voice_fidelity=3, overall=3
WHY: Functional but not sharp. Goals are generic to any marketing manager."""

_ANCHOR_SCORE_5 = """\
=== CALIBRATION EXAMPLE — Score: 5 (Excellent) ===
{
  "name": "DevOps Dana — The Automation-First Platform Lead",
  "summary": "Senior platform engineer at a fintech who treats every manual process as a bug.",
  "goals": ["Automate deployment to <10min cycles", "Reduce CI flakiness from 12% to under 2% by Q3"],
  "pains": ["5+ hours/week debugging flaky CI tests other teams won't fix"],
  "vocabulary": ["toil", "blast radius", "golden path", "SLO", "error budget"],
  "sample_quotes": ["If I can't terraform it, it doesn't exist in my infrastructure."],
  "source_evidence": [
    {"claim": "Automate deployment", "record_ids": ["ga4_003", "ga4_007", "intercom_001"], "confidence": 0.9},
    {"claim": "Debugging flaky CI", "record_ids": ["intercom_001", "ga4_007"], "confidence": 0.95}
  ]
}
SCORES: grounded=5, distinctive=5, coherent=5, actionable=5, voice_fidelity=5, overall=5
WHY: Every claim has strong evidence. Persona is unmistakably specific. Goals quantified."""


def _build_judge_prompt(persona: dict) -> str:
    return (
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


def _build_calibrated_judge_prompt(persona: dict) -> str:
    return (
        "Below are three calibration examples showing what a 1, 3, and 5 look like. "
        "Use these as anchors when scoring the TARGET persona that follows.\n\n"
        + _ANCHOR_SCORE_1 + "\n\n"
        + _ANCHOR_SCORE_3 + "\n\n"
        + _ANCHOR_SCORE_5 + "\n\n"
        + "=== TARGET PERSONA TO SCORE ===\n"
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


def _parse_judge_response(text: str, model: str) -> JudgeScore:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in DEFAULT_DIMENSIONS},
                rationale=f"Failed to parse judge response: {text[:200]}",
                judge_model=model,
            )

    dimensions = {}
    for dim in DEFAULT_DIMENSIONS:
        val = data.get(dim)
        dimensions[dim] = float(val) if val is not None else float("nan")

    return JudgeScore(
        overall=float(data.get("overall", float("nan"))),
        dimensions=dimensions,
        rationale=data.get("rationale", ""),
        judge_model=model,
    )


class LLMJudge:
    def __init__(
        self,
        backend: JudgeBackend | None = None,
        model: str = "claude-sonnet-4-20250514",
        calibration: CalibrationMode = "few_shot",
    ) -> None:
        self.backend = backend
        self.model = model
        self.calibration = calibration

    async def score_persona(self, persona: dict) -> JudgeScore:
        if self.backend is None:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in DEFAULT_DIMENSIONS},
                rationale="No backend configured",
                judge_model=self.model,
            )

        if self.calibration == "few_shot":
            prompt = _build_calibrated_judge_prompt(persona)
        else:
            prompt = _build_judge_prompt(persona)

        response = await self.backend.score(
            system=_JUDGE_SYSTEM_PROMPT,
            prompt=prompt,
        )
        return _parse_judge_response(response, self.model)
