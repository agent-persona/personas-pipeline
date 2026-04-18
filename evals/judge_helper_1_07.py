"""Local judge helper for experiment 1.07 — field interdependence.

Copied from origin/exp-5.13:evaluation/evaluation/judges.py with minimal
adaptation: we only need score_persona(), not transcript or pairwise scoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from anthropic import AsyncAnthropic

CalibrationMode = Literal["none", "few_shot"]

DEFAULT_DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)

HUMANIZED_DIMENSIONS = DEFAULT_DIMENSIONS + ("humanness",)


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
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIJudgeBackend:
    """Judge backend using OpenAI-compatible API (Ollama, vLLM, etc.)."""

    def __init__(self, client, model: str) -> None:
        self.client = client  # openai.AsyncOpenAI
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


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


# ── Humanized scoring (extends default with humanness dimension) ──────

_HUMANIZED_JUDGE_SYSTEM_PROMPT = """\
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
- **humanness**: Does the persona read as a real person? Look for: discourse markers \
('That said', 'Here\\'s the thing'), hedging ('I think', 'probably'), specific anecdotes \
vs generic claims, sentence length variety, contractions, emotional texture. \
1 = robotic AI-generated lists, 3 = acceptable corporate voice, 5 = reads like a real person wrote it.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "grounded": <1-5>,
  "distinctive": <1-5>,
  "coherent": <1-5>,
  "actionable": <1-5>,
  "voice_fidelity": <1-5>,
  "humanness": <1-5>,
  "overall": <1-5>,
  "rationale": "<brief justification>"
}
"""

_HUMANIZED_ANCHOR_SCORE_1 = """\
=== CALIBRATION EXAMPLE — Score: 1 (Very Poor) ===
{
  "name": "Tech User",
  "summary": "A person who uses technology.",
  "goals": ["Be more productive", "Save time"],
  "pains": ["Things are hard"],
  "source_evidence": [{"claim": "Uses technology", "record_ids": ["rec_001"], "confidence": 0.5}]
}
SCORES: grounded=1, distinctive=1, coherent=1, actionable=1, voice_fidelity=1, humanness=1, overall=1
WHY: Completely generic. No real grounding. Unusable. Reads like a template, not a person."""

_HUMANIZED_ANCHOR_SCORE_3 = """\
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
SCORES: grounded=3, distinctive=3, coherent=3, actionable=3, voice_fidelity=3, humanness=2, overall=3
WHY: Functional but not sharp. Goals are generic to any marketing manager. Voice is corporate, not personal."""

_HUMANIZED_ANCHOR_SCORE_5 = """\
=== CALIBRATION EXAMPLE — Score: 5 (Excellent) ===
{
  "name": "DevOps Dana — The Automation-First Platform Lead",
  "summary": "Senior platform engineer at a fintech who treats every manual process as a bug.",
  "backstory": "I started out writing deploy scripts at 2am during incidents. After three years of that, I decided no one on my team should ever have to SSH into a box again. Now I run the platform team and honestly, I still get twitchy when I see a manual step in a runbook.",
  "goals": ["Automate deployment to <10min cycles", "Reduce CI flakiness from 12% to under 2% by Q3"],
  "pains": ["5+ hours/week debugging flaky CI tests other teams won't fix"],
  "vocabulary": ["toil", "blast radius", "golden path", "SLO", "error budget"],
  "sample_quotes": ["If I can't terraform it, it doesn't exist in my infrastructure."],
  "speech_patterns": ["Starts sentences with 'Honestly' or 'Look'", "Uses analogies from cooking"],
  "source_evidence": [
    {"claim": "Automate deployment", "record_ids": ["ga4_003", "ga4_007", "intercom_001"], "confidence": 0.9},
    {"claim": "Debugging flaky CI", "record_ids": ["intercom_001", "ga4_007"], "confidence": 0.95}
  ]
}
SCORES: grounded=5, distinctive=5, coherent=5, actionable=5, voice_fidelity=5, humanness=5, overall=5
WHY: Every claim has strong evidence. Persona is unmistakably specific. Goals quantified. Backstory and speech patterns make this read like a real person."""


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


def _build_humanized_calibrated_judge_prompt(persona: dict) -> str:
    return (
        "Below are three calibration examples showing what a 1, 3, and 5 look like. "
        "Use these as anchors when scoring the TARGET persona that follows.\n\n"
        + _HUMANIZED_ANCHOR_SCORE_1 + "\n\n"
        + _HUMANIZED_ANCHOR_SCORE_3 + "\n\n"
        + _HUMANIZED_ANCHOR_SCORE_5 + "\n\n"
        + "=== TARGET PERSONA TO SCORE ===\n"
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


def _parse_judge_response(
    text: str,
    model: str,
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
) -> JudgeScore:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if data is None:
        # Try regex extraction of individual fields
        data = {}
        for key in list(dimensions) + ["overall"]:
            match = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', text)
            if match:
                data[key] = float(match.group(1))
        rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
        if rat_match:
            data["rationale"] = rat_match.group(1)

    if not data:
        return JudgeScore(
            overall=float("nan"),
            dimensions={d: float("nan") for d in dimensions},
            rationale=f"Failed to parse judge response: {text[:200]}",
            judge_model=model,
        )

    dim_scores = {}
    for dim in dimensions:
        val = data.get(dim)
        dim_scores[dim] = float(val) if val is not None else float("nan")

    return JudgeScore(
        overall=float(data.get("overall", float("nan"))),
        dimensions=dim_scores,
        rationale=data.get("rationale", ""),
        judge_model=model,
    )


class LLMJudge:
    """LLM-as-judge with optional few-shot calibration."""

    DEFAULT_DIMENSIONS = DEFAULT_DIMENSIONS

    def __init__(
        self,
        backend: JudgeBackend | None = None,
        model: str = "claude-sonnet-4-20250514",
        calibration: CalibrationMode = "few_shot",
        dimensions: tuple[str, ...] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.calibration = calibration
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.system_prompt = system_prompt or _JUDGE_SYSTEM_PROMPT

    async def score_persona(self, persona: dict) -> JudgeScore:
        if self.backend is None:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in self.dimensions},
                rationale="No backend configured",
                judge_model=self.model,
            )

        use_humanized = self.dimensions == HUMANIZED_DIMENSIONS

        if self.calibration == "few_shot":
            if use_humanized:
                prompt = _build_humanized_calibrated_judge_prompt(persona)
            else:
                prompt = _build_calibrated_judge_prompt(persona)
        else:
            prompt = _build_judge_prompt(persona)

        response = await self.backend.score(
            system=self.system_prompt,
            prompt=prompt,
        )

        return _parse_judge_response(response, self.model, self.dimensions)
