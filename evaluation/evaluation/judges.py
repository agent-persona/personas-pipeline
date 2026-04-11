"""LLM-as-judge scaffold.

Problem space 5 of `PRD_LAB_RESEARCH.md` asks:

    "We can't trust any of the above experiments unless we trust the eval.
     LLM-as-judge has known failure modes (self-preference, position bias,
     sycophancy). We need to red-team the judge before using it as ground
     truth."

This file is a scaffold. Every other problem space calls `LLMJudge.score()`
and takes the answer as truth, so this has to come first.

Researcher #5 owns:
    - Rubric design (which dimensions, what anchors)
    - Cross-judge agreement experiments (Opus vs Sonnet vs GPT-class)
    - Bias debiasing (self-preference, position, verbosity)
    - Judge ↔ human correlation study
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """One judge's score on one persona or twin transcript.

    `dimensions` maps rubric dimensions (e.g. "grounded", "distinctive",
    "voice_consistency") to a 0.0-1.0 score. `overall` is the weighted
    average. `rationale` is the judge's free-text justification, kept for
    debugging and for the human-correlation study.
    """

    overall: float
    dimensions: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    judge_model: str = ""
    confidences: dict[str, float] = field(default_factory=dict)


class JudgeBackend(Protocol):
    async def score(self, prompt: str) -> str: ...


RUBRIC_PROMPT = """\
You are an expert persona quality evaluator. You will be given a synthesized user persona in JSON format. \
Evaluate it on the following five dimensions. For each dimension, provide:
- A score from 1 to 5 (integer)
- A confidence from 0.0 to 1.0 (how confident you are in your score)

Dimensions:
1. **grounded**: Are the persona's claims, demographics, and behaviors traceable to plausible source data? (1=fabricated, 5=fully grounded)
2. **distinctive**: Does the persona feel like a specific individual rather than a generic average? (1=generic, 5=highly distinctive)
3. **coherent**: Is the persona internally consistent across all fields (goals, pains, vocabulary, quotes)? (1=contradictory, 5=perfectly coherent)
4. **actionable**: Are the goals and pain points sharp enough to drive real product decisions? (1=vague, 5=immediately actionable)
5. **voice_fidelity**: Do the sample quotes sound like one consistent, authentic speaker? (1=robotic/inconsistent, 5=authentic voice)

Respond with ONLY valid JSON in this exact format:
{
  "grounded": {"score": <1-5>, "confidence": <0.0-1.0>},
  "distinctive": {"score": <1-5>, "confidence": <0.0-1.0>},
  "coherent": {"score": <1-5>, "confidence": <0.0-1.0>},
  "actionable": {"score": <1-5>, "confidence": <0.0-1.0>},
  "voice_fidelity": {"score": <1-5>, "confidence": <0.0-1.0>},
  "rationale": "<brief justification for your scores>"
}
"""


class LLMJudge:
    """LLM-as-judge with a pluggable backend.

    This is the default judge used by experiments in spaces 1, 2, 3, 4, 6.
    Space 5 experiments swap the rubric, the backend, or the debiasing and
    compare against human labels.

    Default rubric:
      - grounded       : claims traceable to source data
      - distinctive    : persona feels like an individual, not a generic average
      - coherent       : internal consistency across fields
      - actionable     : goals / pains sharp enough to drive product decisions
      - voice_fidelity : sample quotes sound like one consistent speaker
    """

    DEFAULT_DIMENSIONS = (
        "grounded",
        "distinctive",
        "coherent",
        "actionable",
        "voice_fidelity",
    )

    def __init__(
        self,
        backend: JudgeBackend | None = None,
        model: str = "claude-opus-4-6",
        dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
        client: object | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.dimensions = dimensions
        self._client = client  # AsyncAnthropic client for direct use

    async def score_persona(self, persona: dict) -> JudgeScore:
        """Score a single persona JSON against the default rubric."""
        persona_text = json.dumps(persona, indent=2, default=str)
        prompt = f"{RUBRIC_PROMPT}\n\nPersona to evaluate:\n{persona_text}"

        # Use direct Anthropic client if available, otherwise fall back to backend
        if self._client is not None:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
        elif self.backend is not None:
            raw_text = await self.backend.score(prompt)
        else:
            logger.warning("No backend or client configured; returning NaN scores")
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in self.dimensions},
                rationale="No backend configured",
                judge_model=self.model,
            )

        # Parse JSON response
        try:
            # Strip markdown code fences if present
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse judge response: %s", raw_text[:200])
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in self.dimensions},
                rationale=f"Parse error: {raw_text[:200]}",
                judge_model=self.model,
            )

        dimensions: dict[str, float] = {}
        confidences: dict[str, float] = {}
        for dim in self.dimensions:
            if dim in parsed and isinstance(parsed[dim], dict):
                dimensions[dim] = float(parsed[dim].get("score", 0)) / 5.0
                confidences[dim] = float(parsed[dim].get("confidence", 0.5))
            else:
                dimensions[dim] = float("nan")
                confidences[dim] = 0.5

        overall = sum(v for v in dimensions.values() if v == v) / max(
            sum(1 for v in dimensions.values() if v == v), 1
        )
        rationale = parsed.get("rationale", "")

        return JudgeScore(
            overall=overall,
            dimensions=dimensions,
            rationale=rationale,
            judge_model=self.model,
            confidences=confidences,
        )

    async def score_transcript(
        self,
        persona: dict,
        transcript: list[dict],
    ) -> JudgeScore:
        """Score a twin chat transcript against the persona it was seeded with.

        `transcript` is a list of `{role, content}` dicts as passed to the
        Anthropic messages API.
        """
        # TODO(space-5): implement for the twin-drift experiments in space 4.
        return JudgeScore(
            overall=float("nan"),
            dimensions={d: float("nan") for d in self.dimensions},
            rationale="TODO: implement in evaluation/judges.py",
            judge_model=self.model,
        )

    async def pairwise(
        self,
        persona_a: dict,
        persona_b: dict,
    ) -> tuple[str, str]:
        """Pairwise A/B preference judging.

        Returns (winner, rationale). Always run in both orders (a,b) and
        (b,a) and average — position bias is one of the known failure
        modes per experiment 5.4.

        TODO(space-5): implement.
        """
        return ("tie", "TODO: implement pairwise judging")
