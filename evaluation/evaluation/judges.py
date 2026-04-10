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

from dataclasses import dataclass, field
from typing import Protocol


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


class JudgeBackend(Protocol):
    async def score(self, prompt: str) -> str: ...


class LLMJudge:
    """LLM-as-judge with a pluggable backend.

    This is the default judge used by experiments in spaces 1, 2, 3, 4, 6.
    Space 5 experiments swap the rubric, the backend, or the debiasing and
    compare against human labels.

    Default rubric (TODO: finalize in space 5.5 rubric ablation):
      - grounded       : claims traceable to source data
      - distinctive    : persona feels like an individual, not a generic average
      - coherent       : internal consistency across fields
      - actionable     : goals / pains sharp enough to drive product decisions
      - voice_fidelity : sample quotes sound like one consistent speaker

    TODO(space-5): implement `score()` against the real Anthropic client.
    Today it returns a placeholder JudgeScore so dependent code is runnable.
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
    ) -> None:
        self.backend = backend
        self.model = model
        self.dimensions = dimensions

    async def score_persona(self, persona: dict) -> JudgeScore:
        """Score a single persona JSON against the default rubric."""
        # TODO(space-5): build a rubric prompt, call self.backend, parse.
        return JudgeScore(
            overall=float("nan"),
            dimensions={d: float("nan") for d in self.dimensions},
            rationale="TODO: implement in evaluation/judges.py",
            judge_model=self.model,
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

    PAIRWISE_PROMPT = """\
You are evaluating two persona documents generated from customer behavioral data.
Compare Persona A and Persona B on these dimensions:
- Grounded: claims traceable to source data
- Distinctive: feels like a real individual, not a generic average
- Coherent: internally consistent across all fields
- Actionable: goals/pains sharp enough to drive product decisions
- Voice fidelity: sample quotes sound like one consistent speaker

## Persona A
{persona_a}

## Persona B
{persona_b}

Which persona is better overall? Respond with EXACTLY one of:
- "A" if Persona A is better
- "B" if Persona B is better
- "TIE" if they are equally good

Then on a new line, give a brief rationale (1-2 sentences).
"""

    async def pairwise(
        self,
        persona_a: dict,
        persona_b: dict,
    ) -> tuple[str, str]:
        """Pairwise A/B preference judging.

        Returns (winner, rationale) where winner is 'A', 'B', or 'TIE'.

        Experiment 5.04 tests whether this judge has position or verbosity bias.
        """
        if self.backend is None:
            return ("TIE", "No backend configured")

        import json
        prompt = self.PAIRWISE_PROMPT.format(
            persona_a=json.dumps(persona_a, indent=2, default=str),
            persona_b=json.dumps(persona_b, indent=2, default=str),
        )

        raw = await self.backend.score(prompt)
        raw = raw.strip()

        # Parse the first line for the verdict
        first_line = raw.split("\n")[0].strip().upper()
        if first_line.startswith("A"):
            winner = "A"
        elif first_line.startswith("B"):
            winner = "B"
        else:
            winner = "TIE"

        rationale = raw[len(first_line):].strip() if len(raw) > len(first_line) else ""
        return (winner, rationale)
