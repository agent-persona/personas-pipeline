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
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


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
        anthropic_client: "AsyncAnthropic | None" = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.dimensions = dimensions
        # Direct Anthropic handle for the narrow methods that bypass the
        # full rubric/backend plumbing (e.g. `same_speaker`). Experiment
        # 1.3 passes this in directly; other spaces can ignore it.
        self.anthropic_client = anthropic_client

    async def same_speaker(self, reply_a: str, reply_b: str) -> bool:
        """Return True iff the judge thinks the two replies come from the same person.

        Minimal prompt, no rubric, no score — just a yes/no. Used by
        `evaluation.metrics.pairing_accuracy` in experiment 1.3.

        Self-preference bias: if `self.model` matches the model that
        generated the replies, the judge tends to over-predict "same".
        Callers should flag this in their result files.
        """
        if self.anthropic_client is None:
            raise RuntimeError(
                "LLMJudge.same_speaker requires an anthropic_client; "
                "pass one to LLMJudge(anthropic_client=...)."
            )
        prompt = (
            "You are judging whether two short replies were written by the same person.\n"
            "Focus on voice, word choice, punctuation habits, and tone — not topic.\n\n"
            f"Reply A:\n{reply_a}\n\n"
            f"Reply B:\n{reply_b}\n\n"
            'Answer with a single word: "yes" if the same person wrote both, "no" otherwise.'
        )
        response = await self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        return "yes" in text.strip().lower()

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
