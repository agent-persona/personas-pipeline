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
    - Judge <-> human correlation study
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from synthesis.config import settings
from synthesis.provider_registry import normalize_provider, validate_provider_model


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


@dataclass
class PairwiseResult:
    """Result of a single pairwise comparison between two personas."""

    winner: str  # "a", "b", or "tie"
    criteria_winners: dict[str, str]  # per-criterion winner
    rationale: str
    position_was_swapped: bool  # True if we randomized order


@dataclass
class PairwiseSummary:
    """Aggregated results across multiple pairwise comparisons."""

    preference_rate_a: float
    preference_rate_b: float
    tie_rate: float
    per_criterion_rates: dict[str, dict[str, float]]  # criterion -> {a, b, tie rates}
    results: list[PairwiseResult]


class JudgeBackend(Protocol):
    async def score(self, prompt: str) -> str: ...


def _extract_anthropic_text(content_blocks: list[object]) -> str:
    """Return concatenated text blocks, skipping thinking/tool blocks."""
    text_parts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


class AnthropicJudgeBackend:
    """Judge backend using Anthropic's API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        if not base_url:
            os.environ.pop("ANTHROPIC_BASE_URL", None)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)
        self.model = model

    async def score(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_anthropic_text(response.content)


class OpenAICompatibleJudgeBackend:
    """Judge backend for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def score(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


def _gemini_api_key() -> str:
    selector = settings.gemini_api_key_selector.strip().lower()
    if selector == "turkey":
        return settings.gemini_api_key_turkey
    return settings.gemini_api_key_max


def build_judge_backend_from_settings(model: str) -> JudgeBackend:
    provider = normalize_provider(settings.judge_provider or settings.model_provider)
    validate_provider_model(provider, model, label="Judge")

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing for anthropic judge provider")
        return AnthropicJudgeBackend(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url or None,
            model=model,
        )

    if provider == "zai":
        if not settings.z_ai_glm_api_key:
            raise RuntimeError("Z_AI_GLM_API_KEY missing for zai judge provider")
        return AnthropicJudgeBackend(
            api_key=settings.z_ai_glm_api_key,
            base_url=settings.z_ai_anthropic_base_url,
            model=model,
        )

    if provider == "minimax":
        if not settings.minimax_api_key:
            raise RuntimeError("MINIMAX_API_KEY missing for minimax judge provider")
        return AnthropicJudgeBackend(
            api_key=settings.minimax_api_key,
            base_url=settings.minimax_anthropic_base_url,
            model=model,
        )

    if provider == "gemini":
        api_key = _gemini_api_key()
        if not api_key:
            raise RuntimeError("Gemini API key missing for gemini judge provider")
        return OpenAICompatibleJudgeBackend(
            api_key=api_key,
            base_url=settings.gemini_base_url,
            model=model,
        )

    if provider == "kimi":
        if not settings.kimi_api_key:
            raise RuntimeError("KIMI_API_KEY missing for kimi judge provider")
        return OpenAICompatibleJudgeBackend(
            api_key=settings.kimi_api_key,
            base_url=settings.kimi_base_url,
            model=model,
        )

    raise ValueError(f"Unsupported judge provider '{provider}'")


PAIRWISE_CRITERIA = (
    "groundedness",
    "developmental_fit",
    "historical_fit",
    "capability_coherence",
    "relational_realism",
    "overall_preference",
)

SCORE_DIMENSIONS = (
    "groundedness",
    "developmental_fit",
    "historical_fit",
    "capability_coherence",
    "relational_realism",
    "overall_preference",
)

DIMENSION_GUIDANCE = {
    "groundedness": "Are claims traceable to the provided evidence or internally well-supported?",
    "developmental_fit": "Do age, maturity, and life-stage signals fit together plausibly?",
    "historical_fit": "Do cultural, temporal, and career-era references feel historically plausible?",
    "capability_coherence": "Do skills, tools, and confidence levels align into a believable capability profile?",
    "relational_realism": "Do social context, interpersonal stance, and collaboration behavior feel human and specific?",
    "overall_preference": "Overall, how useful and believable is this persona for research and product decisions?",
}


def _build_pairwise_prompt(persona_left: dict, persona_right: dict) -> str:
    criteria_list = "\n".join(f"  - {c}" for c in PAIRWISE_CRITERIA)
    return f"""You are an expert judge comparing two user personas generated from the same source data.

Compare Persona A and Persona B on each criterion below. For each criterion, pick a winner: "a", "b", or "tie".
Then give an overall preference and a short rationale.

Criteria:
{criteria_list}

--- Persona A ---
{json.dumps(persona_left, indent=2)}

--- Persona B ---
{json.dumps(persona_right, indent=2)}

Respond with ONLY valid JSON in this exact format (no markdown fencing):
{{
  "criteria": {{
    "groundedness": "a" | "b" | "tie",
    "developmental_fit": "a" | "b" | "tie",
    "historical_fit": "a" | "b" | "tie",
    "capability_coherence": "a" | "b" | "tie",
    "relational_realism": "a" | "b" | "tie",
    "overall_preference": "a" | "b" | "tie"
  }},
  "winner": "a" | "b" | "tie",
  "rationale": "brief explanation"
}}"""


def _build_score_prompt(
    persona: dict,
    *,
    dimensions: tuple[str, ...] = SCORE_DIMENSIONS,
    reference_context: dict | None = None,
) -> str:
    dims = "\n".join(
        f"  - {dimension}: {DIMENSION_GUIDANCE.get(dimension, dimension)}"
        for dimension in dimensions
    )
    reference_section = ""
    if reference_context is not None:
        reference_section = (
            "\n\n--- Reference Context ---\n"
            f"{json.dumps(reference_context, indent=2)}"
            "\n\nUse the reference context as the source-of-truth when scoring groundedness "
            "and coherence. Penalize claims unsupported by it."
        )
    score_lines = ",\n".join(f'    "{dimension}": 0-5' for dimension in dimensions)
    return f"""You are an expert judge evaluating a user persona.

Score the following persona on each dimension from 0 to 5 (integers only).
0 = completely absent/wrong, 5 = exceptional.

Dimensions:
{dims}

--- Persona ---
{json.dumps(persona, indent=2)}
{reference_section}

Respond with ONLY valid JSON in this exact format (no markdown fencing):
{{
  "scores": {{
{score_lines}
  }},
  "rationale": "brief explanation"
}}"""


def _flip_winner(winner: str) -> str:
    """Swap a/b labels back after position randomization."""
    if winner == "a":
        return "b"
    if winner == "b":
        return "a"
    return "tie"


class LLMJudge:
    """LLM-as-judge with a pluggable backend."""

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
        self.model = getattr(backend, "model", model) if backend is not None else model
        self.dimensions = dimensions

    async def score_persona(self, persona: dict) -> JudgeScore:
        """Score a single persona JSON against the rubric (0-5 per dimension)."""
        return await self.score_persona_with_context(persona)

    async def score_persona_with_context(
        self,
        persona: dict,
        *,
        dimensions: tuple[str, ...] | None = None,
        reference_context: dict | None = None,
    ) -> JudgeScore:
        """Score a persona with optional rubric ablation and reference context."""
        if self.backend is None:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in SCORE_DIMENSIONS},
                rationale="No backend configured",
                judge_model=self.model,
            )

        selected_dimensions = dimensions or SCORE_DIMENSIONS
        prompt = _build_score_prompt(
            persona,
            dimensions=selected_dimensions,
            reference_context=reference_context,
        )
        raw = await self.backend.score(prompt)

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in SCORE_DIMENSIONS},
                rationale=f"Failed to parse judge response: {raw[:200]}",
                judge_model=self.model,
            )

        scores = parsed.get("scores", {})
        dims: dict[str, float] = {}
        for d in selected_dimensions:
            val = scores.get(d)
            dims[d] = float(val) if val is not None else float("nan")

        valid = [v for v in dims.values() if v == v]  # exclude NaN
        overall = sum(valid) / len(valid) if valid else float("nan")

        return JudgeScore(
            overall=overall,
            dimensions=dims,
            rationale=parsed.get("rationale", ""),
            judge_model=self.model,
        )

    async def score_transcript(
        self,
        persona: dict,
        transcript: list[dict],
    ) -> JudgeScore:
        """Score a twin chat transcript against the persona it was seeded with."""
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
    ) -> PairwiseResult:
        """Pairwise A/B preference judging with position-bias mitigation."""
        if self.backend is None:
            return PairwiseResult(
                winner="tie",
                criteria_winners={c: "tie" for c in PAIRWISE_CRITERIA},
                rationale="No backend configured",
                position_was_swapped=False,
            )

        # Randomly swap order to mitigate position bias
        swapped = random.random() < 0.5
        if swapped:
            left, right = persona_b, persona_a
        else:
            left, right = persona_a, persona_b

        prompt = _build_pairwise_prompt(left, right)
        raw = await self.backend.score(prompt)

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return PairwiseResult(
                winner="tie",
                criteria_winners={c: "tie" for c in PAIRWISE_CRITERIA},
                rationale=f"Failed to parse judge response: {raw[:200]}",
                position_was_swapped=swapped,
            )

        raw_winner = parsed.get("winner", "tie")
        raw_criteria = parsed.get("criteria", {})

        # Un-swap results back to original a/b mapping
        if swapped:
            winner = _flip_winner(raw_winner)
            criteria_winners = {
                c: _flip_winner(raw_criteria.get(c, "tie"))
                for c in PAIRWISE_CRITERIA
            }
        else:
            winner = raw_winner if raw_winner in ("a", "b", "tie") else "tie"
            criteria_winners = {
                c: raw_criteria.get(c, "tie")
                if raw_criteria.get(c, "tie") in ("a", "b", "tie")
                else "tie"
                for c in PAIRWISE_CRITERIA
            }

        return PairwiseResult(
            winner=winner,
            criteria_winners=criteria_winners,
            rationale=parsed.get("rationale", ""),
            position_was_swapped=swapped,
        )


async def run_pairwise_comparison(
    personas_a: list[dict],
    personas_b: list[dict],
    judge: LLMJudge,
) -> PairwiseSummary:
    """Run pairwise comparison on matched persona pairs and aggregate results."""
    if len(personas_a) != len(personas_b):
        raise ValueError(
            f"Persona lists must be the same length: {len(personas_a)} vs {len(personas_b)}"
        )

    results: list[PairwiseResult] = []
    for pa, pb in zip(personas_a, personas_b):
        result = await judge.pairwise(pa, pb)
        results.append(result)

    total = len(results)
    if total == 0:
        return PairwiseSummary(
            preference_rate_a=0.0,
            preference_rate_b=0.0,
            tie_rate=0.0,
            per_criterion_rates={},
            results=[],
        )

    a_wins = sum(1 for r in results if r.winner == "a")
    b_wins = sum(1 for r in results if r.winner == "b")
    ties = sum(1 for r in results if r.winner == "tie")

    per_criterion_rates: dict[str, dict[str, float]] = {}
    for c in PAIRWISE_CRITERIA:
        c_a = sum(1 for r in results if r.criteria_winners.get(c) == "a")
        c_b = sum(1 for r in results if r.criteria_winners.get(c) == "b")
        c_tie = total - c_a - c_b
        per_criterion_rates[c] = {
            "a": c_a / total,
            "b": c_b / total,
            "tie": c_tie / total,
        }

    return PairwiseSummary(
        preference_rate_a=a_wins / total,
        preference_rate_b=b_wins / total,
        tie_rate=ties / total,
        per_criterion_rates=per_criterion_rates,
        results=results,
    )
