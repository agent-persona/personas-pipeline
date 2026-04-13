"""Experiment 5.02: Cross-judge agreement harness.

Wraps multiple LLM judges (Opus, Sonnet, GPT-class, Gemini-class) and scores
identical persona outputs with each. Computes an agreement matrix per rubric
dimension and flags disagreement hotspots — dimensions where judge variance
is high and scores are therefore low-trust.

Usage:
    harness = MultiJudgeHarness.from_env()
    results = await harness.score_all(persona_dict)
    matrix  = compute_agreement_matrix(results)
    hotspots = find_disagreement_hotspots(matrix)
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# ── Rubric ───────────────────────────────────────────────────────────

DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)

JUDGE_RUBRIC_PROMPT = """\
You are an expert evaluator of synthetic marketing personas. Score the \
persona below on each dimension using a 0.0–1.0 scale.

## Scoring rubric

- **grounded** (0.0–1.0): Are claims traceable to source data? Does the \
persona cite specific record IDs and provide evidence entries for goals, \
pains, motivations, and objections? 1.0 = every claim is grounded with \
high-confidence evidence. 0.0 = entirely fabricated with no data linkage.

- **distinctive** (0.0–1.0): Does the persona feel like a real individual \
rather than a generic average? Are the vocabulary, quotes, and motivations \
specific enough to distinguish this persona from any other? 1.0 = vivid, \
unique, and memorable. 0.0 = interchangeable boilerplate.

- **coherent** (0.0–1.0): Is the persona internally consistent? Do the \
demographics, firmographics, goals, pains, vocabulary, and quotes all \
describe the same plausible person? 1.0 = perfectly consistent. 0.0 = \
contradictory or incoherent across fields.

- **actionable** (0.0–1.0): Are the goals, pains, and objections specific \
enough to inform real product and marketing decisions? 1.0 = a product \
team could immediately act on these insights. 0.0 = vague platitudes \
that apply to anyone.

- **voice_fidelity** (0.0–1.0): Do the sample quotes sound like one \
consistent speaker? Is the vocabulary list coherent with the persona's \
role, industry, and level? 1.0 = quotes and vocabulary are distinctive \
and internally consistent. 0.0 = generic or inconsistent voice.

## Output format

Respond with ONLY a JSON object (no markdown fences, no commentary):
{
  "grounded": <float>,
  "distinctive": <float>,
  "coherent": <float>,
  "actionable": <float>,
  "voice_fidelity": <float>,
  "overall": <float>,
  "rationale": "<1-2 sentence justification>"
}

The "overall" score should be the unweighted mean of the five dimension scores.

## Persona to evaluate

"""


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    """Scores from one judge on one persona."""
    judge_model: str
    judge_backend: str  # "anthropic" | "openai" | "google"
    overall: float
    dimensions: dict[str, float]
    rationale: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class AgreementCell:
    """Agreement stats for one dimension across all judges."""
    dimension: str
    mean_score: float
    std_dev: float
    min_score: float
    max_score: float
    spread: float  # max - min
    scores_by_judge: dict[str, float] = field(default_factory=dict)


@dataclass
class AgreementMatrix:
    """Full agreement analysis across all judges and dimensions."""
    cells: list[AgreementCell]
    pairwise_mad: dict[str, float]  # "judge_a vs judge_b" -> mean abs diff
    n_judges: int
    n_personas: int


@dataclass
class DisagreementHotspot:
    """A dimension flagged as low-trust due to high inter-judge variance."""
    dimension: str
    std_dev: float
    spread: float
    scores_by_judge: dict[str, float]
    trust_level: str  # "high", "medium", "low"


# ── Judge backends ───────────────────────────────────────────────────

async def _score_anthropic(
    client: AsyncAnthropic,
    model: str,
    persona_json: str,
) -> JudgeResult:
    """Score a persona using an Anthropic Claude model."""
    prompt = JUDGE_RUBRIC_PROMPT + persona_json

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    parsed = json.loads(text)

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    elif "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    dims = {d: float(parsed.get(d, 0.0)) for d in DIMENSIONS}
    return JudgeResult(
        judge_model=model,
        judge_backend="anthropic",
        overall=float(parsed.get("overall", statistics.mean(dims.values()))),
        dimensions=dims,
        rationale=parsed.get("rationale", ""),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )


async def _score_openai(
    api_key: str,
    model: str,
    persona_json: str,
) -> JudgeResult:
    """Score a persona using an OpenAI GPT-class model."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    prompt = JUDGE_RUBRIC_PROMPT + persona_json

    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    parsed = json.loads(text)

    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    # GPT-4o pricing: $2.50/M input, $10/M output
    cost = (in_tok * 2.5 + out_tok * 10) / 1_000_000

    dims = {d: float(parsed.get(d, 0.0)) for d in DIMENSIONS}
    return JudgeResult(
        judge_model=model,
        judge_backend="openai",
        overall=float(parsed.get("overall", statistics.mean(dims.values()))),
        dimensions=dims,
        rationale=parsed.get("rationale", ""),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )


async def _score_google(
    api_key: str,
    model: str,
    persona_json: str,
) -> JudgeResult:
    """Score a persona using a Google Gemini-class model via OpenAI compat."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    prompt = JUDGE_RUBRIC_PROMPT + persona_json

    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    parsed = json.loads(text)

    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    cost = (in_tok * 1.25 + out_tok * 5) / 1_000_000  # Gemini 1.5 Pro pricing

    dims = {d: float(parsed.get(d, 0.0)) for d in DIMENSIONS}
    return JudgeResult(
        judge_model=model,
        judge_backend="google",
        overall=float(parsed.get("overall", statistics.mean(dims.values()))),
        dimensions=dims,
        rationale=parsed.get("rationale", ""),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )


# ── Judge configurations ─────────────────────────────────────────────

@dataclass
class JudgeConfig:
    """Configuration for one judge model."""
    name: str
    backend: str  # "anthropic" | "openai" | "google"
    model: str
    api_key: str = ""


DEFAULT_JUDGES = [
    JudgeConfig(name="opus", backend="anthropic", model="claude-opus-4-6"),
    JudgeConfig(name="sonnet", backend="anthropic", model="claude-sonnet-4-6"),
    JudgeConfig(name="haiku", backend="anthropic", model="claude-haiku-4-5-20251001"),
    JudgeConfig(name="gpt-4o", backend="openai", model="gpt-4o"),
    JudgeConfig(name="gemini-pro", backend="google", model="gemini-2.5-flash"),
]


# ── Multi-judge harness ──────────────────────────────────────────────

class MultiJudgeHarness:
    """Orchestrates scoring across multiple judge models."""

    def __init__(self, judges: list[JudgeConfig]) -> None:
        self.judges = judges
        self._anthropic_clients: dict[str, AsyncAnthropic] = {}

    @classmethod
    def from_env(
        cls,
        anthropic_key: str = "",
        openai_key: str = "",
        google_key: str = "",
    ) -> MultiJudgeHarness:
        """Build harness from available API keys; skip unavailable backends."""
        import os

        anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = openai_key or os.environ.get("OPENAI_API_KEY", "")
        google_key = google_key or os.environ.get("GOOGLE_API_KEY", "")

        judges = []
        for jc in DEFAULT_JUDGES:
            if jc.backend == "anthropic" and anthropic_key:
                judges.append(JudgeConfig(
                    name=jc.name, backend=jc.backend,
                    model=jc.model, api_key=anthropic_key,
                ))
            elif jc.backend == "openai" and openai_key:
                judges.append(JudgeConfig(
                    name=jc.name, backend=jc.backend,
                    model=jc.model, api_key=openai_key,
                ))
            elif jc.backend == "google" and google_key:
                judges.append(JudgeConfig(
                    name=jc.name, backend=jc.backend,
                    model=jc.model, api_key=google_key,
                ))
            else:
                logger.info("Skipping judge %s: no API key for %s", jc.name, jc.backend)

        if not judges:
            raise ValueError("No judge backends available — set at least ANTHROPIC_API_KEY")

        return cls(judges)

    async def score_persona(self, persona: dict) -> list[JudgeResult]:
        """Score a single persona with all configured judges."""
        import asyncio

        persona_json = json.dumps(persona, indent=2)
        tasks = []
        for jc in self.judges:
            tasks.append(self._score_one(jc, persona_json))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        judge_results = []
        for jc, result in zip(self.judges, results):
            if isinstance(result, Exception):
                logger.error("Judge %s failed: %s", jc.name, result)
                judge_results.append(JudgeResult(
                    judge_model=jc.model,
                    judge_backend=jc.backend,
                    overall=float("nan"),
                    dimensions={d: float("nan") for d in DIMENSIONS},
                    rationale="",
                    error=str(result),
                ))
            else:
                judge_results.append(result)

        return judge_results

    async def _score_one(self, jc: JudgeConfig, persona_json: str) -> JudgeResult:
        if jc.backend == "anthropic":
            if jc.api_key not in self._anthropic_clients:
                self._anthropic_clients[jc.api_key] = AsyncAnthropic(api_key=jc.api_key)
            return await _score_anthropic(
                self._anthropic_clients[jc.api_key], jc.model, persona_json,
            )
        elif jc.backend == "openai":
            return await _score_openai(jc.api_key, jc.model, persona_json)
        elif jc.backend == "google":
            return await _score_google(jc.api_key, jc.model, persona_json)
        else:
            raise ValueError(f"Unknown backend: {jc.backend}")


# ── Agreement analysis ───────────────────────────────────────────────

def compute_agreement_matrix(
    all_results: list[list[JudgeResult]],
) -> AgreementMatrix:
    """Compute agreement stats across judges for a batch of personas.

    Args:
        all_results: list of per-persona judge result lists.
            all_results[i] = list of JudgeResult from each judge for persona i.

    Returns:
        AgreementMatrix with per-dimension stats and pairwise MAD.
    """
    n_personas = len(all_results)
    if n_personas == 0:
        return AgreementMatrix(cells=[], pairwise_mad={}, n_judges=0, n_personas=0)

    # Collect valid results only (filter out errors)
    valid_results = [
        [r for r in persona_results if r.error is None]
        for persona_results in all_results
    ]

    # Get all judge names that appear
    all_judge_names = []
    seen = set()
    for persona_results in valid_results:
        for r in persona_results:
            if r.judge_model not in seen:
                all_judge_names.append(r.judge_model)
                seen.add(r.judge_model)

    n_judges = len(all_judge_names)

    # Per-dimension agreement cells (averaged across personas)
    cells = []
    for dim in DIMENSIONS:
        # Collect all scores for this dimension, grouped by judge
        scores_by_judge: dict[str, list[float]] = {j: [] for j in all_judge_names}
        all_scores: list[float] = []

        for persona_results in valid_results:
            for r in persona_results:
                score = r.dimensions.get(dim, float("nan"))
                if not math.isnan(score):
                    scores_by_judge[r.judge_model].append(score)
                    all_scores.append(score)

        mean_per_judge = {
            j: statistics.mean(s) if s else float("nan")
            for j, s in scores_by_judge.items()
        }

        valid_means = [v for v in mean_per_judge.values() if not math.isnan(v)]
        cells.append(AgreementCell(
            dimension=dim,
            mean_score=statistics.mean(valid_means) if valid_means else float("nan"),
            std_dev=statistics.stdev(valid_means) if len(valid_means) >= 2 else 0.0,
            min_score=min(valid_means) if valid_means else float("nan"),
            max_score=max(valid_means) if valid_means else float("nan"),
            spread=(max(valid_means) - min(valid_means)) if valid_means else 0.0,
            scores_by_judge=mean_per_judge,
        ))

    # Pairwise mean absolute difference (averaged across all dimensions and personas)
    pairwise_mad: dict[str, float] = {}
    for i, j_a in enumerate(all_judge_names):
        for j_b in all_judge_names[i + 1:]:
            diffs = []
            for persona_results in valid_results:
                r_a = next((r for r in persona_results if r.judge_model == j_a), None)
                r_b = next((r for r in persona_results if r.judge_model == j_b), None)
                if r_a and r_b and r_a.error is None and r_b.error is None:
                    for dim in DIMENSIONS:
                        s_a = r_a.dimensions.get(dim, float("nan"))
                        s_b = r_b.dimensions.get(dim, float("nan"))
                        if not math.isnan(s_a) and not math.isnan(s_b):
                            diffs.append(abs(s_a - s_b))
            key = f"{j_a} vs {j_b}"
            pairwise_mad[key] = statistics.mean(diffs) if diffs else float("nan")

    return AgreementMatrix(
        cells=cells,
        pairwise_mad=pairwise_mad,
        n_judges=n_judges,
        n_personas=n_personas,
    )


def find_disagreement_hotspots(
    matrix: AgreementMatrix,
    std_threshold: float = 0.10,
    spread_threshold: float = 0.20,
) -> list[DisagreementHotspot]:
    """Identify dimensions where judges disagree most.

    Args:
        std_threshold: standard deviation above which a dimension is flagged.
        spread_threshold: max-min spread above which a dimension is flagged.

    Returns:
        List of DisagreementHotspot sorted by std_dev descending.
    """
    hotspots = []
    for cell in matrix.cells:
        if math.isnan(cell.std_dev):
            continue

        if cell.std_dev >= std_threshold or cell.spread >= spread_threshold:
            trust = "low"
        elif cell.std_dev >= std_threshold * 0.5 or cell.spread >= spread_threshold * 0.5:
            trust = "medium"
        else:
            trust = "high"

        hotspots.append(DisagreementHotspot(
            dimension=cell.dimension,
            std_dev=cell.std_dev,
            spread=cell.spread,
            scores_by_judge=cell.scores_by_judge,
            trust_level=trust,
        ))

    hotspots.sort(key=lambda h: h.std_dev, reverse=True)
    return hotspots
