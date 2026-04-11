"""Experiment 5.05: Rubric ablation harness.

Builds parameterized rubric prompts (full rubric vs one-dimension-dropped)
and scores personas with each variant. Computes pairwise dimension
correlation, ranking stability, and score shift when dimensions are removed.

Usage:
    harness = RubricAblationHarness(backend, model)
    results = await harness.run_ablation(personas)
    analysis = analyze_ablation(results)
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Sequence

# ── Rubric builder ────────────────────────────────────────────────────

FULL_DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)

DIMENSION_DESCRIPTIONS = {
    "grounded": (
        "**grounded** (1-5): Are claims traceable to source evidence? "
        "Do record IDs exist and make sense? "
        "1 = entirely fabricated. 5 = every claim grounded with evidence."
    ),
    "distinctive": (
        "**distinctive** (1-5): Does this feel like a real individual, "
        "or a generic average? "
        "1 = interchangeable boilerplate. 5 = vivid, unique, memorable."
    ),
    "coherent": (
        "**coherent** (1-5): Are demographics, firmographics, vocabulary, "
        "and quotes internally consistent? "
        "1 = contradictory. 5 = perfectly consistent."
    ),
    "actionable": (
        "**actionable** (1-5): Are goals/pains specific enough to drive "
        "product decisions? "
        "1 = vague platitudes. 5 = immediately actionable insights."
    ),
    "voice_fidelity": (
        "**voice_fidelity** (1-5): Do sample quotes sound like one "
        "consistent speaker? Is vocabulary coherent with role/industry? "
        "1 = generic or inconsistent. 5 = distinctive and consistent voice."
    ),
}


def build_rubric_system_prompt(dimensions: Sequence[str]) -> str:
    """Build a judge system prompt for the given subset of dimensions.

    This is the core of the ablation: by removing one dimension at a time,
    we can measure how the remaining scores shift.
    """
    dim_block = "\n".join(
        f"- {DIMENSION_DESCRIPTIONS[d]}" for d in dimensions
    )

    json_fields = ", ".join(f'"{d}": <1-5>' for d in dimensions)

    return (
        "You are an expert persona evaluator. You score synthesized customer "
        "personas on a 1-5 scale across quality dimensions.\n\n"
        "Scoring scale:\n"
        "  1 = Very poor — fails this dimension entirely\n"
        "  2 = Weak — major gaps, mostly generic or inconsistent\n"
        "  3 = Acceptable — meets minimum bar but unremarkable\n"
        "  4 = Good — solid quality with minor issues\n"
        "  5 = Excellent — publication-ready, specific, grounded, distinctive\n\n"
        f"Dimensions to score:\n{dim_block}\n\n"
        "Respond with ONLY a JSON object in this exact format "
        "(no markdown, no extra text):\n"
        "{\n"
        f"  {json_fields},\n"
        '  "overall": <1-5>,\n'
        '  "rationale": "<brief justification>"\n'
        "}\n\n"
        "The \"overall\" score should be the unweighted mean of the "
        "dimension scores."
    )


def build_judge_prompt(persona: dict) -> str:
    """Format a persona for the judge to score."""
    return (
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class AblationScore:
    """Score from one rubric variant on one persona."""
    variant: str  # "full" or "drop_<dimension>"
    dimensions_used: tuple[str, ...]
    persona_id: str
    scores: dict[str, float]  # dimension -> score
    overall: float
    rationale: str = ""


@dataclass
class AblationResult:
    """All scores for one persona across all rubric variants."""
    persona_id: str
    persona_dict: dict
    control_score: AblationScore | None = None
    ablated_scores: dict[str, AblationScore] = field(default_factory=dict)


@dataclass
class AblationAnalysis:
    """Summary statistics from the ablation experiment."""
    n_personas: int = 0
    # Per-dimension: correlation with other dims in full rubric
    pairwise_correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-dimension: what happens to rankings when this dim is dropped
    ranking_stability: dict[str, float] = field(default_factory=dict)
    # Per-dimension: mean score shift in surviving dimensions
    score_shifts: dict[str, dict[str, float]] = field(default_factory=dict)
    # Identified issues
    redundant_dimensions: list[str] = field(default_factory=list)
    inert_dimensions: list[str] = field(default_factory=list)


# ── Harness ───────────────────────────────────────────────────────────

def _parse_response(text: str, dimensions: Sequence[str]) -> dict[str, float]:
    """Parse judge JSON response into dimension scores."""
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
            return {d: float("nan") for d in dimensions}

    result = {}
    for d in dimensions:
        val = data.get(d)
        result[d] = float(val) if val is not None else float("nan")
    result["overall"] = float(data.get("overall", float("nan")))
    result["rationale"] = data.get("rationale", "")
    return result


class RubricAblationHarness:
    """Run full and ablated rubric scoring across a set of personas."""

    def __init__(self, backend, model: str = "claude-haiku-4-5-20251001") -> None:
        self.backend = backend
        self.model = model

    async def _score_with_rubric(
        self,
        persona: dict,
        persona_id: str,
        dimensions: tuple[str, ...],
        variant: str,
    ) -> AblationScore:
        """Score one persona with a specific rubric variant."""
        system = build_rubric_system_prompt(dimensions)
        prompt = build_judge_prompt(persona)

        response = await self.backend.score(system=system, prompt=prompt)
        parsed = _parse_response(response, dimensions)

        overall = parsed.pop("overall", float("nan"))
        rationale = parsed.pop("rationale", "")
        scores = {k: v for k, v in parsed.items() if k in dimensions}

        # Compute overall from scores if NaN
        valid_scores = [v for v in scores.values() if not math.isnan(v)]
        if math.isnan(overall) and valid_scores:
            overall = statistics.mean(valid_scores)

        return AblationScore(
            variant=variant,
            dimensions_used=dimensions,
            persona_id=persona_id,
            scores=scores,
            overall=overall,
            rationale=str(rationale),
        )

    async def run_ablation(
        self,
        personas: list[tuple[str, dict]],
    ) -> list[AblationResult]:
        """Run full ablation across all personas.

        Args:
            personas: List of (persona_id, persona_dict) tuples.

        Returns:
            List of AblationResult, one per persona.
        """
        results = []
        for persona_id, persona_dict in personas:
            result = AblationResult(
                persona_id=persona_id,
                persona_dict=persona_dict,
            )

            # Control: full rubric
            print(f"    Scoring {persona_id} with full rubric...")
            result.control_score = await self._score_with_rubric(
                persona_dict, persona_id, FULL_DIMENSIONS, "full"
            )

            # Ablated: drop each dimension
            for drop_dim in FULL_DIMENSIONS:
                remaining = tuple(d for d in FULL_DIMENSIONS if d != drop_dim)
                variant = f"drop_{drop_dim}"
                print(f"    Scoring {persona_id} with {variant}...")
                ablated = await self._score_with_rubric(
                    persona_dict, persona_id, remaining, variant
                )
                result.ablated_scores[drop_dim] = ablated

            results.append(result)

        return results


# ── Analysis ──────────────────────────────────────────────────────────

def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns NaN if degenerate."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1)) if n > 1 else 0
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1)) if n > 1 else 0
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def _kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall tau-b rank correlation. Returns NaN if too few items."""
    n = len(xs)
    if n < 3:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    pairs = n * (n - 1) / 2
    if pairs == 0:
        return float("nan")
    return (concordant - discordant) / pairs


def analyze_ablation(results: list[AblationResult]) -> AblationAnalysis:
    """Compute ablation analysis from experiment results."""
    analysis = AblationAnalysis(n_personas=len(results))

    if not results:
        return analysis

    # 1. Pairwise correlations from full rubric scores
    dim_scores: dict[str, list[float]] = {d: [] for d in FULL_DIMENSIONS}
    for r in results:
        if r.control_score:
            for d in FULL_DIMENSIONS:
                val = r.control_score.scores.get(d, float("nan"))
                dim_scores[d].append(val)

    for d1 in FULL_DIMENSIONS:
        analysis.pairwise_correlations[d1] = {}
        for d2 in FULL_DIMENSIONS:
            if d1 == d2:
                analysis.pairwise_correlations[d1][d2] = 1.0
            else:
                xs = [v for v, y in zip(dim_scores[d1], dim_scores[d2])
                      if not math.isnan(v) and not math.isnan(y)]
                ys = [y for v, y in zip(dim_scores[d1], dim_scores[d2])
                      if not math.isnan(v) and not math.isnan(y)]
                analysis.pairwise_correlations[d1][d2] = _pearson_r(xs, ys)

    # 2. Ranking stability: compare overall rankings with and without each dim
    control_overalls = []
    for r in results:
        if r.control_score:
            control_overalls.append(r.control_score.overall)
        else:
            control_overalls.append(float("nan"))

    for drop_dim in FULL_DIMENSIONS:
        ablated_overalls = []
        for r in results:
            abl = r.ablated_scores.get(drop_dim)
            if abl:
                ablated_overalls.append(abl.overall)
            else:
                ablated_overalls.append(float("nan"))

        # Filter out NaN pairs
        valid_pairs = [
            (c, a) for c, a in zip(control_overalls, ablated_overalls)
            if not math.isnan(c) and not math.isnan(a)
        ]
        if len(valid_pairs) >= 3:
            cs, as_ = zip(*valid_pairs)
            tau = _kendall_tau(list(cs), list(as_))
            analysis.ranking_stability[drop_dim] = tau
        elif len(valid_pairs) >= 2:
            # Too few for Kendall tau — use mean absolute delta
            deltas = [abs(c - a) for c, a in valid_pairs]
            analysis.ranking_stability[drop_dim] = 1.0 - statistics.mean(deltas) / 4.0
        else:
            analysis.ranking_stability[drop_dim] = float("nan")

    # 3. Score shift: for each dropped dim, how do surviving dims change?
    for drop_dim in FULL_DIMENSIONS:
        surviving = [d for d in FULL_DIMENSIONS if d != drop_dim]
        shifts: dict[str, list[float]] = {d: [] for d in surviving}
        for r in results:
            if not r.control_score:
                continue
            abl = r.ablated_scores.get(drop_dim)
            if not abl:
                continue
            for d in surviving:
                ctrl_val = r.control_score.scores.get(d, float("nan"))
                abl_val = abl.scores.get(d, float("nan"))
                if not math.isnan(ctrl_val) and not math.isnan(abl_val):
                    shifts[d].append(abl_val - ctrl_val)

        analysis.score_shifts[drop_dim] = {}
        for d in surviving:
            if shifts[d]:
                analysis.score_shifts[drop_dim][d] = statistics.mean(shifts[d])
            else:
                analysis.score_shifts[drop_dim][d] = float("nan")

    # 4. Identify redundant dimensions (correlation > 0.95)
    for d1 in FULL_DIMENSIONS:
        for d2 in FULL_DIMENSIONS:
            if d1 >= d2:
                continue
            corr = analysis.pairwise_correlations.get(d1, {}).get(d2, 0)
            if not math.isnan(corr) and corr > 0.95:
                if d1 not in analysis.redundant_dimensions:
                    analysis.redundant_dimensions.append(d1)
                if d2 not in analysis.redundant_dimensions:
                    analysis.redundant_dimensions.append(d2)

    # 5. Identify inert dimensions (removal doesn't change rankings)
    for drop_dim in FULL_DIMENSIONS:
        tau = analysis.ranking_stability.get(drop_dim, float("nan"))
        if not math.isnan(tau) and tau > 0.95:
            analysis.inert_dimensions.append(drop_dim)

    return analysis


def format_analysis(analysis: AblationAnalysis) -> str:
    """Format analysis as a human-readable report string."""
    lines = []

    lines.append("=== PAIRWISE DIMENSION CORRELATIONS (Pearson r) ===")
    header = f"{'':>16}" + "".join(f"{d:>16}" for d in FULL_DIMENSIONS)
    lines.append(header)
    for d1 in FULL_DIMENSIONS:
        row = f"{d1:>16}"
        for d2 in FULL_DIMENSIONS:
            val = analysis.pairwise_correlations.get(d1, {}).get(d2, float("nan"))
            if math.isnan(val):
                row += f"{'NaN':>16}"
            else:
                row += f"{val:>16.3f}"
        lines.append(row)

    lines.append("")
    lines.append("=== RANKING STABILITY (Kendall tau / stability score) ===")
    for drop_dim in FULL_DIMENSIONS:
        tau = analysis.ranking_stability.get(drop_dim, float("nan"))
        label = "INERT" if not math.isnan(tau) and tau > 0.95 else ""
        if math.isnan(tau):
            lines.append(f"  drop_{drop_dim}: NaN")
        else:
            lines.append(f"  drop_{drop_dim}: tau={tau:.3f} {label}")

    lines.append("")
    lines.append("=== SCORE SHIFTS (mean delta in surviving dims) ===")
    for drop_dim in FULL_DIMENSIONS:
        shifts = analysis.score_shifts.get(drop_dim, {})
        shift_strs = []
        for d, v in shifts.items():
            if math.isnan(v):
                shift_strs.append(f"{d}=NaN")
            else:
                shift_strs.append(f"{d}={v:+.2f}")
        lines.append(f"  drop_{drop_dim}: {', '.join(shift_strs)}")

    lines.append("")
    lines.append("=== FINDINGS ===")
    if analysis.redundant_dimensions:
        lines.append(f"  Redundant (r > 0.95): {analysis.redundant_dimensions}")
    else:
        lines.append("  No redundant dimensions found (all pairwise r <= 0.95)")

    if analysis.inert_dimensions:
        lines.append(f"  Inert (tau > 0.95): {analysis.inert_dimensions}")
    else:
        lines.append("  No inert dimensions found (all removals affect rankings)")

    return "\n".join(lines)
