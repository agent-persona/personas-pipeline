"""Experiment 5.23: Composite realism score.

Tests three weighting schemes for a headline 0-100 composite score derived
from five sub-metrics. Stability = max_composite - min_composite across schemes.

Sub-metrics:
  schema_validity   (0 or 100) — deterministic schema check
  groundedness      (0-100)    — from persona JSON .groundedness field × 100
  distinctiveness   (0-100)    — spread_score × 100 from exp-6.09 (set-level)
  judge_rubric      (0-100)    — 5 dimensions × 0-5 = 25 max, scaled to 100
  turing_estimate   (0-100)    — subjective "would a human think this is real?"

Weighting schemes:
  equal              — 0.2 each
  groundedness_heavy — groundedness 0.4, others 0.15 each
  distinctiveness_heavy — distinctiveness 0.4, others 0.15 each
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-metric data class
# ---------------------------------------------------------------------------

@dataclass
class SubMetrics:
    name: str
    schema_validity: float    # 0–100
    groundedness: float       # 0–100
    distinctiveness: float    # 0–100  (set-level, same for all personas in set)
    judge_rubric: float       # 0–100
    turing_estimate: float    # 0–100

    def as_dict(self) -> dict[str, float]:
        return {
            "schema_validity": self.schema_validity,
            "groundedness": self.groundedness,
            "distinctiveness": self.distinctiveness,
            "judge_rubric": self.judge_rubric,
            "turing_estimate": self.turing_estimate,
        }


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------

SCHEMES: dict[str, dict[str, float]] = {
    "equal": {
        "schema_validity": 0.2,
        "groundedness": 0.2,
        "distinctiveness": 0.2,
        "judge_rubric": 0.2,
        "turing_estimate": 0.2,
    },
    "groundedness_heavy": {
        "schema_validity": 0.15,
        "groundedness": 0.40,
        "distinctiveness": 0.15,
        "judge_rubric": 0.15,
        "turing_estimate": 0.15,
    },
    "distinctiveness_heavy": {
        "schema_validity": 0.15,
        "groundedness": 0.15,
        "distinctiveness": 0.40,
        "judge_rubric": 0.15,
        "turing_estimate": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def composite_score(metrics: SubMetrics, weights: dict[str, float]) -> float:
    """Weighted average of sub-metrics using the given weight dict."""
    values = metrics.as_dict()
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"
    return sum(weights[k] * values[k] for k in weights)


def stability(scores: list[float]) -> float:
    """Max minus min across a list of composite scores."""
    return max(scores) - min(scores)


def judge_rubric_score(rubric_scores_out_of_5: list[int]) -> float:
    """Convert list of 0-5 dimension scores to 0-100 scale.

    5 dimensions × 5 max each = 25 max total.
    """
    assert len(rubric_scores_out_of_5) == 5, "Expect exactly 5 dimension scores"
    return (sum(rubric_scores_out_of_5) / 25) * 100


# ---------------------------------------------------------------------------
# Load persona JSON and extract sub-metrics
# ---------------------------------------------------------------------------

def load_persona_metrics(
    json_path: Path,
    spread_score: float,
    judge_dimensions: list[int],
    turing_estimate: float,
) -> SubMetrics:
    """Read a persona JSON file and assemble SubMetrics.

    Args:
        json_path:        Path to persona_XX.json
        spread_score:     Set-level distinctiveness from exp-6.09 (0-1 scale)
        judge_dimensions: List of 5 integer scores (0-5 each) from judge
        turing_estimate:  0-100 estimate of human-likeness
    """
    data = json.loads(json_path.read_text())

    # schema_validity: presence of required top-level fields signals pass
    required_fields = {"cluster_id", "persona", "groundedness"}
    schema_val = 100.0 if required_fields.issubset(data.keys()) else 0.0

    # groundedness: from JSON field (1.0 scale → × 100)
    groundedness = float(data.get("groundedness", 1.0)) * 100

    name = data.get("persona", {}).get("name", json_path.stem)

    return SubMetrics(
        name=name,
        schema_validity=schema_val,
        groundedness=groundedness,
        distinctiveness=spread_score * 100,
        judge_rubric=judge_rubric_score(judge_dimensions),
        turing_estimate=turing_estimate,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path(__file__).parent.parent / "output"

    # exp-6.09 spread score (set-level, applies equally to both personas)
    SPREAD_SCORE = 0.762

    # Judge rubric: 5 dimensions scored 0-5 each
    # Dimensions: specificity, grounding, coherence, distinctiveness, actionability
    #
    # persona_00 (Alex the API-First DevOps Engineer):
    #   specificity=5, grounding=5, coherence=5, distinctiveness=4, actionability=5 → 24/25
    #
    # persona_01 (Maya the Freelance Brand Designer):
    #   specificity=5, grounding=5, coherence=5, distinctiveness=4, actionability=5 → 24/25

    personas_config = [
        {
            "path": base / "persona_00.json",
            "judge_dimensions": [5, 5, 5, 4, 5],  # 24/25
            "turing_estimate": 88.0,
        },
        {
            "path": base / "persona_01.json",
            "judge_dimensions": [5, 5, 5, 4, 5],  # 24/25
            "turing_estimate": 90.0,
        },
    ]

    all_metrics: list[SubMetrics] = []
    for cfg in personas_config:
        m = load_persona_metrics(
            json_path=cfg["path"],
            spread_score=SPREAD_SCORE,
            judge_dimensions=cfg["judge_dimensions"],
            turing_estimate=cfg["turing_estimate"],
        )
        all_metrics.append(m)

    # Print sub-metric values
    print("=" * 70)
    print("EXPERIMENT 5.23: COMPOSITE REALISM SCORE")
    print("=" * 70)
    print()
    print("Sub-metric values:")
    print(f"{'Metric':<25} {'persona_00':>12} {'persona_01':>12}")
    print("-" * 50)
    metric_keys = ["schema_validity", "groundedness", "distinctiveness", "judge_rubric", "turing_estimate"]
    for key in metric_keys:
        vals = [getattr(m, key) for m in all_metrics]
        print(f"{key:<25} {vals[0]:>11.1f} {vals[1]:>11.1f}")
    print()

    # Compute composite scores per scheme per persona
    print("Composite scores by weighting scheme:")
    print(f"{'Scheme':<25} {'persona_00':>12} {'persona_01':>12} {'Mean':>10}")
    print("-" * 62)

    scheme_means: dict[str, list[float]] = {}  # scheme → [per-persona composite]
    for scheme_name, weights in SCHEMES.items():
        scores = [composite_score(m, weights) for m in all_metrics]
        mean = sum(scores) / len(scores)
        scheme_means[scheme_name] = scores
        print(f"{scheme_name:<25} {scores[0]:>11.2f} {scores[1]:>11.2f} {mean:>9.2f}")
    print()

    # Stability: per-persona (how much does each persona's score vary across schemes?)
    print("Stability analysis (max - min composite across schemes):")
    print(f"{'Persona':<35} {'Stability':>12}")
    print("-" * 50)
    all_stabilities: list[float] = []
    for i, m in enumerate(all_metrics):
        persona_scores_across_schemes = [scheme_means[s][i] for s in SCHEMES]
        stab = stability(persona_scores_across_schemes)
        all_stabilities.append(stab)
        print(f"{m.name:<35} {stab:>11.2f}")

    mean_stability = sum(all_stabilities) / len(all_stabilities)
    print(f"\n{'Mean stability (all personas)':<35} {mean_stability:>11.2f}")
    print()

    # Interpretation
    if mean_stability < 5:
        interp = "HIGHLY STABLE — safe to use as headline number"
        signal = "STRONG"
    elif mean_stability < 10:
        interp = "MODERATELY STABLE — usable with caveats"
        signal = "MODERATE"
    elif mean_stability < 15:
        interp = "UNSTABLE — weighting choice materially affects headline"
        signal = "WEAK"
    else:
        interp = "HIGHLY UNSTABLE — composite is not a reliable headline number"
        signal = "NOISE/NEGATIVE"

    print(f"Interpretation: {interp}")
    print(f"Signal strength: {signal}")
    print()

    # Also print all composite scores per scheme for FINDINGS.md
    print("--- Raw values for FINDINGS.md ---")
    for scheme_name, scores in scheme_means.items():
        mean = sum(scores) / len(scores)
        print(f"{scheme_name}: p00={scores[0]:.2f}, p01={scores[1]:.2f}, mean={mean:.2f}")
    print(f"Stability values: p00={all_stabilities[0]:.2f}, p01={all_stabilities[1]:.2f}, mean={mean_stability:.2f}")


if __name__ == "__main__":
    main()
