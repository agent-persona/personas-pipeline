"""Experiment 5.19: Continuous vs binary metrics.

Evaluates two personas on five rubric dimensions using:
  - Binary scale: 0 (fail) or 1 (pass)
  - Continuous scale: 1-5 fine-grained quality rating

Claude Code acts as judge, scoring both personas thoughtfully.
Computes reliability statistics to determine which scale type
provides more discriminative signal.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


DIMENSIONS = ["grounded", "distinctive", "coherent", "actionable", "voice_fidelity"]

# ---------------------------------------------------------------------------
# Judge scores — assigned by Claude Code after reading both personas in full.
# ---------------------------------------------------------------------------
#
# persona_00: Alex the API-First DevOps Engineer
#   grounded      — cites hubspot_000, ga4_000-009, intercom_000 with confidence
#                   scores per claim. Every assertion traceable to a record ID.
#                   binary=1, continuous=5
#   distinctive   — fintech DevOps + IaC obsession + API-first stance is highly
#                   specific; not a generic "developer" archetype.
#                   binary=1, continuous=5
#   coherent      — goals (automate everything), pains (GraphQL drift, fragmented
#                   docs), motivations (reduce toil, IaC credibility) lock together
#                   into a single logical arc.
#                   binary=1, continuous=5
#   actionable    — product team knows exactly what to ship: stable GraphQL schema
#                   versioning, vendor-maintained Terraform provider, webhook retry
#                   SLA documentation.
#                   binary=1, continuous=5
#   voice_fidelity — quotes are sharp and technical; "I don't want to click
#                   anything" and "Webhooks are great until they silently drop
#                   events" sound like a real engineer. Slight formulaic edge keeps
#                   it from a perfect 5.
#                   binary=1, continuous=4
#
# persona_01: Maya the Freelance Brand Designer
#   grounded      — cites hubspot_004, intercom_004, ga4_011-020 with claim-level
#                   confidence. All key claims are evidence-linked.
#                   binary=1, continuous=5
#   distinctive   — solo freelancer, white-label obsession, brand-kit workflow is
#                   a specific niche. Archetype is recognisable but implementation
#                   detail is sharp.
#                   binary=1, continuous=4
#   coherent      — "time = money" runs through goals, pains, motivations and
#                   objections without contradiction.
#                   binary=1, continuous=5
#   actionable    — product team knows the three asks: white-label at lower tier,
#                   deeper template library, cleaner export presets.
#                   binary=1, continuous=5
#   voice_fidelity — "If the client sees your logo they'll ask why they shouldn't
#                   just use it themselves" is authentic. Consistent register.
#                   binary=1, continuous=4

SCORES: dict[str, dict[str, dict[str, int]]] = {
    "persona_00": {
        "grounded":       {"binary": 1, "continuous": 5},
        "distinctive":    {"binary": 1, "continuous": 5},
        "coherent":       {"binary": 1, "continuous": 5},
        "actionable":     {"binary": 1, "continuous": 5},
        "voice_fidelity": {"binary": 1, "continuous": 4},
    },
    "persona_01": {
        "grounded":       {"binary": 1, "continuous": 5},
        "distinctive":    {"binary": 1, "continuous": 4},
        "coherent":       {"binary": 1, "continuous": 5},
        "actionable":     {"binary": 1, "continuous": 5},
        "voice_fidelity": {"binary": 1, "continuous": 4},
    },
}


def binary_discrimination_rate(scores: dict[str, dict[str, dict[str, int]]]) -> float:
    """Fraction of dimensions where the two personas scored DIFFERENTLY on binary scale.

    Designed for exactly two personas — extend to N by comparing all pairs if needed.
    """
    personas = list(scores.keys())
    if len(personas) != 2:
        raise ValueError("Discrimination rate requires exactly two personas")
    p0, p1 = personas
    different = sum(
        1
        for dim in DIMENSIONS
        if scores[p0][dim]["binary"] != scores[p1][dim]["binary"]
    )
    return different / len(DIMENSIONS)


def coefficient_of_variation(scores: dict[str, dict[str, dict[str, int]]]) -> float:
    """CV (std/mean) of continuous scores pooled across all personas × dimensions."""
    all_continuous = [
        scores[p][dim]["continuous"]
        for p in scores
        for dim in DIMENSIONS
    ]
    mean = statistics.mean(all_continuous)
    stdev = statistics.pstdev(all_continuous)  # population stdev (full set, not sample)
    return stdev / mean if mean else 0.0


def per_persona_cv(
    scores: dict[str, dict[str, dict[str, int]]],
) -> dict[str, float]:
    """CV for each persona across its five dimension scores."""
    result = {}
    for persona, dims in scores.items():
        vals = [dims[d]["continuous"] for d in DIMENSIONS]
        mean = statistics.mean(vals)
        stdev = statistics.pstdev(vals)
        result[persona] = stdev / mean if mean else 0.0
    return result


def per_dimension_cv(
    scores: dict[str, dict[str, dict[str, int]]],
) -> dict[str, float]:
    """CV for each dimension across personas."""
    result = {}
    for dim in DIMENSIONS:
        vals = [scores[p][dim]["continuous"] for p in scores]
        mean = statistics.mean(vals)
        stdev = statistics.pstdev(vals)
        result[dim] = stdev / mean if mean else 0.0
    return result


def run_analysis(persona_dir: Path) -> dict:
    """Load personas for validation, then apply pre-scored judgements and compute stats.

    Scores are hardcoded judge assessments (Claude Code as judge) — loading the
    persona files here confirms they exist and matches the SCORES keys.
    """
    persona_files = {p.stem for p in sorted(persona_dir.glob("persona_*.json"))}
    missing = set(SCORES.keys()) - persona_files
    if missing:
        raise FileNotFoundError(f"Persona files missing for scored keys: {missing}")

    bdr = binary_discrimination_rate(SCORES)
    pooled_cv = coefficient_of_variation(SCORES)
    p_cv = per_persona_cv(SCORES)
    d_cv = per_dimension_cv(SCORES)

    # Determine which scale provides more signal.
    # Binary: discrimination rate = 0 means zero signal between these two personas.
    # Continuous: any CV > 0 means some spread exists.
    binary_signal = "NONE" if bdr == 0.0 else "SOME"
    continuous_signal = "NONE" if pooled_cv == 0.0 else "SOME"

    if bdr == 0.0 and pooled_cv > 0.0:
        winner = "continuous"
    elif bdr > 0.0 and pooled_cv == 0.0:
        winner = "binary"
    elif bdr == 0.0 and pooled_cv == 0.0:
        winner = "neither — both flat"
    else:
        # Both have signal; prefer the one with relatively more spread.
        # Normalise binary discrimination (already 0-1) and continuous CV.
        winner = "continuous" if pooled_cv > bdr else "binary"

    results = {
        "scores": SCORES,
        "binary_discrimination_rate": bdr,
        "continuous_pooled_cv": pooled_cv,
        "per_persona_cv": p_cv,
        "per_dimension_cv": d_cv,
        "binary_signal": binary_signal,
        "continuous_signal": continuous_signal,
        "more_discriminative_scale": winner,
    }
    return results


def print_report(results: dict) -> None:
    print("\n=== Experiment 5.19: Continuous vs Binary Metrics ===\n")

    print("Judge Scores")
    print(f"{'Dimension':<18} {'p00 bin':>7} {'p00 1-5':>7} {'p01 bin':>7} {'p01 1-5':>7}")
    print("-" * 50)
    for dim in DIMENSIONS:
        p00 = results["scores"]["persona_00"][dim]
        p01 = results["scores"]["persona_01"][dim]
        print(
            f"{dim:<18} {p00['binary']:>7} {p00['continuous']:>7} "
            f"{p01['binary']:>7} {p01['continuous']:>7}"
        )

    print()
    print("Reliability Analysis")
    bdr = results["binary_discrimination_rate"]
    cv = results["continuous_pooled_cv"]
    print(f"  Binary  discrimination rate : {bdr:.2f} ({bdr*100:.0f}% of dimensions differ)")
    print(f"  Continuous pooled CV        : {cv:.4f}")
    print()
    print("  Per-persona CV (continuous):")
    for p, v in results["per_persona_cv"].items():
        print(f"    {p}: {v:.4f}")
    print()
    print("  Per-dimension CV (continuous):")
    for d, v in results["per_dimension_cv"].items():
        print(f"    {d:<18}: {v:.4f}")
    print()

    winner = results["more_discriminative_scale"]
    print(f"  More discriminative scale: {winner.upper()}")

    if bdr == 0.0:
        print()
        print("  NOTE: Binary yielded zero discrimination between these two high-quality")
        print("  personas — both passed every dimension. Continuous scale captured subtle")
        print("  quality differences (scores of 4 vs 5) that binary cannot express.")
        print("  Hypothesis REFUTED for high-quality persona sets.")
    else:
        print()
        print("  NOTE: Binary did discriminate. Hypothesis supported.")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    persona_dir = repo_root / "output"
    results = run_analysis(persona_dir)
    print_report(results)

    out_dir = repo_root / "output" / "experiments" / "exp-5.19-continuous-vs-binary-metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
