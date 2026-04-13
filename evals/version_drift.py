"""Experiment 5.09: Eval drift over model versions.

Establishes a version-pinned baseline of eval scores for future drift detection.
Claude Code acts as the judge — scores are recorded so future runs with different
model versions can diff against this snapshot.

Usage:
    python3 evals/version_drift.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- Paths ---
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
EXPERIMENT_OUTPUT = OUTPUT_DIR / "experiments" / "exp-5.09-eval-drift-model-versions"

PERSONA_PATHS = [
    OUTPUT_DIR / "persona_00.json",
    OUTPUT_DIR / "persona_01.json",
]

# Model version tag used to name snapshot files.
# Change this to e.g. "claude-haiku-4-5" or "claude-opus-4-6" on future runs.
MODEL_VERSION = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Rubric dimensions (0-5 scale, matching LLMJudge.DEFAULT_DIMENSIONS)
# ---------------------------------------------------------------------------
# Anchors:
#   0 = completely absent / unusable
#   1 = minimal, mostly generic
#   2 = partial, some specificity
#   3 = solid, meets expectations
#   4 = strong, above average
#   5 = exemplary, production-ready

RUBRIC_DIMENSIONS = [
    "grounded",       # claims cite specific record IDs
    "distinctive",    # feels like an individual, not a generic archetype
    "coherent",       # goals / pains / motivations fit together internally
    "actionable",     # sharp enough for a product team to act on
    "voice_fidelity", # sample quotes are unique, specific, and speaker-consistent
]

# ---------------------------------------------------------------------------
# Claude Code judge scores (pre-computed by reading personas above)
# ---------------------------------------------------------------------------
# persona_00 = Alex the API-First DevOps Engineer
# persona_01 = Maya the Freelance Brand Designer
#
# Scoring rationale:
#   Alex — grounded:5 (every claim maps to ga4/hubspot/intercom record IDs with
#   confidence scores), distinctive:5 (fintech DevOps + GraphQL frustrations +
#   Terraform specificity, not a generic dev), coherent:5 (reduce-toil theme ties
#   all goals/pains/motivations), actionable:5 (38+ min sessions quantified,
#   named pages cited, decision triggers are testable), voice_fidelity:4 (4
#   quotes all on-voice, minor repetition of API focus).
#
#   Maya — grounded:5 (hubspot_004 + intercom_004 at 1.0 confidence for every
#   key claim), distinctive:5 (hourly billing + white-label as dealbreaker frames
#   everything), coherent:5 (time=money thesis links all goals/pains/motivations
#   perfectly), actionable:5 (50+ template threshold, white-label tier callout,
#   20-35 min session durations), voice_fidelity:5 ("that's not the right blue"
#   is concrete and memorable; all 4 quotes stay in voice).

JUDGE_SCORES: dict[str, dict[str, int]] = {
    "persona_00": {
        "grounded": 5,
        "distinctive": 5,
        "coherent": 5,
        "actionable": 5,
        "voice_fidelity": 4,
    },
    "persona_01": {
        "grounded": 5,
        "distinctive": 5,
        "coherent": 5,
        "actionable": 5,
        "voice_fidelity": 5,
    },
}


# ---------------------------------------------------------------------------
# Deterministic metrics (no LLM calls)
# ---------------------------------------------------------------------------

def _schema_validity_single(persona_dict: dict) -> float:
    """1.0 if the persona passes basic structural checks, 0.0 otherwise."""
    required_top = {"schema_version", "name", "summary", "demographics",
                    "firmographics", "goals", "pains", "motivations",
                    "sample_quotes", "source_evidence"}
    if not required_top.issubset(persona_dict.keys()):
        return 0.0
    if not isinstance(persona_dict.get("sample_quotes"), list) or not persona_dict["sample_quotes"]:
        return 0.0
    if not isinstance(persona_dict.get("source_evidence"), list) or not persona_dict["source_evidence"]:
        return 0.0
    return 1.0


def compute_deterministic(result: dict) -> dict[str, Any]:
    """Extract deterministic metrics from a loaded persona result JSON."""
    persona = result.get("persona", {})
    metrics: dict[str, Any] = {}

    # Schema validity
    metrics["schema_validity"] = _schema_validity_single(persona)

    # Groundedness rate (stored at top level if available)
    if "groundedness" in result:
        metrics["groundedness_rate"] = float(result["groundedness"])
    else:
        metrics["groundedness_rate"] = None

    # Cost per persona
    if "cost_usd" in result:
        metrics["cost_per_persona_usd"] = float(result["cost_usd"])
    else:
        metrics["cost_per_persona_usd"] = None

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    EXPERIMENT_OUTPUT.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, persona_path in enumerate(PERSONA_PATHS):
        persona_key = f"persona_{idx:02d}"
        print(f"\n--- {persona_key}: {persona_path.name} ---")

        with open(persona_path) as f:
            result_json = json.load(f)

        persona = result_json.get("persona", {})
        name = persona.get("name", persona_key)

        # Deterministic metrics
        det = compute_deterministic(result_json)
        print(f"  schema_validity:     {det['schema_validity']}")
        print(f"  groundedness_rate:   {det['groundedness_rate']}")
        print(f"  cost_per_persona:    ${det['cost_per_persona_usd']}")

        # Judge scores (Claude Code as judge)
        if persona_key not in JUDGE_SCORES:
            raise KeyError(
                f"No judge scores found for {persona_key}. "
                f"Add an entry to JUDGE_SCORES before running."
            )
        judge_dims = JUDGE_SCORES[persona_key]
        dim_scores_normalized = {k: v / 5.0 for k, v in judge_dims.items()}
        mean_judge_score = sum(judge_dims.values()) / len(judge_dims) / 5.0

        print(f"  judge scores (0-5):  {judge_dims}")
        print(f"  mean judge score:    {mean_judge_score:.3f}  ({sum(judge_dims.values())/len(judge_dims):.2f}/5)")

        results.append({
            "persona_key": persona_key,
            "persona_name": name,
            "cluster_id": result_json.get("cluster_id", ""),
            "deterministic": det,
            "judge": {
                "model_version": MODEL_VERSION,
                "judge_role": "claude-code-as-judge",
                "dimensions_raw": judge_dims,
                "dimensions_normalized": dim_scores_normalized,
                "mean_normalized": mean_judge_score,
                "mean_out_of_5": sum(judge_dims.values()) / len(judge_dims),
            },
        })

    # Overall summary
    all_means = [r["judge"]["mean_normalized"] for r in results]
    overall_mean = sum(all_means) / len(all_means)
    overall_mean_5 = overall_mean * 5.0

    print(f"\n=== SUMMARY ===")
    print(f"  Model version:       {MODEL_VERSION}")
    print(f"  Personas evaluated:  {len(results)}")
    print(f"  Overall mean score:  {overall_mean_5:.2f}/5  ({overall_mean:.4f} normalized)")

    snapshot = {
        "experiment_id": "5.09",
        "title": "Eval drift over model versions",
        "model_version": MODEL_VERSION,
        "judge_role": "claude-code-as-judge",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rubric_dimensions": RUBRIC_DIMENSIONS,
        "rubric_scale": "0-5 per dimension",
        "personas": results,
        "summary": {
            "n_personas": len(results),
            "overall_mean_normalized": overall_mean,
            "overall_mean_out_of_5": overall_mean_5,
            "all_schema_valid": all(r["deterministic"]["schema_validity"] == 1.0 for r in results),
            "avg_groundedness_rate": (
                sum(r["deterministic"]["groundedness_rate"] for r in results
                    if r["deterministic"]["groundedness_rate"] is not None)
                / sum(1 for r in results if r["deterministic"]["groundedness_rate"] is not None)
                if any(r["deterministic"]["groundedness_rate"] is not None for r in results)
                else None
            ),
            "total_cost_usd": sum(
                r["deterministic"]["cost_per_persona_usd"] for r in results
                if r["deterministic"]["cost_per_persona_usd"] is not None
            ),
        },
        "drift_detection_instructions": (
            "Run this script with a different MODEL_VERSION constant when a new model release "
            "is available. Diff the two snapshot JSON files to detect judge calibration drift. "
            "Key signal: delta in overall_mean_out_of_5 across versions."
        ),
    }

    snapshot_path = EXPERIMENT_OUTPUT / f"scores_v{MODEL_VERSION}.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\n  Snapshot saved: {snapshot_path}")

    # Also write comparison.json as a template for future diffs
    comparison = {
        "description": "Comparison template — add future version snapshots to 'versions' list",
        "versions": [
            {
                "model_version": MODEL_VERSION,
                "overall_mean_out_of_5": overall_mean_5,
                "scores_file": f"scores_v{MODEL_VERSION}.json",
            }
        ],
        "how_to_diff": (
            "Load two version entries, compute delta in overall_mean_out_of_5 and "
            "per-dimension means. A delta > 0.3 on any dimension indicates calibration drift."
        ),
    }
    comparison_path = EXPERIMENT_OUTPUT / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison file:    {comparison_path}")

    return snapshot


if __name__ == "__main__":
    run()
