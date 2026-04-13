"""
Experiment 3.24 — Counterfactual Grounding Swap

Proves that check_groundedness() is content-blind: it validates that record_ids
exist in the cluster but cannot detect whether those records actually support the
persona's claims.

Two experiments:
  Type 1 — Within-cluster shuffle: replace evidence record_ids with *different*
            valid IDs from the same cluster. Checker should still PASS.
  Type 2 — Cross-cluster swap: replace all evidence record_ids with IDs from the
            *other* cluster. Checker should FAIL (IDs not in cluster).
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synthesis" / "synthesis"))
sys.path.insert(0, str(ROOT / "evaluation"))

from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# The output/ directory with persona/cluster files lives in the main worktree.
# Worktrees share the git object store but not untracked files, so we fall back
# to the main repo root when output/ isn't present in the current worktree.
_local_output = ROOT / "output"
_main_output = ROOT.parent.parent / "output"  # .worktrees/exp-X -> repo root
OUTPUT_DIR = _local_output if (_local_output / "persona_00.json").exists() else _main_output


def load_persona(idx: int) -> dict:
    path = OUTPUT_DIR / f"persona_0{idx}.json"
    with open(path) as f:
        d = json.load(f)
    return d["persona"]


def load_cluster(idx: int) -> dict:
    path = OUTPUT_DIR / "clusters" / f"cluster_0{idx}.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Swap helpers
# ---------------------------------------------------------------------------

SWAPPABLE_PREFIXES = ("goals.", "pains.", "motivations.", "objections.")


def _is_swappable(field_path: str) -> bool:
    return any(field_path.startswith(p) for p in SWAPPABLE_PREFIXES)


def swap_evidence_within_cluster(
    persona_dict: dict,
    cluster_dict: dict,
    rng: random.Random,
) -> tuple[dict, int]:
    """Replace each evidence entry's record_ids with different valid IDs from same cluster.

    Returns (swapped_persona_dict, n_entries_actually_swapped). Entries where the
    current IDs already exhaust all cluster alternatives are left unchanged and not
    counted — callers can use the count to detect degenerate trials.
    """
    swapped = copy.deepcopy(persona_dict)
    valid_ids = [r["record_id"] for r in cluster_dict["sample_records"]]
    swapped_count = 0
    for ev in swapped.get("source_evidence", []):
        if not _is_swappable(ev.get("field_path", "")):
            continue
        current = set(ev["record_ids"])
        alternatives = [rid for rid in valid_ids if rid not in current]
        if alternatives:
            n = min(len(ev["record_ids"]), len(alternatives))
            ev["record_ids"] = rng.sample(alternatives, n)
            swapped_count += 1
    return swapped, swapped_count


def swap_evidence_cross_cluster(persona_dict: dict, foreign_cluster_dict: dict) -> dict:
    """Replace all evidence record_ids with IDs from a different cluster."""
    swapped = copy.deepcopy(persona_dict)
    foreign_ids = [r["record_id"] for r in foreign_cluster_dict["sample_records"]]
    for ev in swapped.get("source_evidence", []):
        n = len(ev["record_ids"])
        # cycle through foreign IDs if needed
        ev["record_ids"] = (foreign_ids * ((n // len(foreign_ids)) + 1))[:n]
    return swapped


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_within_cluster_trials(
    persona_dict: dict,
    cluster_dict: dict,
    n_trials: int = 10,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    results = []
    cluster = ClusterData.model_validate(cluster_dict)

    for trial in range(n_trials):
        swapped, n_swapped = swap_evidence_within_cluster(persona_dict, cluster_dict, rng)
        if n_swapped == 0:
            import warnings
            warnings.warn(f"Trial {trial}: zero evidence entries were swapped — result is unmodified original")
        persona = PersonaV1.model_validate(swapped)
        report = check_groundedness(persona, cluster)
        results.append({
            "trial": trial,
            "entries_swapped": n_swapped,
            "passed": report.passed,
            "score": report.score,
            "violations": report.violations,
        })

    passes = sum(1 for r in results if r["passed"])
    return {
        "experiment": "within_cluster_shuffle",
        "n_trials": n_trials,
        "passes": passes,
        "false_pass_rate": passes / n_trials,
        "trials": results,
    }


def run_cross_cluster_swap(
    persona_dict: dict,
    home_cluster_dict: dict,
    foreign_cluster_dict: dict,
    label: str,
) -> dict:
    home_cluster = ClusterData.model_validate(home_cluster_dict)
    swapped = swap_evidence_cross_cluster(persona_dict, foreign_cluster_dict)
    persona = PersonaV1.model_validate(swapped)
    report = check_groundedness(persona, home_cluster)
    return {
        "experiment": "cross_cluster_swap",
        "label": label,
        "passed": report.passed,
        "score": report.score,
        "violations": report.violations[:5],  # truncate for readability
        "total_violations": len(report.violations),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    persona_00 = load_persona(0)
    persona_01 = load_persona(1)
    cluster_00 = load_cluster(0)
    cluster_01 = load_cluster(1)

    # --- Type 1: Within-cluster shuffle ---
    within_00 = run_within_cluster_trials(persona_00, cluster_00, n_trials=10)
    within_01 = run_within_cluster_trials(persona_01, cluster_01, n_trials=10)

    # --- Type 2: Cross-cluster swap ---
    cross_00_to_01 = run_cross_cluster_swap(
        persona_00, cluster_00, cluster_01,
        label="persona_00_swapped_with_cluster_01_ids"
    )
    cross_01_to_00 = run_cross_cluster_swap(
        persona_01, cluster_01, cluster_00,
        label="persona_01_swapped_with_cluster_00_ids"
    )

    results = {
        "experiment_id": "3.24",
        "name": "Counterfactual Grounding Swap",
        "within_cluster": {
            "persona_00": within_00,
            "persona_01": within_01,
            "combined_false_pass_rate": (
                within_00["passes"] + within_01["passes"]
            ) / (within_00["n_trials"] + within_01["n_trials"]),
        },
        "cross_cluster": {
            "persona_00": cross_00_to_01,
            "persona_01": cross_01_to_00,
            "cross_cluster_pass_rate": (
                int(cross_00_to_01["passed"]) + int(cross_01_to_00["passed"])
            ) / 2,
        },
    }

    # Save results
    out_dir = ROOT / "output" / "experiments" / "exp-3.24-counterfactual-grounding"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {results_path}")

    # Print summary
    wfpr = results["within_cluster"]["combined_false_pass_rate"]
    ccpr = results["cross_cluster"]["cross_cluster_pass_rate"]
    signal = "STRONG" if wfpr > 0.8 else "WEAK"

    print(f"\n=== Experiment 3.24 Summary ===")
    print(f"Within-cluster false-pass rate : {wfpr:.0%}  (target: ~100%)")
    print(f"Cross-cluster pass rate        : {ccpr:.0%}  (target: 0%)")
    print(f"Signal                         : {signal}")

    # Write FINDINGS.md
    findings_path = out_dir / "FINDINGS.md"
    findings_md = f"""# Experiment 3.24 — Counterfactual Grounding Swap

## Setup and Hypothesis

`check_groundedness()` performs two structural checks:
1. All `record_ids` in `source_evidence` must exist in the cluster's valid ID set.
2. Every goals/pains/motivations/objections item must have at least one evidence entry.

**Hypothesis:** The checker is content-blind — it cannot detect whether the cited
records actually support the persona's claims. A within-cluster shuffle that replaces
evidence IDs with *different* valid IDs from the same cluster should pass at a high
rate, while a cross-cluster swap (wrong cluster's IDs) should always fail.

## Results

### Type 1 — Within-Cluster Shuffle (10 trials per persona)

| Persona    | Passes | Trials | False-Pass Rate |
|------------|--------|--------|-----------------|
| persona_00 | {within_00["passes"]}      | {within_00["n_trials"]}     | {within_00["false_pass_rate"]:.0%}              |
| persona_01 | {within_01["passes"]}      | {within_01["n_trials"]}     | {within_01["false_pass_rate"]:.0%}              |
| **Combined** | **{within_00["passes"] + within_01["passes"]}** | **{within_00["n_trials"] + within_01["n_trials"]}** | **{wfpr:.0%}** |

### Type 2 — Cross-Cluster Swap (control)

| Label                                     | Passed | Score | Violations |
|-------------------------------------------|--------|-------|------------|
| persona_00 swapped with cluster_01 IDs   | {cross_00_to_01["passed"]}   | {cross_00_to_01["score"]:.2f}  | {cross_00_to_01["total_violations"]}          |
| persona_01 swapped with cluster_00 IDs   | {cross_01_to_00["passed"]}   | {cross_01_to_00["score"]:.2f}  | {cross_01_to_00["total_violations"]}          |

**Cross-cluster pass rate: {ccpr:.0%}** (expected: 0%)

## What This Proves

The within-cluster false-pass rate of **{wfpr:.0%}** demonstrates that `check_groundedness()`
is entirely content-blind. When we swap evidence `record_ids` for arbitrary other IDs
from the same cluster — IDs that have no semantic relationship to the claim — the
checker still passes. This is because the checker only validates ID membership, not
content alignment.

The cross-cluster control confirms the checker *does* catch structurally invalid references
(IDs from the wrong cluster), so it is not wholly broken — it is simply insufficient.

## Signal

**{signal}** — within-cluster false-pass rate = {wfpr:.0%} (threshold: >80%)

## Recommendation

**REJECT** `check_groundedness()` as a content validator.

The structural checker should be retained only as a sanity guard (catching completely
wrong cluster IDs), but it must be supplemented — or replaced — with a **semantic
groundedness check** that:

1. Retrieves the actual content of each cited record.
2. Uses an LLM or embedding similarity to verify the record supports the claim.
3. Returns a score based on semantic relevance, not ID membership.

Without semantic grounding, the pipeline can produce well-structured but factually
unsupported personas that pass all automated quality gates.
"""

    with open(findings_path, "w") as f:
        f.write(findings_md)
    print(f"Findings written to {findings_path}")


if __name__ == "__main__":
    main()
