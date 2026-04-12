"""Experiment 6.20: Persona deletion robustness.

Hypothesis: Removing one persona and re-synthesizing from the same cluster
reproduces a similar archetype (high absorption). The pipeline is robust
to deletion — coverage redistributes rather than collapses.

Setup:
  1. Synthesize full persona set (N personas from N clusters).
  2. For each persona, delete it and re-synthesize its cluster with the
     remaining persona names as existing_persona_names.
  3. Compare replacement to deleted original (similarity, absorption).
  4. Check if replacement duplicates a surviving persona.

Metrics:
  - Replacement similarity (Jaccard to deleted original)
  - Absorption rate (fraction of deleted topics in replacement)
  - Survivor similarity (does replacement collapse into a duplicate?)

Usage:
    python scripts/experiment_6_20.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from deletion_robustness import (  # noqa: E402
    DeletionResult,
    RobustnessReport,
    build_robustness_report,
    compute_deletion_result,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


def get_clusters() -> list[ClusterData]:
    crawler_records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(c) for c in cluster_dicts]


def print_results(report: RobustnessReport, synth_cost: float) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 6.20 — PERSONA DELETION ROBUSTNESS — RESULTS")
    p("=" * 100)
    p(f"\n  Original personas: {report.n_personas}")
    p(f"  Deletion tests: {len(report.deletions)}")

    p(f"\n-- DELETION RESULTS --")
    header = (f"  {'Deleted':<25}{'Replacement':<25}{'Repl Sim':>10}"
              f"{'Absorb':>10}{'Surv Sim':>10}{'Dup?':>8}")
    p(header)
    p("  " + "-" * 88)

    for d in report.deletions:
        dup = "YES" if d.is_duplicate_of_survivor else "no"
        p(f"  {d.deleted_name[:23]:<25}{d.replacement_name[:23]:<25}"
          f"{d.replacement_similarity:>10.3f}{d.absorption_rate:>10.3f}"
          f"{d.survivor_similarity:>10.3f}{dup:>8}")

    # Detail per deletion
    for d in report.deletions:
        p(f"\n  Deleted: {d.deleted_name}")
        p(f"    Replacement: {d.replacement_name}")
        p(f"    Replacement similarity: {d.replacement_similarity:.3f} "
          f"(Jaccard on {d.deleted_word_count} identity words)")
        p(f"    Absorption rate: {d.absorption_rate:.3f} "
          f"({d.absorbed_words}/{d.deleted_word_count} words absorbed)")
        p(f"    Survivor similarity: {d.survivor_similarity:.3f} "
          f"({'DUPLICATE' if d.is_duplicate_of_survivor else 'distinct'})")
        p(f"    Surviving personas: {', '.join(d.surviving_names)}")

    # Summary
    p(f"\n-- SUMMARY --")
    p(f"  Avg replacement similarity: {report.avg_replacement_similarity:.3f}")
    p(f"  Avg absorption rate:        {report.avg_absorption_rate:.3f}")
    p(f"  Avg survivor similarity:    {report.avg_survivor_similarity:.3f}")
    p(f"  Duplicates of survivors:    {report.duplicates_found}/{len(report.deletions)}")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    if report.avg_replacement_similarity > 0.4 and report.duplicates_found == 0:
        strength = "STRONG FINDING"
        detail = (f"Pipeline is robust to deletion. Replacements absorb "
                  f"{report.avg_absorption_rate:.0%} of deleted topics "
                  f"(similarity={report.avg_replacement_similarity:.3f}) "
                  f"without duplicating survivors.")
    elif report.avg_replacement_similarity > 0.2:
        strength = "MODERATE FINDING"
        detail = (f"Partial robustness. Replacements capture some deleted "
                  f"topics ({report.avg_absorption_rate:.0%}) but with "
                  f"variation (similarity={report.avg_replacement_similarity:.3f}).")
    elif report.duplicates_found > 0:
        strength = "STRONG FINDING (negative)"
        detail = (f"Coverage collapse: {report.duplicates_found} replacements "
                  f"duplicated survivors instead of filling the gap.")
    else:
        strength = "WEAK FINDING"
        detail = (f"Low replacement similarity ({report.avg_replacement_similarity:.3f}). "
                  f"Re-synthesis produces different archetypes from the same cluster.")

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")
    p(f"\n  Synthesis cost: ${synth_cost:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


async def main():
    print("=" * 72)
    print("EXPERIMENT 6.20: Persona deletion robustness")
    print("Hypothesis: Removing a persona and re-synthesizing reproduces")
    print("  the same archetype — coverage redistributes, not collapses")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/4] Segmenting...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    # Step 2: Synthesize full set
    print("\n[2/4] Synthesizing full persona set...")
    originals: list[tuple[ClusterData, dict]] = []
    synth_cost = 0.0
    for cluster in clusters:
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            originals.append((cluster, pd))
            synth_cost += result.total_cost_usd
            print(f"      {result.persona.name}")
        except Exception as e:
            print(f"      FAILED: {e}")

    if len(originals) < 2:
        print("ERROR: Need at least 2 personas for deletion test")
        sys.exit(1)

    # Step 3: Delete each persona and re-synthesize
    print("\n[3/4] Running deletion tests...")
    deletions: list[DeletionResult] = []

    for i, (del_cluster, del_persona) in enumerate(originals):
        del_name = del_persona.get("name", "?")
        survivors = [pd for j, (_, pd) in enumerate(originals) if j != i]
        survivor_names = [s.get("name", "?") for s in survivors]

        print(f"\n  Deleting: {del_name}")
        print(f"    Survivors: {', '.join(survivor_names)}")

        # Re-synthesize the deleted persona's cluster with survivors as existing
        # Update the cluster's tenant context with existing persona names
        cluster_dict = del_cluster.model_dump()
        cluster_dict["tenant"]["existing_persona_names"] = survivor_names
        modified_cluster = ClusterData.model_validate(cluster_dict)

        try:
            result = await synthesize(modified_cluster, synth_backend)
            replacement = result.persona.model_dump(mode="json")
            synth_cost += result.total_cost_usd
            print(f"    Replacement: {result.persona.name}")

            dr = compute_deletion_result(del_persona, replacement, survivors)
            deletions.append(dr)
            print(f"    Similarity: {dr.replacement_similarity:.3f}, "
                  f"Absorption: {dr.absorption_rate:.3f}, "
                  f"Survivor sim: {dr.survivor_similarity:.3f}")
        except Exception as e:
            print(f"    Re-synthesis FAILED: {e}")

    # Step 4: Report
    print("\n[4/4] Generating report...")
    report = build_robustness_report(deletions, len(originals))
    report_text = print_results(report, synth_cost)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "6.20",
        "title": "Persona deletion robustness",
        "hypothesis": "Deletion + re-synthesis reproduces the same archetype",
        "model": settings.default_model,
        "n_personas": len(originals),
        "deletions": [
            {
                "deleted": d.deleted_name,
                "replacement": d.replacement_name,
                "replacement_similarity": d.replacement_similarity,
                "absorption_rate": d.absorption_rate,
                "absorbed_words": d.absorbed_words,
                "deleted_word_count": d.deleted_word_count,
                "survivor_similarity": d.survivor_similarity,
                "is_duplicate": d.is_duplicate_of_survivor,
            }
            for d in deletions
        ],
        "avg_replacement_similarity": report.avg_replacement_similarity,
        "avg_absorption_rate": report.avg_absorption_rate,
        "synthesis_cost_usd": synth_cost,
    }

    (output_dir / "exp_6_20_results.json").write_text(json.dumps(results_data, indent=2))
    (output_dir / "exp_6_20_report.txt").write_text(report_text)
    print(f"\nResults saved to: {output_dir / 'exp_6_20_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
