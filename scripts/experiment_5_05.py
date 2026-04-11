"""Experiment 5.05: Rubric ablation.

Hypothesis: Some rubric dimensions are redundant (correlation > 0.95) or
inert (removing them doesn't change persona rankings). Identifying these
lets us simplify the rubric without losing discriminative power.

Method:
  1. Generate personas from golden tenant (multiple synthesis repeats for
     sample size since tenant_acme_corp yields ~2 clusters).
  2. Score each persona with the full 5-dimension rubric (control).
  3. For each dimension: build an ablated rubric (4 remaining dims), re-score.
  4. Compute: pairwise correlation, ranking stability, score shifts.

Usage:
    python scripts/experiment_5_05.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

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

from evaluation.judges import JudgeBackend  # noqa: E402
from rubric_ablation import (  # noqa: E402
    RubricAblationHarness,
    analyze_ablation,
    format_analysis,
    FULL_DIMENSIONS,
)

# ── Config ────────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

# Multiple synthesis repeats to get enough personas for meaningful stats
SYNTHESIS_REPEATS = 3

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.05-rubric-ablation"


# ── Pipeline helpers ──────────────────────────────────────────────────

def get_clusters() -> list[ClusterData]:
    """Run ingest + segmentation."""
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


async def generate_personas(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
    repeats: int,
) -> list[tuple[str, dict]]:
    """Generate personas, running synthesis multiple times for sample size.

    Returns list of (persona_id, persona_dict) tuples.
    """
    personas = []
    for rep in range(repeats):
        for cluster in clusters:
            try:
                result = await synthesize(cluster, backend)
                persona_dict = result.persona.model_dump(mode="json")
                pid = f"{cluster.cluster_id}_rep{rep}"
                personas.append((pid, persona_dict))
                print(f"      Generated: {result.persona.name} ({pid})")
            except Exception as e:
                print(f"      FAILED: cluster={cluster.cluster_id} rep={rep}: {e}")
    return personas


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 5.05: Rubric Ablation")
    print("Hypothesis: Some rubric dimensions are redundant or inert")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        print("Harness is ready but cannot run without API key.")
        sys.exit(1)

    t_start = time.monotonic()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Use haiku for synthesis (cost-efficient), haiku for judging too
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)
    judge_backend = JudgeBackend(client=client, model=settings.default_model)

    # 1. Generate personas
    print("\n[1/4] Generating personas...")
    clusters = get_clusters()
    print(f"      {len(clusters)} clusters from {TENANT_ID}")
    print(f"      Running {SYNTHESIS_REPEATS} repeats per cluster...")

    personas = await generate_personas(clusters, synth_backend, SYNTHESIS_REPEATS)
    print(f"      Total personas: {len(personas)}")

    if not personas:
        print("ERROR: No personas generated. Cannot run ablation.")
        sys.exit(1)

    # 2. Run ablation
    print("\n[2/4] Running rubric ablation...")
    harness = RubricAblationHarness(backend=judge_backend, model=settings.default_model)
    results = await harness.run_ablation(personas)

    # 3. Analyze
    print("\n[3/4] Analyzing results...")
    analysis = analyze_ablation(results)
    report = format_analysis(analysis)
    print("\n" + report)

    # 4. Save results
    print("\n[4/4] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build serializable results
    results_data = {
        "experiment": "5.05",
        "title": "Rubric ablation",
        "hypothesis": (
            "Some rubric dimensions are redundant (r > 0.95) or inert "
            "(removal doesn't change rankings)"
        ),
        "config": {
            "tenant": TENANT_ID,
            "synthesis_repeats": SYNTHESIS_REPEATS,
            "n_personas": len(personas),
            "n_clusters": len(clusters),
            "judge_model": settings.default_model,
            "dimensions": list(FULL_DIMENSIONS),
        },
        "pairwise_correlations": {
            d1: {
                d2: (v if not (isinstance(v, float) and v != v) else None)
                for d2, v in corrs.items()
            }
            for d1, corrs in analysis.pairwise_correlations.items()
        },
        "ranking_stability": {
            k: (v if not (isinstance(v, float) and v != v) else None)
            for k, v in analysis.ranking_stability.items()
        },
        "score_shifts": {
            dk: {
                d: (v if not (isinstance(v, float) and v != v) else None)
                for d, v in shifts.items()
            }
            for dk, shifts in analysis.score_shifts.items()
        },
        "redundant_dimensions": analysis.redundant_dimensions,
        "inert_dimensions": analysis.inert_dimensions,
        "per_persona": [],
        "duration_seconds": time.monotonic() - t_start,
    }

    # Per-persona detail
    for r in results:
        entry = {
            "persona_id": r.persona_id,
            "persona_name": r.persona_dict.get("name", "unknown"),
            "control": None,
            "ablated": {},
        }
        if r.control_score:
            entry["control"] = {
                "scores": r.control_score.scores,
                "overall": r.control_score.overall,
                "rationale": r.control_score.rationale,
            }
        for drop_dim, abl in r.ablated_scores.items():
            entry["ablated"][f"drop_{drop_dim}"] = {
                "scores": abl.scores,
                "overall": abl.overall,
                "rationale": abl.rationale,
            }
        results_data["per_persona"].append(entry)

    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"  Results saved to: {results_path}")

    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text(report)
    print(f"  Report saved to:  {report_path}")

    # Print summary
    duration = time.monotonic() - t_start
    print(f"\nDone in {duration:.1f}s")
    print(f"Personas scored: {len(personas)}")
    print(f"Total scoring calls: {len(personas) * (1 + len(FULL_DIMENSIONS))}")

    if analysis.redundant_dimensions:
        print(f"REDUNDANT dimensions: {analysis.redundant_dimensions}")
    else:
        print("No redundant dimensions found")

    if analysis.inert_dimensions:
        print(f"INERT dimensions: {analysis.inert_dimensions}")
    else:
        print("No inert dimensions found")


if __name__ == "__main__":
    asyncio.run(main())
