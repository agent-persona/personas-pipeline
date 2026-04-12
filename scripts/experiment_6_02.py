"""Experiment 6.02: Coverage gaps.

Hypothesis: The persona set covers 60-70% of source records, leaving
30-40% uncovered — potentially the most interesting outlier segments.

Setup:
  1. Fetch all source records for the golden tenant.
  2. Synthesize personas from clusters.
  3. For each source record, compute max similarity to any persona.
  4. Report coverage fraction and identify uncovered records.

Metrics:
  - % of population represented (coverage fraction)
  - Per-persona coverage counts
  - Similarity distribution
  - Uncovered record characteristics

Usage:
    python scripts/experiment_6_02.py
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

from coverage_gaps import (  # noqa: E402
    CoverageReport,
    compute_coverage,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

COVERAGE_THRESHOLD = 0.05  # min similarity to count as covered


# ── Pipeline ──────────────────────────────────────────────────────────

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


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(
    report: CoverageReport,
    n_personas: int,
    synth_cost: float,
) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 6.02 — COVERAGE GAPS — RESULTS")
    p("=" * 100)

    p(f"\n  Personas: {n_personas}")
    p(f"  Source records: {report.total_records}")
    p(f"  Coverage threshold: {COVERAGE_THRESHOLD}")

    # Main metric
    pct = report.coverage_fraction * 100
    p(f"\n-- COVERAGE METRIC --")
    p(f"  Population represented: {report.covered_records}/{report.total_records} "
      f"({pct:.1f}%)")
    p(f"  Uncovered records:      {report.uncovered_records} "
      f"({100 - pct:.1f}%)")

    # Similarity distribution
    p(f"\n-- SIMILARITY DISTRIBUTION --")
    p(f"  Mean:   {report.avg_similarity:.4f}")
    p(f"  Median: {report.median_similarity:.4f}")
    p(f"  Min:    {report.min_similarity:.4f}")
    p(f"  Max:    {report.max_similarity:.4f}")

    # Histogram
    buckets = {"0.00-0.02": 0, "0.02-0.05": 0, "0.05-0.10": 0,
               "0.10-0.20": 0, "0.20+": 0}
    for r in report.per_record:
        s = r.max_similarity
        if s < 0.02:
            buckets["0.00-0.02"] += 1
        elif s < 0.05:
            buckets["0.02-0.05"] += 1
        elif s < 0.10:
            buckets["0.05-0.10"] += 1
        elif s < 0.20:
            buckets["0.10-0.20"] += 1
        else:
            buckets["0.20+"] += 1

    p(f"\n  Similarity histogram:")
    for bucket, count in buckets.items():
        bar = "#" * (count * 3)
        p(f"    {bucket}: {count:>3} {bar}")

    # Per-persona coverage
    p(f"\n-- PER-PERSONA COVERAGE --")
    for name, count in sorted(report.per_persona_coverage.items(),
                               key=lambda x: x[1], reverse=True):
        bar = "#" * count
        p(f"  {name[:40]:<42} {count:>3} records  {bar}")

    # Uncovered records
    if report.uncovered_details:
        p(f"\n-- UNCOVERED RECORDS ({len(report.uncovered_details)}) --")
        for r in report.uncovered_details[:10]:
            p(f"  {r.record_id}: max_sim={r.max_similarity:.4f}, "
              f"best_match={r.best_persona[:30]}, "
              f"words={r.record_words}")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    if pct >= 80:
        strength = "WEAK FINDING"
        detail = (f"High coverage ({pct:.0f}%). The persona set represents "
                  f"most of the population. Few gaps to address.")
    elif pct >= 60:
        strength = "MODERATE FINDING"
        detail = (f"Expected coverage ({pct:.0f}%). Matches the hypothesis "
                  f"that 60-70% coverage is typical. {report.uncovered_records} "
                  f"records may represent outlier segments worth investigating.")
    elif pct >= 40:
        strength = "STRONG FINDING"
        detail = (f"Low coverage ({pct:.0f}%). Significant population gaps. "
                  f"{report.uncovered_records} records are poorly represented "
                  f"by the current persona set.")
    else:
        strength = "STRONG FINDING (negative)"
        detail = (f"Very low coverage ({pct:.0f}%). The persona set fails to "
                  f"represent most of the population. Clustering or synthesis "
                  f"parameters need adjustment.")

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")
    p(f"\n  Synthesis cost: ${synth_cost:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 6.02: Coverage gaps")
    print("Hypothesis: Persona set covers 60-70% of source records,")
    print("  leaving 30-40% as uncovered outlier segments")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    # Step 1: Fetch all source records
    print("\n[1/4] Fetching source records...")
    crawler_records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    record_dicts = [r.model_dump() for r in raw_records]
    print(f"      Got {len(record_dicts)} source records")

    # Step 2: Segment and synthesize
    print("\n[2/4] Segmenting and synthesizing personas...")
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = [ClusterData.model_validate(c) for c in cluster_dicts]
    print(f"      Got {len(clusters)} clusters")

    personas: list[dict] = []
    synth_cost = 0.0
    for cluster in clusters:
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            personas.append(pd)
            synth_cost += result.total_cost_usd
            print(f"      {result.persona.name} (cost=${result.total_cost_usd:.4f})")
        except Exception as e:
            print(f"      FAILED cluster {cluster.cluster_id}: {e}")

    if not personas:
        print("ERROR: No personas synthesized")
        sys.exit(1)

    # Step 3: Compute coverage
    print(f"\n[3/4] Computing coverage ({len(record_dicts)} records x {len(personas)} personas)...")
    report = compute_coverage(record_dicts, personas, threshold=COVERAGE_THRESHOLD)

    # Step 4: Report
    print("\n[4/4] Generating report...")
    report_text = print_results(report, len(personas), synth_cost)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "6.02",
        "title": "Coverage gaps",
        "hypothesis": "60-70% coverage, 30-40% uncovered outliers",
        "model": settings.default_model,
        "n_records": report.total_records,
        "n_personas": len(personas),
        "coverage_threshold": COVERAGE_THRESHOLD,
        "coverage_fraction": report.coverage_fraction,
        "covered_records": report.covered_records,
        "uncovered_records": report.uncovered_records,
        "avg_similarity": report.avg_similarity,
        "median_similarity": report.median_similarity,
        "min_similarity": report.min_similarity,
        "max_similarity": report.max_similarity,
        "per_persona_coverage": report.per_persona_coverage,
        "uncovered_record_ids": [r.record_id for r in report.uncovered_details],
        "per_record": [
            {
                "record_id": r.record_id,
                "max_similarity": r.max_similarity,
                "best_persona": r.best_persona,
                "is_covered": r.is_covered,
            }
            for r in report.per_record
        ],
        "synthesis_cost_usd": synth_cost,
    }

    results_path = output_dir / "exp_6_02_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_6_02_report.txt"
    report_path.write_text(report_text)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
