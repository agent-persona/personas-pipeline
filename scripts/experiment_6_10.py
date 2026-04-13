"""Experiment 6.10: Diversity along specific axes.

Hypothesis: The synthesis pipeline defaults to uniform demographics on
certain axes (e.g., always "25-34" for age), producing personas that
are distinct in goals/pains but homogeneous on business-critical axes.

Setup:
  1. Synthesize personas from the golden tenant.
  2. Extract values for each axis (age, income, geo, role, industry, etc).
  3. Compute collapse rate and Gini per axis.

Metrics:
  - Axis-level Gini (0 = diverse, 1 = collapsed)
  - Collapse-to-default rate per axis
  - Number of collapsed vs diverse axes

Usage:
    python scripts/experiment_6_10.py
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

from axis_diversity import (  # noqa: E402
    AXES,
    DiversityReport,
    analyze_diversity,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


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

def print_results(report: DiversityReport, synth_cost: float) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 6.10 — DIVERSITY ALONG SPECIFIC AXES — RESULTS")
    p("=" * 100)
    p(f"\n  Personas: {report.n_personas}")

    # Main table
    p(f"\n-- AXIS DIVERSITY --")
    header = f"  {'Axis':<22}{'Unique':>8}{'Collapse':>10}{'Gini':>8}  {'Most Common Value':<35}{'Status'}"
    p(header)
    p("  " + "-" * 100)

    for a in sorted(report.axes, key=lambda x: x.gini, reverse=True):
        status = "COLLAPSED" if a.collapse_rate > 0.6 else ("diverse" if a.collapse_rate < 0.4 else "moderate")
        most_common_display = a.most_common[:33] if a.most_common else "N/A"
        p(f"  {a.axis:<22}{a.unique_count:>8}{a.collapse_rate:>10.2f}{a.gini:>8.3f}"
          f"  {most_common_display:<35}{status}")

    # Value breakdowns per axis
    p(f"\n-- VALUE BREAKDOWNS --")
    for a in report.axes:
        p(f"\n  {a.axis}:")
        for val, count in sorted(a.value_counts.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * (count * 5)
            pct = count / a.total * 100 if a.total > 0 else 0
            p(f"    {val[:45]:<47} {count:>2} ({pct:>5.1f}%) {bar}")

    # Summary
    p(f"\n-- SUMMARY --")
    p(f"  Avg Gini coefficient: {report.avg_gini:.3f}")
    p(f"  Avg collapse rate:    {report.avg_collapse_rate:.3f}")
    p(f"  Collapsed axes (>0.6): {', '.join(report.collapsed_axes) or 'none'}")
    p(f"  Diverse axes (<0.4):   {', '.join(report.diverse_axes) or 'none'}")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    n_collapsed = len(report.collapsed_axes)
    n_total = len(report.axes)

    if n_collapsed >= n_total * 0.5:
        strength = "STRONG FINDING"
        detail = (f"{n_collapsed}/{n_total} axes are collapsed. The pipeline "
                  f"defaults to uniform values on most dimensions. Diversity "
                  f"injection needed in the synthesis prompt.")
    elif n_collapsed >= 2:
        strength = "MODERATE FINDING"
        detail = (f"{n_collapsed} axes collapsed: {', '.join(report.collapsed_axes)}. "
                  f"These axes need explicit diversification in synthesis.")
    elif n_collapsed == 1:
        strength = "WEAK FINDING"
        detail = (f"Only {report.collapsed_axes[0]} is collapsed. Most axes "
                  f"show reasonable diversity.")
    else:
        strength = "NULL RESULT"
        detail = "No collapsed axes. Persona set shows diversity across all dimensions."

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")
    p(f"\n  Synthesis cost: ${synth_cost:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 6.10: Diversity along specific axes")
    print("Hypothesis: Pipeline defaults to uniform values on some")
    print("  axes, producing homogeneous demographics despite distinct personas")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/3] Synthesizing personas...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    personas: list[dict] = []
    synth_cost = 0.0
    for cluster in clusters:
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            personas.append(pd)
            synth_cost += result.total_cost_usd
            print(f"      {result.persona.name}")
        except Exception as e:
            print(f"      FAILED: {e}")

    if len(personas) < 2:
        print("ERROR: Need at least 2 personas")
        sys.exit(1)

    print(f"\n[2/3] Analyzing axis diversity ({len(personas)} personas x {len(AXES)} axes)...")
    report = analyze_diversity(personas)

    print("\n[3/3] Generating report...")
    report_text = print_results(report, synth_cost)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "6.10",
        "title": "Diversity along specific axes",
        "hypothesis": "Pipeline defaults to uniform values on some axes",
        "model": settings.default_model,
        "n_personas": report.n_personas,
        "avg_gini": report.avg_gini,
        "avg_collapse_rate": report.avg_collapse_rate,
        "collapsed_axes": report.collapsed_axes,
        "diverse_axes": report.diverse_axes,
        "axes": [
            {
                "axis": a.axis,
                "unique_count": a.unique_count,
                "collapse_rate": a.collapse_rate,
                "gini": a.gini,
                "most_common": a.most_common,
                "value_counts": a.value_counts,
            }
            for a in report.axes
        ],
        "synthesis_cost_usd": synth_cost,
    }

    (output_dir / "exp_6_10_results.json").write_text(json.dumps(results_data, indent=2))
    (output_dir / "exp_6_10_report.txt").write_text(report_text)
    print(f"\nResults saved to: {output_dir / 'exp_6_10_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
