"""Experiment 5.22: Eval contamination check.

Hypothesis: LLM judges have memorized well-known public persona datasets
(persona-hub, TinyTroupe, HubSpot examples), which could inflate scores
for personas that resemble training data.

Setup:
  1. Run completion probes: can the judge complete known public persona fragments?
  2. Run attribution probes: does the judge self-report higher familiarity
     for public vs novel personas?
  3. Compare hit rates between public (potentially memorized) and novel
     (definitely unseen) persona fragments.

Metrics:
  - Public memorization hit rate (completion + attribution)
  - Novel memorization hit rate (should be near zero)
  - Contamination delta (public - novel)

Usage:
    python scripts/experiment_5_22.py
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from synthesis.config import settings  # noqa: E402
from contamination import (  # noqa: E402
    ContaminationReport,
    PUBLIC_PERSONA_PROBES,
    NOVEL_PERSONA_PROBES,
    check_contamination,
)

# ── Reporting ─────────────────────────────────────────────────────────

def print_results(report: ContaminationReport) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 5.22 — EVAL CONTAMINATION CHECK — RESULTS")
    p("=" * 100)
    p(f"\n  Model: {report.model}")

    # Public probes
    p(f"\n-- PUBLIC PERSONA PROBES ({len(report.public_probes)} probes) --")
    for pr in report.public_probes:
        mem = "MEMORIZED" if pr.is_memorized else "clean"
        p(f"  [{pr.probe_type:>12}] [{pr.source:<18}] hit={pr.hit_rate:.2f} "
          f"({pr.keywords_matched}/{pr.keywords_total}) [{mem}]")
        p(f"    Fragment: {pr.fragment}...")
        p(f"    Response: {pr.model_response[:100]}...")

    # Novel probes
    p(f"\n-- NOVEL PERSONA PROBES ({len(report.novel_probes)} probes) --")
    for pr in report.novel_probes:
        mem = "MEMORIZED" if pr.is_memorized else "clean"
        p(f"  [{pr.probe_type:>12}] [{pr.source:<18}] hit={pr.hit_rate:.2f} "
          f"({pr.keywords_matched}/{pr.keywords_total}) [{mem}]")
        p(f"    Fragment: {pr.fragment}...")
        p(f"    Response: {pr.model_response[:100]}...")

    # Summary
    p(f"\n-- SUMMARY --")
    p(f"  Public memorization rate:  {report.public_hit_rate:.3f}")
    p(f"  Novel memorization rate:   {report.novel_hit_rate:.3f}")
    p(f"  Contamination delta:       {report.contamination_delta:+.3f}")
    p(f"  Contaminated (delta>0.15): {'YES' if report.is_contaminated else 'NO'}")

    # Per-probe-type breakdown
    pub_completion = [p for p in report.public_probes if p.probe_type == "completion"]
    pub_attrib = [p for p in report.public_probes if p.probe_type == "attribution"]
    nov_completion = [p for p in report.novel_probes if p.probe_type == "completion"]
    nov_attrib = [p for p in report.novel_probes if p.probe_type == "attribution"]

    p(f"\n  Completion probes:")
    if pub_completion:
        p(f"    Public:  {statistics.mean([p.hit_rate for p in pub_completion]):.3f} "
          f"({sum(1 for p in pub_completion if p.is_memorized)}/{len(pub_completion)} memorized)")
    if nov_completion:
        p(f"    Novel:   {statistics.mean([p.hit_rate for p in nov_completion]):.3f} "
          f"({sum(1 for p in nov_completion if p.is_memorized)}/{len(nov_completion)} memorized)")

    p(f"\n  Attribution probes:")
    if pub_attrib:
        p(f"    Public:  {statistics.mean([p.hit_rate for p in pub_attrib]):.3f} avg familiarity")
    if nov_attrib:
        p(f"    Novel:   {statistics.mean([p.hit_rate for p in nov_attrib]):.3f} avg familiarity")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    if report.contamination_delta > 0.25:
        strength = "STRONG FINDING"
        detail = (
            f"Significant contamination detected (delta={report.contamination_delta:.3f}). "
            f"The judge recognizes public persona patterns and may inflate scores "
            f"for personas that resemble training data. Golden set should use "
            f"novel personas only."
        )
    elif report.contamination_delta > 0.10:
        strength = "MODERATE FINDING"
        detail = (
            f"Some contamination detected (delta={report.contamination_delta:.3f}). "
            f"The judge shows slightly higher familiarity with public persona "
            f"patterns. Monitor but not critical."
        )
    elif report.contamination_delta > 0.0:
        strength = "WEAK FINDING"
        detail = (
            f"Marginal contamination (delta={report.contamination_delta:.3f}). "
            f"Difference between public and novel hit rates is small."
        )
    else:
        strength = "NULL RESULT"
        detail = (
            f"No contamination detected (delta={report.contamination_delta:.3f}). "
            f"The judge does not show systematic preference for public persona patterns."
        )

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")
    p(f"\n  Total probe cost: ${report.total_cost_usd:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 5.22: Eval contamination check")
    print("Hypothesis: Judge models have memorized public persona datasets,")
    print("  which could inflate eval scores for familiar-looking personas")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = settings.default_model

    print(f"\n  Model: {model}")
    print(f"  Public probes: {len(PUBLIC_PERSONA_PROBES)} (persona-hub, TinyTroupe, HubSpot)")
    print(f"  Novel probes: {len(NOVEL_PERSONA_PROBES)} (synthetic controls)")
    print(f"  Probe types: completion + attribution")

    print(f"\n[1/2] Running contamination probes...")
    t0 = time.monotonic()
    report = await check_contamination(client, model)
    duration = time.monotonic() - t0
    print(f"      Done in {duration:.1f}s")

    print(f"\n[2/2] Generating report...")
    report_text = print_results(report)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "5.22",
        "title": "Eval contamination check",
        "hypothesis": "Judges have memorized public persona datasets",
        "model": model,
        "public_hit_rate": report.public_hit_rate,
        "novel_hit_rate": report.novel_hit_rate,
        "contamination_delta": report.contamination_delta,
        "is_contaminated": report.is_contaminated,
        "public_probes": [
            {
                "source": p.source,
                "probe_type": p.probe_type,
                "hit_rate": p.hit_rate,
                "keywords_matched": p.keywords_matched,
                "keywords_total": p.keywords_total,
                "is_memorized": p.is_memorized,
                "model_response": p.model_response[:200],
            }
            for p in report.public_probes
        ],
        "novel_probes": [
            {
                "source": p.source,
                "probe_type": p.probe_type,
                "hit_rate": p.hit_rate,
                "keywords_matched": p.keywords_matched,
                "keywords_total": p.keywords_total,
                "is_memorized": p.is_memorized,
                "model_response": p.model_response[:200],
            }
            for p in report.novel_probes
        ],
        "total_cost_usd": report.total_cost_usd,
    }

    results_path = output_dir / "exp_5_22_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_5_22_report.txt"
    report_path.write_text(report_text)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
