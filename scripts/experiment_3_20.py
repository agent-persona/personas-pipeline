"""Experiment 3.20: Confidence-weighted corroboration.

Hypothesis: The LLM synthesizer assigns high confidence to claims backed by
only 1 record (over-confident hallucination risk). By cross-checking
confidence against corroboration depth (number of distinct records), we can
identify and reject these over-confident claims, improving calibration.

Setup:
  1. Synthesize personas using the standard pipeline.
  2. For each persona, compute corroboration analysis (confidence vs depth).
  3. Measure calibration score and count over-confident claims.
  4. Compare control (no enforcement) vs enforced (reject over-confident).

Metrics:
  - Calibration score (1.0 = confidence matches corroboration perfectly)
  - Over-confident claim count (confidence >= 0.8, corroboration == 1)
  - Avg confidence and avg corroboration depth
  - Groundedness impact when enforcement is enabled

Usage:
    python scripts/experiment_3_20.py
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
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
from synthesis.engine.groundedness import (  # noqa: E402
    check_corroboration,
    check_groundedness,
    CorroborationReport,
)
from synthesis.models.cluster import ClusterData  # noqa: E402

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


# ── Metrics ──────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    variant: str
    cluster_id: str = ""
    persona_name: str = ""
    success: bool = False
    # Groundedness
    groundedness_score: float = 0.0
    # Corroboration
    total_evidence: int = 0
    over_confident: int = 0
    well_corroborated: int = 0
    under_confident: int = 0
    avg_confidence: float = 0.0
    avg_corroboration: float = 0.0
    calibration_score: float = 0.0
    # Per-evidence breakdown
    confidence_distribution: list[float] = field(default_factory=list)
    corroboration_distribution: list[int] = field(default_factory=list)
    # Cost
    cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(all_metrics: list[RunMetrics]) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    by_variant: dict[str, list[RunMetrics]] = {}
    for m in all_metrics:
        by_variant.setdefault(m.variant, []).append(m)
    variants = list(by_variant.keys())

    p("\n" + "=" * 100)
    p("EXPERIMENT 3.20 — CONFIDENCE-WEIGHTED CORROBORATION — RESULTS")
    p("=" * 100)

    header = f"{'Metric':<40}"
    for v in variants:
        header += f"{v:>28}"
    p(header)
    p("-" * (40 + 28 * len(variants)))

    def row(label, getter, fmt=".3f"):
        line = f"{label:<40}"
        for v in variants:
            valid = [m for m in by_variant[v] if m.success]
            if valid:
                avg = statistics.mean([getter(m) for m in valid])
                line += f"{avg:>28{fmt}}"
            else:
                line += f"{'FAILED':>28}"
        p(line)

    row("Groundedness score",       lambda m: m.groundedness_score)
    row("Total evidence entries",   lambda m: m.total_evidence, fmt=".0f")
    row("Over-confident claims",    lambda m: m.over_confident, fmt=".1f")
    row("Well-corroborated claims", lambda m: m.well_corroborated, fmt=".1f")
    row("Under-confident claims",   lambda m: m.under_confident, fmt=".1f")
    row("Avg confidence",           lambda m: m.avg_confidence)
    row("Avg corroboration depth",  lambda m: m.avg_corroboration)
    row("Calibration score",        lambda m: m.calibration_score)
    row("Cost (USD)",               lambda m: m.cost_usd, fmt=".4f")
    row("Attempts",                 lambda m: m.attempts, fmt=".1f")

    p("-" * (40 + 28 * len(variants)))

    # Per-persona detail
    p("\n-- PER-PERSONA DETAIL --")
    for m in all_metrics:
        if not m.success:
            continue
        p(f"\n  [{m.variant}] {m.persona_name} (cluster {m.cluster_id[:12]})")
        p(f"    Evidence: {m.total_evidence} entries")
        p(f"    Over-confident: {m.over_confident} | "
          f"Well-corroborated: {m.well_corroborated} | "
          f"Under-confident: {m.under_confident}")
        p(f"    Calibration: {m.calibration_score:.3f}")

        # Confidence histogram
        if m.confidence_distribution:
            buckets = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}
            for c in m.confidence_distribution:
                if c < 0.5:
                    buckets["0.0-0.5"] += 1
                elif c < 0.7:
                    buckets["0.5-0.7"] += 1
                elif c < 0.8:
                    buckets["0.7-0.8"] += 1
                elif c < 0.9:
                    buckets["0.8-0.9"] += 1
                else:
                    buckets["0.9-1.0"] += 1
            p(f"    Confidence dist: {buckets}")

        # Corroboration histogram
        if m.corroboration_distribution:
            depth_counts: dict[int, int] = {}
            for d in m.corroboration_distribution:
                depth_counts[d] = depth_counts.get(d, 0) + 1
            p(f"    Corroboration dist: {dict(sorted(depth_counts.items()))}")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    all_valid = [m for m in all_metrics if m.success]
    if all_valid:
        avg_cal = statistics.mean([m.calibration_score for m in all_valid])
        total_over = sum(m.over_confident for m in all_valid)
        total_ev = sum(m.total_evidence for m in all_valid)
        over_rate = total_over / total_ev if total_ev > 0 else 0

        p(f"\n  Avg calibration score: {avg_cal:.3f}")
        p(f"  Over-confident rate:  {over_rate:.1%} ({total_over}/{total_ev})")

        if over_rate > 0.20:
            strength = "STRONG FINDING"
            detail = (
                f"{over_rate:.0%} of claims are over-confident (high confidence, "
                f"single record). The corroboration floor would catch these."
            )
        elif over_rate > 0.05:
            strength = "MODERATE FINDING"
            detail = (
                f"{over_rate:.0%} over-confident claims detected. The corroboration "
                f"check provides a useful quality signal."
            )
        elif total_over > 0:
            strength = "WEAK FINDING"
            detail = (
                f"Only {total_over} over-confident claims found. The synthesizer "
                f"is mostly well-calibrated but the check catches edge cases."
            )
        else:
            strength = "NULL RESULT"
            detail = (
                "No over-confident claims detected. The synthesizer already "
                "calibrates confidence to corroboration depth well."
            )

        if avg_cal < 0.5:
            strength = "STRONG FINDING"
            detail += f" Calibration score {avg_cal:.3f} is poor — confidence " \
                      f"does not track corroboration depth."
        elif avg_cal < 0.7:
            detail += f" Calibration {avg_cal:.3f} is moderate."

        p(f"\n  Signal: {strength}")
        p(f"  {detail}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 3.20: Confidence-weighted corroboration")
    print("Hypothesis: Synthesizer assigns high confidence to thinly-")
    print("  corroborated claims; cross-checking catches these")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/3] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    all_metrics: list[RunMetrics] = []

    # Run synthesis and analyze corroboration
    print("\n[2/3] Synthesizing and analyzing corroboration...")
    for cluster in clusters:
        print(f"\n  Cluster: {cluster.cluster_id}")
        t0 = time.monotonic()

        # Standard synthesis (control — no enforcement)
        m_ctrl = RunMetrics(variant="control", cluster_id=cluster.cluster_id)
        try:
            result = await synthesize(cluster, synth_backend)
            persona = result.persona
            pd = persona.model_dump(mode="json")

            # Standard groundedness (no corroboration enforcement)
            ground = check_groundedness(persona, cluster, enforce_corroboration=False)
            corr = ground.corroboration

            m_ctrl.success = True
            m_ctrl.persona_name = persona.name
            m_ctrl.groundedness_score = ground.score
            m_ctrl.total_evidence = corr.total_evidence
            m_ctrl.over_confident = corr.over_confident_count
            m_ctrl.well_corroborated = corr.well_corroborated_count
            m_ctrl.under_confident = corr.under_confident_count
            m_ctrl.avg_confidence = corr.avg_confidence
            m_ctrl.avg_corroboration = corr.avg_corroboration
            m_ctrl.calibration_score = corr.calibration_score
            m_ctrl.cost_usd = result.total_cost_usd
            m_ctrl.attempts = result.attempts

            # Distributions for detailed analysis
            for ev in persona.source_evidence:
                m_ctrl.confidence_distribution.append(ev.confidence)
                m_ctrl.corroboration_distribution.append(ev.corroboration_depth)

            print(f"    control: {persona.name} | "
                  f"over-confident={corr.over_confident_count}, "
                  f"calibration={corr.calibration_score:.3f}, "
                  f"cost=${result.total_cost_usd:.4f}")

            # Enforced variant — same persona, stricter check
            m_enf = RunMetrics(variant="enforced", cluster_id=cluster.cluster_id)
            ground_enf = check_groundedness(persona, cluster, enforce_corroboration=True)
            corr_enf = ground_enf.corroboration

            m_enf.success = True
            m_enf.persona_name = persona.name
            m_enf.groundedness_score = ground_enf.score
            m_enf.total_evidence = corr_enf.total_evidence
            m_enf.over_confident = corr_enf.over_confident_count
            m_enf.well_corroborated = corr_enf.well_corroborated_count
            m_enf.under_confident = corr_enf.under_confident_count
            m_enf.avg_confidence = corr_enf.avg_confidence
            m_enf.avg_corroboration = corr_enf.avg_corroboration
            m_enf.calibration_score = corr_enf.calibration_score
            m_enf.cost_usd = result.total_cost_usd  # same persona, no extra cost
            m_enf.attempts = result.attempts
            m_enf.confidence_distribution = list(m_ctrl.confidence_distribution)
            m_enf.corroboration_distribution = list(m_ctrl.corroboration_distribution)

            passed_str = "PASS" if ground_enf.passed else "FAIL"
            print(f"    enforced: groundedness={ground_enf.score:.2f} ({passed_str}), "
                  f"violations={len(ground_enf.violations)}")

        except Exception as e:
            print(f"    FAILED: {e}")

        m_ctrl.duration_seconds = time.monotonic() - t0
        m_enf.duration_seconds = m_ctrl.duration_seconds
        all_metrics.append(m_ctrl)
        all_metrics.append(m_enf)

    # Report
    print("\n[3/3] Generating report...")
    report = print_results(all_metrics)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "3.20",
        "title": "Confidence-weighted corroboration",
        "hypothesis": "Synthesizer assigns high confidence to thinly-corroborated claims",
        "model": settings.default_model,
        "metrics": [
            {
                "variant": m.variant,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "success": m.success,
                "groundedness_score": m.groundedness_score,
                "total_evidence": m.total_evidence,
                "over_confident": m.over_confident,
                "well_corroborated": m.well_corroborated,
                "under_confident": m.under_confident,
                "avg_confidence": m.avg_confidence,
                "avg_corroboration": m.avg_corroboration,
                "calibration_score": m.calibration_score,
                "confidence_distribution": m.confidence_distribution,
                "corroboration_distribution": m.corroboration_distribution,
                "cost_usd": m.cost_usd,
                "attempts": m.attempts,
            }
            for m in all_metrics
        ],
    }

    results_path = output_dir / "exp_3_20_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_3_20_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
