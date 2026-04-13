"""Experiment 2.11: Multi-agent debate.

Hypothesis: Adding an adversary that challenges each claim and a judge that
resolves disputes will improve groundedness and reduce hallucination compared
to single-pass synthesis.

Variants:
  - control:  Standard single-pass synthesis
  - debate:   Synthesizer + adversary + judge (three-way loop)

Metrics:
  - Groundedness score (deterministic check)
  - Hallucination count (claims dropped by the judge)
  - Evidence confidence distribution (pre/post debate)
  - Cost overhead of the debate process

Usage:
    python scripts/experiment_2_11.py
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
from synthesis.engine.debate import debate_synthesize, DebateResult  # noqa: E402
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
    evidence_count: int = 0
    avg_confidence: float = 0.0
    low_confidence_count: int = 0  # confidence < 0.7
    # Debate specifics
    total_challenges: int = 0
    claims_kept: int = 0
    claims_revised: int = 0
    claims_dropped: int = 0
    debate_rounds: int = 0
    # Cost
    synthesis_cost_usd: float = 0.0
    debate_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0


def _evidence_stats(persona_dict: dict) -> tuple[int, float, int]:
    """Return (count, avg_confidence, low_confidence_count)."""
    evidence = persona_dict.get("source_evidence", [])
    if not evidence:
        return 0, 0.0, 0
    confs = [e.get("confidence", 0.0) for e in evidence]
    return len(evidence), statistics.mean(confs), sum(1 for c in confs if c < 0.7)


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
    p("EXPERIMENT 2.11 — MULTI-AGENT DEBATE — RESULTS")
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

    row("Groundedness score",        lambda m: m.groundedness_score)
    row("Evidence count",            lambda m: m.evidence_count, fmt=".0f")
    row("Avg evidence confidence",   lambda m: m.avg_confidence)
    row("Low-confidence claims (<0.7)", lambda m: m.low_confidence_count, fmt=".1f")
    row("Challenges raised",        lambda m: m.total_challenges, fmt=".0f")
    row("Claims kept",              lambda m: m.claims_kept, fmt=".0f")
    row("Claims revised",           lambda m: m.claims_revised, fmt=".0f")
    row("Claims dropped",           lambda m: m.claims_dropped, fmt=".0f")
    row("Synthesis cost (USD)",      lambda m: m.synthesis_cost_usd, fmt=".4f")
    row("Debate overhead (USD)",     lambda m: m.debate_cost_usd, fmt=".4f")
    row("Total cost (USD)",          lambda m: m.total_cost_usd, fmt=".4f")
    row("Attempts",                  lambda m: m.attempts, fmt=".1f")
    row("Duration (s)",              lambda m: m.duration_seconds, fmt=".1f")

    p("-" * (40 + 28 * len(variants)))

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    ctrl = [m for m in by_variant.get("control", []) if m.success]
    debate = [m for m in by_variant.get("debate", []) if m.success]

    if ctrl and debate:
        ctrl_ground = statistics.mean([m.groundedness_score for m in ctrl])
        debate_ground = statistics.mean([m.groundedness_score for m in debate])
        d_ground = debate_ground - ctrl_ground

        ctrl_conf = statistics.mean([m.avg_confidence for m in ctrl])
        debate_conf = statistics.mean([m.avg_confidence for m in debate])
        d_conf = debate_conf - ctrl_conf

        ctrl_low = statistics.mean([m.low_confidence_count for m in ctrl])
        debate_low = statistics.mean([m.low_confidence_count for m in debate])

        total_dropped = sum(m.claims_dropped for m in debate)
        total_revised = sum(m.claims_revised for m in debate)
        total_challenges = sum(m.total_challenges for m in debate)

        ctrl_cost = statistics.mean([m.total_cost_usd for m in ctrl])
        debate_cost = statistics.mean([m.total_cost_usd for m in debate])
        cost_mult = debate_cost / ctrl_cost if ctrl_cost > 0 else 0

        p(f"\n  Groundedness lift:     {d_ground:+.4f} "
          f"({'IMPROVED' if d_ground > 0.01 else 'SIMILAR' if d_ground > -0.01 else 'DEGRADED'})")
        p(f"  Confidence shift:     {d_conf:+.4f}")
        p(f"  Low-conf claims:      control={ctrl_low:.1f}, debate={debate_low:.1f}")
        p(f"  Hallucinations caught: {total_dropped} dropped, {total_revised} revised "
          f"out of {total_challenges} challenges")
        p(f"  Cost multiplier:      {cost_mult:.1f}x")

        signals = []
        if d_ground > 0.01:
            signals.append("GROUNDEDNESS_UP")
        if total_dropped > 0:
            signals.append("HALLUCINATIONS_CAUGHT")
        if total_revised > 0:
            signals.append("CLAIMS_REFINED")

        if len(signals) >= 2:
            strength = "STRONG FINDING"
        elif len(signals) == 1:
            strength = "MODERATE FINDING"
        else:
            strength = "WEAK FINDING"

        p(f"\n  Signal: {strength}")
        if total_dropped > 0 or total_revised > 0:
            p(f"  The debate process {'actively improved' if d_ground > 0 else 'refined'} "
              f"the persona by catching {total_dropped + total_revised} problematic claims. "
              f"Cost overhead is {cost_mult:.1f}x — "
              f"{'acceptable' if cost_mult < 3 else 'significant'}.")
        else:
            p(f"  The adversary found no issues to challenge — either the synthesizer "
              f"is already well-calibrated or the adversary threshold is too lenient.")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 2.11: Multi-agent debate")
    print("Hypothesis: Adversary + judge loop improves groundedness")
    print("  and catches hallucinations vs single-pass synthesis")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)
    model = settings.default_model

    print("\n[1/4] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    all_metrics: list[RunMetrics] = []

    # Step 2: Control (single-pass)
    print("\n[2/4] Running control (single-pass synthesis)...")
    for cluster in clusters:
        print(f"  Cluster: {cluster.cluster_id}")
        t0 = time.monotonic()
        metrics = RunMetrics(variant="control", cluster_id=cluster.cluster_id)
        try:
            result = await synthesize(cluster, synth_backend)
            persona_dict = result.persona.model_dump(mode="json")
            ev_count, avg_conf, low_conf = _evidence_stats(persona_dict)

            metrics.success = True
            metrics.persona_name = result.persona.name
            metrics.groundedness_score = result.groundedness.score
            metrics.evidence_count = ev_count
            metrics.avg_confidence = avg_conf
            metrics.low_confidence_count = low_conf
            metrics.synthesis_cost_usd = result.total_cost_usd
            metrics.total_cost_usd = result.total_cost_usd
            metrics.attempts = result.attempts
            print(f"    {result.persona.name}: groundedness={result.groundedness.score:.2f}, "
                  f"evidence={ev_count}, avg_conf={avg_conf:.2f}, cost=${result.total_cost_usd:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
        metrics.duration_seconds = time.monotonic() - t0
        all_metrics.append(metrics)

    # Step 3: Debate (multi-agent)
    print("\n[3/4] Running debate (synthesizer + adversary + judge)...")
    for cluster in clusters:
        print(f"  Cluster: {cluster.cluster_id}")
        t0 = time.monotonic()
        metrics = RunMetrics(variant="debate", cluster_id=cluster.cluster_id)
        try:
            result = await debate_synthesize(
                cluster, client,
                synth_model=model,
                adversary_model=model,
                judge_model=model,
                max_debate_rounds=1,
            )
            persona_dict = result.persona.model_dump(mode="json")
            ev_count, avg_conf, low_conf = _evidence_stats(persona_dict)

            metrics.success = True
            metrics.persona_name = result.persona.name
            metrics.groundedness_score = result.groundedness.score
            metrics.evidence_count = ev_count
            metrics.avg_confidence = avg_conf
            metrics.low_confidence_count = low_conf
            metrics.total_challenges = result.total_challenges
            metrics.claims_kept = result.total_kept
            metrics.claims_revised = result.total_revised
            metrics.claims_dropped = result.total_dropped
            metrics.debate_rounds = len(result.debate_rounds)
            metrics.attempts = result.synthesis_attempts

            # Cost breakdown
            debate_overhead = sum(
                r.adversary_cost_usd + r.judge_cost_usd
                for r in result.debate_rounds
            )
            metrics.synthesis_cost_usd = result.total_cost_usd - debate_overhead
            metrics.debate_cost_usd = debate_overhead
            metrics.total_cost_usd = result.total_cost_usd

            print(f"    {result.persona.name}: groundedness={result.groundedness.score:.2f}, "
                  f"challenges={result.total_challenges}, "
                  f"kept={result.total_kept}/revised={result.total_revised}/"
                  f"dropped={result.total_dropped}, "
                  f"cost=${result.total_cost_usd:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
        metrics.duration_seconds = time.monotonic() - t0
        all_metrics.append(metrics)

    # Step 4: Report
    print("\n[4/4] Comparing results...")
    report = print_results(all_metrics)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "2.11",
        "title": "Multi-agent debate",
        "hypothesis": "Adversary + judge loop improves groundedness and catches hallucinations",
        "model": model,
        "n_clusters": len(clusters),
        "metrics": [
            {
                "variant": m.variant,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "success": m.success,
                "groundedness_score": m.groundedness_score,
                "evidence_count": m.evidence_count,
                "avg_confidence": m.avg_confidence,
                "low_confidence_count": m.low_confidence_count,
                "total_challenges": m.total_challenges,
                "claims_kept": m.claims_kept,
                "claims_revised": m.claims_revised,
                "claims_dropped": m.claims_dropped,
                "synthesis_cost_usd": m.synthesis_cost_usd,
                "debate_cost_usd": m.debate_cost_usd,
                "total_cost_usd": m.total_cost_usd,
                "attempts": m.attempts,
                "duration_seconds": m.duration_seconds,
            }
            for m in all_metrics
        ],
    }

    results_path = output_dir / "exp_2_11_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_2_11_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
