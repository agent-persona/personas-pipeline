"""Experiment 2.02: Critique and reflexion loops.

Hypothesis: 1 reflexion round is the sweet spot. 0 rounds under-optimizes,
3+ rounds overfits to the critic and kills distinctiveness.

Variants:
  - 0 rounds (control): standard single-pass synthesis
  - 1 round:  synthesize -> critic -> re-synthesize with feedback
  - 2 rounds: two critic-revision cycles
  - 3 rounds: three critic-revision cycles

Metrics:
  - Convergence speed: how quickly do critic scores plateau?
  - Distinctiveness collapse: does distinctiveness drop at high round counts?
  - Cost-per-quality-unit: is the improvement worth the LLM spend?

Usage:
    python scripts/experiment_2_02.py
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
from synthesis.engine.critique import reflexion_synthesize, CritiqueScore  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

REVISION_VARIANTS = [0, 1, 2, 3]
DIMENSIONS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")


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
    revision_rounds: int
    cluster_id: str = ""
    persona_name: str = ""
    success: bool = False
    # Scores
    initial_overall: float = float("nan")
    initial_distinctive: float = float("nan")
    final_overall: float = float("nan")
    final_distinctive: float = float("nan")
    final_dimensions: dict = field(default_factory=dict)
    groundedness: float = 0.0
    # Convergence
    score_trajectory: list[float] = field(default_factory=list)
    distinctive_trajectory: list[float] = field(default_factory=list)
    # Cost
    total_cost_usd: float = 0.0
    cost_per_quality: float = 0.0  # cost / final_overall
    duration_seconds: float = 0.0


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(all_metrics: list[RunMetrics]) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    by_rounds: dict[int, list[RunMetrics]] = {}
    for m in all_metrics:
        by_rounds.setdefault(m.revision_rounds, []).append(m)

    p("\n" + "=" * 100)
    p("EXPERIMENT 2.02 — CRITIQUE AND REFLEXION LOOPS — RESULTS")
    p("=" * 100)

    header = f"{'Metric':<35}"
    for r in REVISION_VARIANTS:
        label = f"{r} rounds" + (" (control)" if r == 0 else "")
        header += f"{label:>16}"
    p(header)
    p("-" * (35 + 16 * len(REVISION_VARIANTS)))

    def row(label, getter, fmt=".3f", signed=False):
        line = f"{label:<35}"
        for r in REVISION_VARIANTS:
            valid = [m for m in by_rounds.get(r, []) if m.success]
            if valid:
                vals = [getter(m) for m in valid]
                vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
                if vals:
                    avg = statistics.mean(vals)
                    if signed:
                        s = f"{avg:+{fmt}}"
                        line += f"{s:>16}"
                    else:
                        line += f"{avg:>16{fmt}}"
                else:
                    line += f"{'N/A':>16}"
            else:
                line += f"{'FAILED':>16}"
        p(line)

    row("Initial critic overall",   lambda m: m.initial_overall)
    row("Final critic overall",     lambda m: m.final_overall)
    row("Overall delta",            lambda m: (m.final_overall - m.initial_overall) if not math.isnan(m.initial_overall) and not math.isnan(m.final_overall) else float("nan"), signed=True)
    row("Initial distinctive",     lambda m: m.initial_distinctive)
    row("Final distinctive",       lambda m: m.final_distinctive)
    row("Distinctive delta",        lambda m: (m.final_distinctive - m.initial_distinctive) if not math.isnan(m.initial_distinctive) and not math.isnan(m.final_distinctive) else float("nan"), signed=True)
    for dim in DIMENSIONS:
        if dim != "distinctive":
            row(f"  final {dim}",   lambda m, d=dim: m.final_dimensions.get(d, float("nan")))
    row("Groundedness",             lambda m: m.groundedness)
    row("Total cost (USD)",          lambda m: m.total_cost_usd, fmt=".4f")
    row("Cost/quality",              lambda m: m.cost_per_quality, fmt=".4f")

    p("-" * (35 + 16 * len(REVISION_VARIANTS)))

    # Convergence trajectories
    p("\n-- CONVERGENCE TRAJECTORIES --")
    for m in all_metrics:
        if not m.success or not m.score_trajectory:
            continue
        traj = " -> ".join(f"{s:.3f}" for s in m.score_trajectory)
        dtraj = " -> ".join(f"{s:.3f}" for s in m.distinctive_trajectory)
        p(f"  [{m.revision_rounds}R] {m.persona_name[:30]}: overall={traj}")
        p(f"       distinctive={dtraj}")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    # Control has no critic scores (0 rounds skips critique), so use
    # the initial critique from round-1 runs as baseline proxy.
    ctrl = [m for m in by_rounds.get(0, []) if m.success]
    ctrl_cost = statistics.mean([m.total_cost_usd for m in ctrl]) if ctrl else 0.03

    # Use 1-round initial scores as baseline for comparison
    r1 = [m for m in by_rounds.get(1, []) if m.success]
    baseline_overall_vals = [m.initial_overall for m in r1 if not math.isnan(m.initial_overall)]
    baseline_distinct_vals = [m.initial_distinctive for m in r1 if not math.isnan(m.initial_distinctive)]
    baseline_overall = statistics.mean(baseline_overall_vals) if baseline_overall_vals else float("nan")
    baseline_distinct = statistics.mean(baseline_distinct_vals) if baseline_distinct_vals else float("nan")

    if not math.isnan(baseline_overall):
        p(f"\n  Baseline (pre-critique): overall={baseline_overall:.3f}, "
          f"distinctive={baseline_distinct:.3f}")

    for r in [1, 2, 3]:
        valid = [m for m in by_rounds.get(r, []) if m.success]
        if not valid:
            p(f"\n  {r} rounds: ALL FAILED")
            continue

        v_overall = [m.final_overall for m in valid if not math.isnan(m.final_overall)]
        v_distinct = [m.final_distinctive for m in valid if not math.isnan(m.final_distinctive)]
        v_cost = statistics.mean([m.total_cost_usd for m in valid])

        avg_overall = statistics.mean(v_overall) if v_overall else float("nan")
        avg_distinct = statistics.mean(v_distinct) if v_distinct else float("nan")
        cost_mult = v_cost / ctrl_cost if ctrl_cost > 0 else 0

        p(f"\n  {r} round(s):")
        if not math.isnan(avg_overall):
            p(f"    Final overall:     {avg_overall:.3f}")
        if not math.isnan(avg_distinct):
            distinct_sig = "COLLAPSED" if not math.isnan(baseline_distinct) and avg_distinct < baseline_distinct - 0.05 else "HELD" if not math.isnan(baseline_distinct) and avg_distinct >= baseline_distinct - 0.03 else "SLIGHT DIP"
            p(f"    Final distinctive: {avg_distinct:.3f} ({distinct_sig})")
        p(f"    Cost multiplier:   {cost_mult:.1f}x")

    # Sweet spot analysis
    p("\n  Sweet spot analysis:")
    best_r = 0
    best_score = 0.0
    for r in [1, 2, 3]:
        valid = [m for m in by_rounds.get(r, []) if m.success]
        v_overall = [m.final_overall for m in valid if not math.isnan(m.final_overall)]
        if v_overall and statistics.mean(v_overall) > best_score:
            best_score = statistics.mean(v_overall)
            best_r = r
    p(f"    Best overall quality at: {best_r} round(s) ({best_score:.3f})")

    # Distinctiveness collapse check
    r3_distinct = [m.final_distinctive for m in by_rounds.get(3, [])
                   if m.success and not math.isnan(m.final_distinctive)]
    if r3_distinct and not math.isnan(baseline_distinct):
        d3_avg = statistics.mean(r3_distinct)
        collapse = baseline_distinct - d3_avg
        if collapse > 0.05:
            p(f"    DISTINCTIVENESS COLLAPSE at 3 rounds: -{collapse:.3f}")
        else:
            p(f"    No distinctiveness collapse at 3 rounds")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 2.02: Critique and reflexion loops")
    print("Hypothesis: 1 reflexion round is the sweet spot;")
    print("  3+ rounds overfit and kill distinctiveness")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = settings.default_model

    print("\n[1/3] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    all_metrics: list[RunMetrics] = []

    print("\n[2/3] Running synthesis with 0-3 revision rounds...")
    for rounds in REVISION_VARIANTS:
        label = f"{rounds} rounds" + (" (control)" if rounds == 0 else "")
        print(f"\n  -- {label} --")

        for cluster in clusters:
            print(f"    Cluster: {cluster.cluster_id}")
            t0 = time.monotonic()
            m = RunMetrics(revision_rounds=rounds, cluster_id=cluster.cluster_id)

            try:
                result = await reflexion_synthesize(
                    cluster, client, model,
                    revision_rounds=rounds,
                )
                m.success = True
                m.persona_name = result.persona.name
                m.groundedness = result.groundedness.score
                m.total_cost_usd = result.total_cost_usd

                # Initial critique scores
                if result.initial_critique:
                    m.initial_overall = result.initial_critique.overall
                    m.initial_distinctive = result.initial_critique.dimensions.get("distinctive", float("nan"))
                    m.score_trajectory.append(result.initial_critique.overall)
                    m.distinctive_trajectory.append(m.initial_distinctive)

                # Per-round trajectories
                for rr in result.revision_rounds:
                    if rr.pre_critique:
                        m.score_trajectory.append(rr.pre_critique.overall)
                        m.distinctive_trajectory.append(
                            rr.pre_critique.dimensions.get("distinctive", float("nan"))
                        )

                # Final critique scores
                if result.final_critique:
                    m.final_overall = result.final_critique.overall
                    m.final_distinctive = result.final_critique.dimensions.get("distinctive", float("nan"))
                    m.final_dimensions = dict(result.final_critique.dimensions)
                    m.score_trajectory.append(result.final_critique.overall)
                    m.distinctive_trajectory.append(m.final_distinctive)
                elif result.initial_critique:
                    # 0-round: initial = final
                    m.final_overall = result.initial_critique.overall
                    m.final_distinctive = result.initial_critique.dimensions.get("distinctive", float("nan"))
                    m.final_dimensions = dict(result.initial_critique.dimensions)

                m.cost_per_quality = (
                    m.total_cost_usd / m.final_overall
                    if not math.isnan(m.final_overall) and m.final_overall > 0
                    else float("nan")
                )

                final_str = f"final={m.final_overall:.3f}" if not math.isnan(m.final_overall) else "final=N/A"
                dist_str = f"dist={m.final_distinctive:.3f}" if not math.isnan(m.final_distinctive) else "dist=N/A"
                print(f"      {m.persona_name}: {final_str}, {dist_str}, "
                      f"cost=${m.total_cost_usd:.4f}")

            except Exception as e:
                print(f"      FAILED: {e}")

            m.duration_seconds = time.monotonic() - t0
            all_metrics.append(m)

    print("\n[3/3] Comparing results...")
    report = print_results(all_metrics)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    results_data = {
        "experiment": "2.02",
        "title": "Critique and reflexion loops",
        "hypothesis": "1 reflexion round is the sweet spot; 3+ kills distinctiveness",
        "model": model,
        "metrics": [
            {
                "revision_rounds": m.revision_rounds,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "success": m.success,
                "initial_overall": safe(m.initial_overall),
                "initial_distinctive": safe(m.initial_distinctive),
                "final_overall": safe(m.final_overall),
                "final_distinctive": safe(m.final_distinctive),
                "final_dimensions": {k: safe(v) for k, v in m.final_dimensions.items()},
                "groundedness": m.groundedness,
                "score_trajectory": [safe(s) for s in m.score_trajectory],
                "distinctive_trajectory": [safe(s) for s in m.distinctive_trajectory],
                "total_cost_usd": m.total_cost_usd,
                "cost_per_quality": safe(m.cost_per_quality),
            }
            for m in all_metrics
        ],
    }

    results_path = output_dir / "exp_2_02_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_2_02_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
