"""Experiment 3.06 — Sparse-data ablation.

Measures how the persona synthesis pipeline degrades as source data
becomes sparser. Runs the full pipeline (ingest -> segment -> synthesize)
at multiple data-density tiers and reports groundedness, schema validity,
cluster count, and cost at each tier.

Two modes:
  --offline   (default) Skip LLM synthesis; measure segmentation & cluster
              health only. No API key needed.
  --online    Run full synthesis with the Anthropic API. Requires
              ANTHROPIC_API_KEY in synthesis/.env.

Usage:
    python evals/sparse_data.py                  # offline mode
    python evals/sparse_data.py --online         # full LLM synthesis
    python evals/sparse_data.py --tiers 200,50,10  # custom tiers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from crawler.connectors.dense_fixture import DenseFixtureConnector, downsample
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.models.cluster import ClusterData

TENANT_ID = "tenant_dense_fixture"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

# Default density tiers (record counts)
DEFAULT_TIERS = [200, 100, 50, 20, 10, 5]


@dataclass
class TierResult:
    tier_name: str
    n_records: int
    n_clusters: int
    records_per_cluster: list[int]
    avg_records_per_cluster: float
    # Online-only fields
    groundedness_scores: list[float] = field(default_factory=list)
    mean_groundedness: float | None = None
    schema_valid: float | None = None
    total_cost_usd: float = 0.0
    synthesis_failures: int = 0
    total_attempts: int = 0
    duration_ms: int = 0


def run_tier_offline(
    records: list[RawRecord],
    tier_name: str,
) -> TierResult:
    """Run segmentation only (no LLM calls) and measure cluster health."""
    clusters_raw = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    records_per = [len(c.sample_records) for c in clusters]
    avg = sum(records_per) / len(records_per) if records_per else 0.0

    return TierResult(
        tier_name=tier_name,
        n_records=len(records),
        n_clusters=len(clusters),
        records_per_cluster=records_per,
        avg_records_per_cluster=avg,
    )


async def run_tier_online(
    records: list[RawRecord],
    tier_name: str,
) -> TierResult:
    """Run full pipeline including LLM synthesis and groundedness checks."""
    from anthropic import AsyncAnthropic
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "synthesis" / ".env")

    from synthesis.config import settings
    from synthesis.engine.model_backend import AnthropicBackend
    from synthesis.engine.synthesizer import SynthesisError, synthesize
    from synthesis.models.persona import PersonaV1
    from evaluation.metrics import groundedness_rate, schema_validity

    clusters_raw = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    records_per = [len(c.sample_records) for c in clusters]
    avg = sum(records_per) / len(records_per) if records_per else 0.0

    if not clusters:
        return TierResult(
            tier_name=tier_name,
            n_records=len(records),
            n_clusters=0,
            records_per_cluster=[],
            avg_records_per_cluster=0.0,
        )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    g_scores: list[float] = []
    persona_dicts: list[dict] = []
    total_cost = 0.0
    failures = 0
    total_attempts = 0

    t0 = time.monotonic()
    for cluster in clusters:
        try:
            result = await synthesize(cluster, backend)
            g_scores.append(result.groundedness.score)
            persona_dicts.append(result.persona.model_dump(mode="json"))
            total_cost += result.total_cost_usd
            total_attempts += result.attempts
        except SynthesisError:
            failures += 1
            total_attempts += 3  # max attempts exhausted

    duration_ms = int((time.monotonic() - t0) * 1000)

    sv = schema_validity(persona_dicts, PersonaV1) if persona_dicts else 0.0
    mg = sum(g_scores) / len(g_scores) if g_scores else 0.0

    return TierResult(
        tier_name=tier_name,
        n_records=len(records),
        n_clusters=len(clusters),
        records_per_cluster=records_per,
        avg_records_per_cluster=avg,
        groundedness_scores=g_scores,
        mean_groundedness=mg,
        schema_valid=sv,
        total_cost_usd=total_cost,
        synthesis_failures=failures,
        total_attempts=total_attempts,
        duration_ms=duration_ms,
    )


def print_results(results: list[TierResult], online: bool) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3.06 — SPARSE-DATA ABLATION RESULTS")
    print("=" * 80)

    # Header
    if online:
        header = f"{'Tier':<14} {'Records':>7} {'Clusters':>8} {'Avg Recs':>8} {'Ground.':>8} {'Valid':>6} {'Cost':>8} {'Fails':>5} {'Att.':>5}"
    else:
        header = f"{'Tier':<14} {'Records':>7} {'Clusters':>8} {'Avg Recs':>8} {'Recs/Cluster':>14}"

    print(header)
    print("-" * len(header))

    for r in results:
        if online:
            mg = f"{r.mean_groundedness:.2f}" if r.mean_groundedness is not None else "N/A"
            sv = f"{r.schema_valid:.2f}" if r.schema_valid is not None else "N/A"
            print(
                f"{r.tier_name:<14} {r.n_records:>7} {r.n_clusters:>8} "
                f"{r.avg_records_per_cluster:>8.1f} {mg:>8} {sv:>6} "
                f"${r.total_cost_usd:>7.4f} {r.synthesis_failures:>5} {r.total_attempts:>5}"
            )
        else:
            rpc = ", ".join(str(x) for x in r.records_per_cluster) if r.records_per_cluster else "—"
            print(
                f"{r.tier_name:<14} {r.n_records:>7} {r.n_clusters:>8} "
                f"{r.avg_records_per_cluster:>8.1f} {rpc:>14}"
            )

    print("-" * len(header))

    # Summary
    if online and results:
        scores = [r.mean_groundedness for r in results if r.mean_groundedness is not None]
        if scores:
            collapse_idx = next(
                (i for i, s in enumerate(scores) if s < 0.9),
                None,
            )
            if collapse_idx is not None:
                print(
                    f"\n>> COLLAPSE POINT: groundedness drops below 0.9 at "
                    f"tier '{results[collapse_idx].tier_name}' "
                    f"({results[collapse_idx].n_records} records, "
                    f"score={scores[collapse_idx]:.2f})"
                )
            else:
                print("\n>> No collapse detected — groundedness >= 0.9 at all tiers.")
    elif not online:
        zero_cluster = [r for r in results if r.n_clusters == 0]
        if zero_cluster:
            print(
                f"\n>> SEGMENTATION COLLAPSE: no clusters formed at "
                f"{', '.join(r.tier_name for r in zero_cluster)}"
            )

    print()


def save_results(results: list[TierResult], output_path: Path) -> None:
    """Save results to JSON for downstream comparison."""
    data = []
    for r in results:
        data.append({
            "tier_name": r.tier_name,
            "n_records": r.n_records,
            "n_clusters": r.n_clusters,
            "records_per_cluster": r.records_per_cluster,
            "avg_records_per_cluster": r.avg_records_per_cluster,
            "groundedness_scores": r.groundedness_scores,
            "mean_groundedness": r.mean_groundedness,
            "schema_valid": r.schema_valid,
            "total_cost_usd": r.total_cost_usd,
            "synthesis_failures": r.synthesis_failures,
            "total_attempts": r.total_attempts,
            "duration_ms": r.duration_ms,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3.06: Sparse-data ablation")
    parser.add_argument("--online", action="store_true", help="Run full LLM synthesis (requires API key)")
    parser.add_argument("--tiers", type=str, default=None, help="Comma-separated record counts (default: 200,100,50,20,10,5)")
    parser.add_argument("--dense-size", type=int, default=200, help="Size of dense baseline fixture")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")] if args.tiers else DEFAULT_TIERS

    print(f"Mode: {'online (LLM synthesis)' if args.online else 'offline (segmentation only)'}")
    print(f"Dense baseline: {args.dense_size} records")
    print(f"Tiers: {tiers}")

    # Generate dense fixture
    connector = DenseFixtureConnector(n_records=args.dense_size)
    all_records_raw = connector.fetch(TENANT_ID)
    all_records = [RawRecord.model_validate(r.model_dump()) for r in all_records_raw]
    print(f"\nGenerated {len(all_records)} dense fixture records.")

    results: list[TierResult] = []
    for n in tiers:
        tier_name = f"{n}_records"
        sampled = [RawRecord.model_validate(r.model_dump()) for r in downsample(all_records_raw, n)]
        print(f"\n--- Tier: {tier_name} ({len(sampled)} records) ---")

        if args.online:
            result = await run_tier_online(sampled, tier_name)
        else:
            result = run_tier_offline(sampled, tier_name)
        results.append(result)

    print_results(results, online=args.online)

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = REPO_ROOT / "output" / "exp_3_06_results.json"
    save_results(results, out_path)


if __name__ == "__main__":
    asyncio.run(main())
