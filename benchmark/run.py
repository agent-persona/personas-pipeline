"""Benchmark runner — executes the uniform benchmark against the current branch.

For each tenant:
  1. Load deterministic records
  2. Run segmentation
  3. Synthesize a persona for each cluster (async, parallel)
  4. Record all metrics

Output: one JSON file per tenant under --output dir, plus a summary.json.

Usage:
  python benchmark/run.py --output benchmark/results/main/
  python benchmark/run.py --tenant bench_dense_saas --output /tmp/bench/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import SynthesisError, synthesize
from synthesis.models.cluster import ClusterData

sys.path.insert(0, str(REPO_ROOT / "benchmark"))
from tenants import load_tenant, tenant_hash, TENANTS

# Concurrency limit per tenant (stay under Anthropic rate limits)
CLUSTER_CONCURRENCY = 3
# Concurrency limit across tenants within a run
TENANT_CONCURRENCY = 2


@dataclass
class ClusterResult:
    cluster_id: str
    n_records: int
    record_ids: list[str]
    # Synthesis
    persona_name: str = ""
    persona_json: dict = field(default_factory=dict)
    groundedness: float = 0.0
    attempts: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    failed: bool = False
    error: str = ""


@dataclass
class TenantResult:
    tenant_id: str
    tenant_hash: str
    industry: str
    product: str
    # Ingest/segment
    n_records: int = 0
    n_clusters: int = 0
    segment_latency_s: float = 0.0
    # Synthesis aggregates
    clusters: list[ClusterResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    mean_groundedness: float = 0.0
    success_rate: float = 0.0
    # Error if the whole tenant blew up
    error: str = ""


async def _synth_one(
    cluster: ClusterData,
    backend: AnthropicBackend,
    semaphore: asyncio.Semaphore,
) -> ClusterResult:
    result = ClusterResult(
        cluster_id=cluster.cluster_id,
        n_records=len(cluster.sample_records),
        record_ids=cluster.all_record_ids,
    )
    async with semaphore:
        t0 = time.monotonic()
        # Retry on rate-limit errors
        max_retries = 4
        for attempt in range(max_retries):
            try:
                r = await synthesize(cluster, backend)
                result.persona_name = r.persona.name
                result.persona_json = r.persona.model_dump(mode="json")
                result.groundedness = r.groundedness.score
                result.attempts = r.attempts
                result.cost_usd = r.total_cost_usd
                break
            except SynthesisError as e:
                is_rate = any("429" in str(a.validation_errors) + str(a.groundedness_violations)
                              or "rate_limit" in (str(a.validation_errors) + str(a.groundedness_violations)).lower()
                              for a in e.attempts)
                if is_rate and attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                result.failed = True
                result.error = str(e)[:200]
                result.attempts = len(e.attempts)
                result.cost_usd = sum(a.cost_usd for a in e.attempts)
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate = "429" in err_str or "rate_limit" in err_str or "overloaded" in err_str
                is_transient = (
                    "apiconnectionerror" in type(e).__name__.lower()
                    or "connection error" in err_str
                    or "connection reset" in err_str
                    or "timeout" in err_str
                )
                if (is_rate or is_transient) and attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                result.failed = True
                result.error = f"{type(e).__name__}: {e}"[:200]
                break
        result.latency_s = time.monotonic() - t0
    return result


async def run_tenant(
    tenant_id: str,
    client: AsyncAnthropic,
    model: str,
) -> TenantResult:
    _, records, meta = load_tenant(tenant_id)
    th = tenant_hash(tenant_id)
    result = TenantResult(
        tenant_id=tenant_id,
        tenant_hash=th,
        industry=meta.get("industry", ""),
        product=meta.get("product", ""),
        n_records=len(records),
    )

    try:
        # Segment
        raw = [RawRecord.model_validate(r.model_dump()) for r in records]
        t0 = time.monotonic()
        clusters_raw = segment(
            raw,
            tenant_industry=meta.get("industry"),
            tenant_product=meta.get("product"),
            existing_persona_names=[],
            similarity_threshold=0.15,
            min_cluster_size=2,
        )
        clusters = [ClusterData.model_validate(c) for c in clusters_raw]
        result.segment_latency_s = time.monotonic() - t0
        result.n_clusters = len(clusters)

        if not clusters:
            result.error = "no clusters formed"
            return result

        # Synthesize all clusters concurrently
        backend = AnthropicBackend(client=client, model=model)
        semaphore = asyncio.Semaphore(CLUSTER_CONCURRENCY)
        t0 = time.monotonic()
        cluster_results = await asyncio.gather(*[
            _synth_one(c, backend, semaphore) for c in clusters
        ])
        result.total_latency_s = time.monotonic() - t0
        result.clusters = cluster_results

        # Aggregates
        result.total_cost_usd = sum(c.cost_usd for c in cluster_results)
        ok = [c for c in cluster_results if not c.failed]
        result.success_rate = len(ok) / len(cluster_results) if cluster_results else 0
        result.mean_groundedness = (
            sum(c.groundedness for c in ok) / len(ok) if ok else 0.0
        )
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"

    return result


async def main(output_dir: Path, tenant_names: list[str], model: str | None = None) -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env", file=sys.stderr)
        sys.exit(1)

    model = model or settings.default_model
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {model}")
    print(f"Output:   {output_dir}")
    print(f"Tenants:  {', '.join(tenant_names)}")

    t_start = time.monotonic()

    # Run tenants concurrently too (bounded)
    tenant_sem = asyncio.Semaphore(TENANT_CONCURRENCY)

    async def run_with_sem(name: str) -> TenantResult:
        async with tenant_sem:
            print(f"  [{name}] starting...")
            r = await run_tenant(name, client, model)
            ok_count = sum(1 for c in r.clusters if not c.failed)
            print(
                f"  [{name}] done: {ok_count}/{len(r.clusters)} ok, "
                f"g={r.mean_groundedness:.2f}, ${r.total_cost_usd:.4f}, "
                f"{r.total_latency_s:.1f}s"
            )
            # Write result immediately
            out_path = output_dir / f"{name}.json"
            out_path.write_text(json.dumps(asdict(r), indent=2, default=str))
            return r

    results = await asyncio.gather(*[run_with_sem(n) for n in tenant_names])

    # Summary
    total_elapsed = time.monotonic() - t_start
    summary = {
        "model": model,
        "tenants": tenant_names,
        "tenant_hashes": {n: tenant_hash(n) for n in tenant_names},
        "n_personas": sum(len(r.clusters) for r in results),
        "n_personas_ok": sum(sum(1 for c in r.clusters if not c.failed) for r in results),
        "total_cost_usd": sum(r.total_cost_usd for r in results),
        "mean_groundedness": (
            sum(r.mean_groundedness for r in results) / len(results) if results else 0
        ),
        "wall_clock_s": total_elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"Done in {total_elapsed:.1f}s")
    print(f"  Personas: {summary['n_personas_ok']}/{summary['n_personas']}")
    print(f"  Mean groundedness: {summary['mean_groundedness']:.2f}")
    print(f"  Total cost: ${summary['total_cost_usd']:.4f}")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tenant", action="append", help="Tenant name (can be repeated). Default: all.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    tenant_names = args.tenant or list(TENANTS.keys())
    asyncio.run(main(args.output, tenant_names, args.model))
