"""Benchmark runner variant that passes distance_metric to segment().

Used only for the feat/adaptive-gower-segmentation branch evaluation.
Identical to run.py except segment() is called with a configurable
distance_metric parameter.
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

CLUSTER_CONCURRENCY = 3
TENANT_CONCURRENCY = 2


@dataclass
class ClusterResult:
    cluster_id: str
    n_records: int
    record_ids: list[str]
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
    distance_metric: str = "jaccard"
    n_records: int = 0
    n_clusters: int = 0
    segment_latency_s: float = 0.0
    clusters: list[ClusterResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    mean_groundedness: float = 0.0
    success_rate: float = 0.0
    error: str = ""


async def _synth_one(cluster, backend, semaphore):
    result = ClusterResult(
        cluster_id=cluster.cluster_id,
        n_records=len(cluster.sample_records),
        record_ids=cluster.all_record_ids,
    )
    async with semaphore:
        t0 = time.monotonic()
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
                is_rate = any("429" in (str(a.validation_errors) + str(a.groundedness_violations)).lower()
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
                if is_rate and attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                result.failed = True
                result.error = f"{type(e).__name__}: {e}"[:200]
                break
        result.latency_s = time.monotonic() - t0
    return result


async def run_tenant(tenant_id, client, model, distance_metric):
    _, records, meta = load_tenant(tenant_id)
    th = tenant_hash(tenant_id)
    result = TenantResult(
        tenant_id=tenant_id, tenant_hash=th,
        industry=meta.get("industry", ""), product=meta.get("product", ""),
        distance_metric=distance_metric,
        n_records=len(records),
    )

    try:
        raw = [RawRecord.model_validate(r.model_dump()) for r in records]
        t0 = time.monotonic()
        clusters_raw = segment(
            raw,
            tenant_industry=meta.get("industry"),
            tenant_product=meta.get("product"),
            existing_persona_names=[],
            similarity_threshold=0.15,
            min_cluster_size=2,
            distance_metric=distance_metric,
        )
        clusters = [ClusterData.model_validate(c) for c in clusters_raw]
        result.segment_latency_s = time.monotonic() - t0
        result.n_clusters = len(clusters)

        if not clusters:
            result.error = "no clusters formed"
            return result

        backend = AnthropicBackend(client=client, model=model)
        semaphore = asyncio.Semaphore(CLUSTER_CONCURRENCY)
        t0 = time.monotonic()
        cluster_results = await asyncio.gather(*[
            _synth_one(c, backend, semaphore) for c in clusters
        ])
        result.total_latency_s = time.monotonic() - t0
        result.clusters = cluster_results

        result.total_cost_usd = sum(c.cost_usd for c in cluster_results)
        ok = [c for c in cluster_results if not c.failed]
        result.success_rate = len(ok) / len(cluster_results) if cluster_results else 0
        result.mean_groundedness = sum(c.groundedness for c in ok) / len(ok) if ok else 0.0
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"

    return result


async def main(output_dir, tenant_names, distance_metric, model=None):
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key", file=sys.stderr)
        sys.exit(1)
    model = model or settings.default_model
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {model}")
    print(f"Metric: {distance_metric}")
    print(f"Output: {output_dir}")

    t_start = time.monotonic()
    tenant_sem = asyncio.Semaphore(TENANT_CONCURRENCY)

    async def run_with_sem(name):
        async with tenant_sem:
            print(f"  [{name}] starting...")
            r = await run_tenant(name, client, model, distance_metric)
            ok = sum(1 for c in r.clusters if not c.failed)
            print(f"  [{name}] done: {ok}/{len(r.clusters)} ok, g={r.mean_groundedness:.2f}, "
                  f"${r.total_cost_usd:.4f}, {r.total_latency_s:.1f}s")
            (output_dir / f"{name}.json").write_text(json.dumps(asdict(r), indent=2, default=str))
            return r

    results = await asyncio.gather(*[run_with_sem(n) for n in tenant_names])

    total_elapsed = time.monotonic() - t_start
    summary = {
        "model": model,
        "distance_metric": distance_metric,
        "tenants": tenant_names,
        "tenant_hashes": {n: tenant_hash(n) for n in tenant_names},
        "n_personas": sum(len(r.clusters) for r in results),
        "n_personas_ok": sum(sum(1 for c in r.clusters if not c.failed) for r in results),
        "total_cost_usd": sum(r.total_cost_usd for r in results),
        "mean_groundedness": sum(r.mean_groundedness for r in results) / len(results) if results else 0,
        "wall_clock_s": total_elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"Done in {total_elapsed:.1f}s ({summary['n_personas_ok']}/{summary['n_personas']} personas, "
          f"${summary['total_cost_usd']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tenant", action="append")
    parser.add_argument("--model", default=None)
    parser.add_argument("--distance-metric", default="jaccard", choices=["jaccard", "gower"])
    args = parser.parse_args()

    tenant_names = args.tenant or list(TENANTS.keys())
    asyncio.run(main(args.output, tenant_names, args.distance_metric, args.model))
