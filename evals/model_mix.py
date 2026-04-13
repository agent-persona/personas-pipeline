"""Experiment 2.03 — Model-mix tiering.

Test different model combinations for synthesis: all-Haiku, all-Sonnet,
and Haiku-draft/Sonnet-revise. Compare quality-per-dollar.

Usage:
    python evals/model_mix.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from crawler import fetch_all
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import SynthesisError, synthesize
from synthesis.models.cluster import ClusterData

# Configs to test: (name, model, max_retries)
# Since only Haiku is available on this API key, we test the cost-quality
# tradeoff by varying retry budget (more retries = higher cost, better quality)
CONFIGS = [
    ("haiku-0retry", "claude-haiku-4-5-20251001", 0),
    ("haiku-2retry", "claude-haiku-4-5-20251001", 2),
    ("haiku-5retry", "claude-haiku-4-5-20251001", 5),
]


async def run_config(name: str, model: str, max_retries: int,
                     clusters: list[ClusterData], client: AsyncAnthropic):
    backend = AnthropicBackend(client=client, model=model)
    results = []
    total_cost = 0.0
    t0 = time.monotonic()
    for cluster in clusters:
        print(f"  [{name}] {cluster.cluster_id}...", end="", flush=True)
        try:
            result = await synthesize(cluster, backend, max_retries=max_retries)
            total_cost += result.total_cost_usd
            results.append({
                "cluster_id": cluster.cluster_id,
                "persona": result.persona.name,
                "groundedness": result.groundedness.score,
                "attempts": result.attempts,
                "cost": result.total_cost_usd,
                "goals": result.persona.goals[:3],
                "failed": False,
            })
            print(f" {result.persona.name} (g={result.groundedness.score:.2f}, att={result.attempts}, ${result.total_cost_usd:.4f})")
        except SynthesisError as e:
            cost = sum(a.cost_usd for a in e.attempts)
            total_cost += cost
            results.append({
                "cluster_id": cluster.cluster_id, "persona": "FAILED",
                "groundedness": 0, "attempts": len(e.attempts),
                "cost": cost, "goals": [], "failed": True,
            })
            print(f" FAILED (att={len(e.attempts)}, ${cost:.4f})")
    elapsed = time.monotonic() - t0
    return {"config": name, "model": model, "max_retries": max_retries,
            "results": results, "total_cost": total_cost, "elapsed_s": elapsed,
            "success_rate": sum(1 for r in results if not r["failed"]) / len(results) if results else 0}


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    records = fetch_all("tenant_acme_corp")
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]
    clusters_raw = segment(raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool", existing_persona_names=[],
        similarity_threshold=0.15, min_cluster_size=2)
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    print(f"Clusters: {len(clusters)}\n")

    all_configs = []
    for name, model, max_retries in CONFIGS:
        print(f"--- Config: {name} (retries={max_retries}) ---")
        result = await run_config(name, model, max_retries, clusters, client)
        all_configs.append(result)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2.03 -- MODEL-MIX TIERING (retry budget sweep)")
    print("=" * 80)
    print(f"\n{'Config':<14} {'Retries':>7} {'Success':>7} {'Avg G':>6} {'Cost':>8} {'Time':>6}")
    print("-" * 55)
    for c in all_configs:
        ok = [r for r in c["results"] if not r["failed"]]
        avg_g = sum(r["groundedness"] for r in ok) / len(ok) if ok else 0
        print(f"{c['config']:<14} {c['max_retries']:>7} {c['success_rate']:>6.0%} {avg_g:>6.2f} "
              f"${c['total_cost']:>7.4f} {c['elapsed_s']:>5.1f}s")

    # Quality per dollar
    print(f"\n{'Config':<14} {'Quality/Dollar':>14}  (groundedness / cost)")
    print("-" * 35)
    for c in all_configs:
        ok = [r for r in c["results"] if not r["failed"]]
        avg_g = sum(r["groundedness"] for r in ok) / len(ok) if ok else 0
        qpd = avg_g / c["total_cost"] if c["total_cost"] > 0 else 0
        print(f"{c['config']:<14} {qpd:>14.1f}")

    out = REPO_ROOT / "output" / "exp_2_03_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_configs, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
