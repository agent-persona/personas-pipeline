"""Shared baseline control run for all experiments.

Runs the default pipeline (ingest → segment → synthesize) with no
modifications, scores the output with evaluation/metrics.py shared metrics,
and saves results to output/experiments/baseline/.

Every experiment references this baseline for its control condition.

Usage:
    python scripts/run_baseline_control.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402
from evaluation.metrics import schema_validity, groundedness_rate, cost_per_persona  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "baseline"

TWIN_PROBE = "What's the single biggest frustration you have with your current tools?"


async def main():
    print("=" * 72)
    print("SHARED BASELINE CONTROL RUN")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)
    model = settings.default_model

    # ---- 1. Ingest ----
    print("\n[1/4] Ingest...")
    raw_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]
    print(f"  {len(records)} records")

    # ---- 2. Segment ----
    print("\n[2/4] Segment...")
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = sorted(
        [ClusterData.model_validate(c) for c in cluster_dicts],
        key=lambda c: c.cluster_id,
    )
    print(f"  {len(clusters)} clusters: {[c.cluster_id for c in clusters]}")

    # ---- 3. Synthesize ----
    print("\n[3/4] Synthesize (default config, no modifications)...")
    persona_results = []
    groundedness_reports = []
    total_cost = 0.0

    for i, cluster in enumerate(clusters):
        print(f"  [{i+1}/{len(clusters)}] {cluster.cluster_id} ({len(cluster.sample_records)} records)...")
        try:
            r = await synthesize(cluster, backend)
            p_dict = r.persona.model_dump(mode="json")
            persona_results.append({
                "cluster_id": cluster.cluster_id,
                "n_records": len(cluster.sample_records),
                "status": "ok",
                "persona": p_dict,
                "cost_usd": r.total_cost_usd,
                "groundedness_score": r.groundedness.score,
                "attempts": r.attempts,
            })
            groundedness_reports.append(r.groundedness)
            total_cost += r.total_cost_usd
            print(f"    [OK] {p_dict['name']}  cost=${r.total_cost_usd:.4f}  "
                  f"grounded={r.groundedness.score:.2f}  attempts={r.attempts}")
        except SynthesisError as e:
            c = sum(a.cost_usd for a in e.attempts)
            total_cost += c
            persona_results.append({
                "cluster_id": cluster.cluster_id,
                "n_records": len(cluster.sample_records),
                "status": "failed",
                "error": str(e),
                "cost_usd": c,
            })
            print(f"    [FAIL] {e}")

    ok_results = [r for r in persona_results if r["status"] == "ok"]
    personas = [r["persona"] for r in ok_results]

    # ---- 4. Twin probe ----
    print(f"\n[4/4] Twin probe: \"{TWIN_PROBE}\"")
    twin_replies = []
    for p in personas:
        twin = TwinChat(p, client=client, model=model)
        reply = await twin.reply(TWIN_PROBE)
        twin_replies.append({
            "persona_name": p["name"],
            "reply": reply.text,
            "cost_usd": reply.estimated_cost_usd,
        })
        total_cost += reply.estimated_cost_usd
        print(f"  [{p['name']}] {reply.text[:100]}...")

    # ---- Shared metrics ----
    print("\n  Computing shared metrics...")
    sv = schema_validity(personas, PersonaV1)
    gr = groundedness_rate(groundedness_reports) if groundedness_reports else 0.0
    cpp = cost_per_persona(total_cost, len(personas))

    print(f"    schema_validity:    {sv:.2f}")
    print(f"    groundedness_rate:  {gr:.2f}")
    print(f"    cost_per_persona:   ${cpp:.4f}")

    # ---- Write ----
    summary = {
        "run_type": "baseline_control",
        "tenant_id": TENANT_ID,
        "model": model,
        "n_clusters": len(clusters),
        "n_personas_ok": len(ok_results),
        "n_personas_failed": len(persona_results) - len(ok_results),
        "shared_metrics": {
            "schema_validity": sv,
            "groundedness_rate": gr,
            "cost_per_persona_usd": cpp,
        },
        "total_cost_usd": total_cost,
        "persona_names": [p["name"] for p in personas],
        "cluster_sizes": [len(c.sample_records) for c in clusters],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "personas.json").write_text(json.dumps(ok_results, indent=2, default=str))
    (OUTPUT_DIR / "twin_replies.json").write_text(json.dumps(twin_replies, indent=2))

    print("\n" + "=" * 72)
    print("BASELINE SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
