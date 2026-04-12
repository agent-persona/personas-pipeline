"""End-to-end pipeline: crawler -> segment -> synthesize -> twin chat.

Demonstrates the entire personas framework wired together through the
orchestration DAG runner. Uses Haiku to keep costs minimal.

Usage:
    python scripts/run_full_pipeline.py
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

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from orchestration import Pipeline, Stage  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output"

# exp-2.05: sweep N few-shot exemplars per cluster. N=0 is the experimental
# control (byte-equal to pre-experiment SYSTEM_PROMPT) and is run in the same
# session as N>0 so segmentation/cluster stochasticity is held constant.
EXEMPLAR_SWEEP = [0, 1, 3, 5]


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# ============================================================================
# Stage definitions — each one matches the orchestration Stage signature
# ============================================================================


def stage_ingest(_: None) -> list[RawRecord]:
    """Pull behavioral records from all configured connectors."""
    crawler_records = fetch_all(TENANT_ID)
    # Crawler.Record and segmentation.RawRecord have the same shape — round-trip
    # through dict to convert without an inter-package import.
    return [RawRecord.model_validate(r.model_dump()) for r in crawler_records]


def stage_segment(records: list[RawRecord]) -> list[ClusterData]:
    """Cluster records into behavioral segments."""
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(c) for c in cluster_dicts]


async def stage_synthesize(clusters: list[ClusterData]) -> list[dict]:
    """Synthesize one persona per cluster at each N in the exemplar sweep."""
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Sort for run-to-run ordering stability; cluster_id may differ across
    # runs but ordering inside a run is determined by this line.
    clusters = sorted(clusters, key=lambda c: c.cluster_id)

    personas: list[dict] = []
    total_cost = 0.0
    cost_by_n: dict[int, float] = {n: 0.0 for n in EXEMPLAR_SWEEP}

    for i, cluster in enumerate(clusters):
        print(
            f"  [{i + 1}/{len(clusters)}] synthesizing {cluster.cluster_id} "
            f"across N in {EXEMPLAR_SWEEP}..."
        )
        sweep: list[dict] = []
        for n in EXEMPLAR_SWEEP:
            try:
                result = await synthesize(cluster, backend, few_shot_count=n)
            except SynthesisError as exc:
                # Record the failure — it's a data point, not a crash.
                failed_cost = sum(a.cost_usd for a in exc.attempts)
                sweep.append({
                    "n_exemplars": n,
                    "persona": None,
                    "cost_usd": failed_cost,
                    "groundedness": 0.0,
                    "groundedness_violations": ["SYNTHESIS_FAILED: " + str(exc)],
                    "attempts": len(exc.attempts),
                    "failed": True,
                })
                total_cost += failed_cost
                cost_by_n[n] += failed_cost
                print(f"      [FAIL] N={n}: ${failed_cost:.4f}, "
                      f"attempts={len(exc.attempts)}: {exc}")
                continue
            sweep.append({
                "n_exemplars": n,
                "persona": result.persona.model_dump(mode="json"),
                "cost_usd": result.total_cost_usd,
                "groundedness": result.groundedness.score,
                "groundedness_violations": result.groundedness.violations,
                "attempts": result.attempts,
            })
            total_cost += result.total_cost_usd
            cost_by_n[n] += result.total_cost_usd
            print(
                f"      [OK] N={n}: ${result.total_cost_usd:.4f}, "
                f"groundedness={result.groundedness.score:.2f}, "
                f"attempts={result.attempts}"
            )

        # Top-level shape preserved from the N=0 slice for backward compat.
        # Use the first successful sweep entry; prefer N=0 but fall back.
        baseline = sweep[0]  # N=0 is always index 0 since EXEMPLAR_SWEEP starts with 0
        if baseline.get("failed"):
            baseline = next((s for s in sweep if not s.get("failed")), baseline)
        personas.append({
            "cluster_id": cluster.cluster_id,
            "persona": baseline["persona"],
            "cost_usd": baseline["cost_usd"],
            "groundedness": baseline["groundedness"],
            "attempts": baseline["attempts"],
            "exemplar_sweep": sweep,
        })

    print(f"\n  Synthesis total: ${total_cost:.4f}")
    print("  Cost by N:")
    for n in EXEMPLAR_SWEEP:
        print(f"    N={n}: ${cost_by_n[n]:.4f}")
    return personas


async def stage_twin_chat(personas: list[dict]) -> list[dict]:
    """Demo: ask each twin one question against the N=0 persona only.

    For exp-2.05, twin chat is held constant at the N=0 baseline persona so
    the sweep cost is bounded to the synthesis stage. Downstream contracts
    (twin_demo_reply, twin_demo_cost on the top-level entry) are preserved.
    """
    if not settings.anthropic_api_key:
        return personas

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    question = "What's the single biggest frustration you have with your current tools?"

    print(f"\n  Q: {question}  (N=0 persona only)\n")
    for entry in personas:
        if entry["persona"] is None:
            print(f"  --- {entry['cluster_id']} --- SKIPPED (no persona)")
            continue
        twin = TwinChat(entry["persona"], client=client, model=settings.default_model)
        reply = await twin.reply(question)
        entry["twin_demo_reply"] = reply.text
        entry["twin_demo_cost"] = reply.estimated_cost_usd
        print(f"  --- {entry['persona']['name']} ---")
        print(f"  {reply.text}")
        print(f"  (${reply.estimated_cost_usd:.4f})\n")

    return personas


def stage_persist(personas: list[dict]) -> list[dict]:
    """Save personas to disk (proxy for Postgres + Pinecone)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    for i, entry in enumerate(personas):
        path = OUTPUT_DIR / f"persona_{i:02d}.json"
        path.write_text(json.dumps(entry, indent=2, default=str))
    print(f"  Wrote {len(personas)} personas to {OUTPUT_DIR}/")
    return personas


# ============================================================================
# Build and run the pipeline
# ============================================================================


async def main():
    banner("PERSONAS FRAMEWORK — END-TO-END PIPELINE")
    print(f"  Tenant:  {TENANT_ID}")
    print(f"  Model:   {settings.default_model}")
    print(f"  Output:  {OUTPUT_DIR}")

    pipeline = Pipeline([
        Stage(name="ingest", fn=stage_ingest, description="Pull from connectors"),
        Stage(name="segment", fn=stage_segment, description="Cluster by behavior"),
        Stage(name="synthesize", fn=stage_synthesize, description="Generate personas"),
        Stage(name="twin_chat", fn=stage_twin_chat, description="Demo twin replies"),
        Stage(name="persist", fn=stage_persist, description="Save outputs"),
    ])

    state = await pipeline.run(initial_input=None, tenant_id=TENANT_ID)

    banner("RUN SUMMARY")
    print(f"  Run ID:    {state.run_id}")
    print(f"  Success:   {state.success}")
    print(f"  Duration:  {state.total_duration_ms} ms")
    print(f"\n  Stages:")
    for s in state.stages:
        status = "ok" if s.success else "FAILED"
        print(f"    - {s.name:14s} {status:8s} {s.duration_ms:6d} ms  -> {s.output_summary}")

    print(f"\n  See: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
