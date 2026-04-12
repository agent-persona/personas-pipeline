"""exp-3.17 — Evidence ablation.

Measures how persona quality degrades as the most informative records
are removed from the cluster.

Creates 5 ablated clusters (remove top 0%, 10%, 25%, 50%, 75% by
informativeness), synthesizes a persona from each, and scores each on
structural groundedness + Claude-as-judge depth.

Hypothesis: removing top 10% causes ≥0.3 groundedness drop and ≥1.0 depth drop.

Usage:
    python scripts/run_exp_3_17.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))

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

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-3.17"

ABLATION_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75]


def record_informativeness(rec) -> float:
    """Score a record by payload richness: key count × total text length."""
    payload = rec.payload or {}
    key_count = len(payload)
    text_length = sum(len(str(v)) for v in payload.values())
    return key_count * text_length


def create_ablated_cluster(cluster: ClusterData, remove_pct: float) -> ClusterData:
    """Remove the top remove_pct% most informative records."""
    records = sorted(cluster.sample_records, key=record_informativeness, reverse=True)
    n_remove = int(len(records) * remove_pct)
    remaining = records[n_remove:]
    if not remaining:
        remaining = records[-1:]  # keep at least 1
    ablated = cluster.model_copy(deep=True)
    ablated.sample_records = remaining
    ablated.summary.cluster_size = len(remaining)
    ablated.cluster_id = f"{cluster.cluster_id}_ablated_{int(remove_pct * 100)}pct"
    return ablated


async def judge_persona_quality(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """Claude-as-judge: rate persona on groundedness and depth."""
    record_summary = []
    for rec in cluster.sample_records[:10]:
        record_summary.append(f"- {rec.record_id}: {json.dumps(rec.payload)}")

    judge_prompt = f"""Rate this persona on two dimensions (1-5 scale):

**Groundedness** (1-5): How well is each claim traceable to the provided source records?
5 = every claim clearly supported, 1 = mostly fabricated

**Depth** (1-5): How specific and actionable are the persona's goals, pains, vocabulary?
5 = sharp and memorable, 1 = generic platitudes

Persona:
- Name: {persona_dict.get('name')}
- Summary: {persona_dict.get('summary')}
- Goals: {persona_dict.get('goals')}
- Pains: {persona_dict.get('pains')}
- Vocabulary: {persona_dict.get('vocabulary')}
- Quotes: {persona_dict.get('sample_quotes')}

Source records available:
{chr(10).join(record_summary)}

Respond with STRICT JSON only:
{{"groundedness": <int 1-5>, "depth": <int 1-5>, "rationale": "<1-2 sentences>"}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        parsed = {}
    return {
        "groundedness_judge": parsed.get("groundedness"),
        "depth_judge": parsed.get("depth"),
        "rationale": parsed.get("rationale"),
    }


async def main() -> None:
    print("=" * 72)
    print("exp-3.17 — Evidence ablation")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    # ---- Ingest + segment ----
    print("\n[1/3] Fetching and clustering mock records...")
    raw_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]
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
        key=lambda c: len(c.sample_records),
        reverse=True,
    )
    print(f"  Got {len(clusters)} clusters, using largest: {clusters[0].cluster_id} "
          f"({len(clusters[0].sample_records)} records)")

    base_cluster = clusters[0]
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # ---- Run ablation levels ----
    print("\n[2/3] Running ablation levels...")
    results = []

    for pct in ABLATION_LEVELS:
        ablated = create_ablated_cluster(base_cluster, pct)
        label = f"ablated_{int(pct * 100)}pct"
        print(f"\n  [{label}] {len(ablated.sample_records)} records remaining...")

        try:
            r = await synthesize(ablated, backend)
            p_dict = r.persona.model_dump(mode="json")

            # Structural groundedness
            structural_g = r.groundedness.score

            # Claude-as-judge
            judge = await judge_persona_quality(p_dict, ablated, client, settings.default_model)

            print(
                f"    [OK] {p_dict['name'][:40]}  "
                f"structural_g={structural_g:.2f}  "
                f"judge_g={judge['groundedness_judge']}  "
                f"judge_d={judge['depth_judge']}  "
                f"cost=${r.total_cost_usd:.4f}"
            )

            results.append({
                "ablation_pct": int(pct * 100),
                "n_records_remaining": len(ablated.sample_records),
                "status": "ok",
                "persona_name": p_dict["name"],
                "persona": p_dict,
                "structural_groundedness": structural_g,
                "judge_groundedness": judge["groundedness_judge"],
                "judge_depth": judge["depth_judge"],
                "judge_rationale": judge["rationale"],
                "cost_usd": r.total_cost_usd,
                "attempts": r.attempts,
            })
        except SynthesisError as e:
            print(f"    [FAIL] {e}")
            results.append({
                "ablation_pct": int(pct * 100),
                "n_records_remaining": len(ablated.sample_records),
                "status": "failed",
                "error": str(e),
                "cost_usd": sum(a.cost_usd for a in e.attempts),
            })

    # ---- Summary ----
    print("\n[3/3] Writing results...")

    ok_results = [r for r in results if r["status"] == "ok"]

    # Sensitivity: compare 0% to 10%
    g_at_0 = next((r["structural_groundedness"] for r in ok_results if r["ablation_pct"] == 0), None)
    g_at_10 = next((r["structural_groundedness"] for r in ok_results if r["ablation_pct"] == 10), None)
    d_at_0 = next((r["judge_depth"] for r in ok_results if r["ablation_pct"] == 0), None)
    d_at_10 = next((r["judge_depth"] for r in ok_results if r["ablation_pct"] == 10), None)

    summary = {
        "experiment_id": "3.17",
        "branch": "exp-3.17-evidence-ablation",
        "model": settings.default_model,
        "base_cluster_id": base_cluster.cluster_id,
        "base_cluster_records": len(base_cluster.sample_records),
        "ablation_levels": [int(p * 100) for p in ABLATION_LEVELS],
        "results": [
            {
                "ablation_pct": r["ablation_pct"],
                "n_records_remaining": r["n_records_remaining"],
                "status": r["status"],
                "structural_groundedness": r.get("structural_groundedness"),
                "judge_groundedness": r.get("judge_groundedness"),
                "judge_depth": r.get("judge_depth"),
                "persona_name": r.get("persona_name"),
            }
            for r in results
        ],
        "sensitivity": {
            "groundedness_drop_at_10pct": (g_at_0 - g_at_10) if g_at_0 is not None and g_at_10 is not None else None,
            "depth_drop_at_10pct": (d_at_0 - d_at_10) if d_at_0 is not None and d_at_10 is not None else None,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ablated_personas.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
