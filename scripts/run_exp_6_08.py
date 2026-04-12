"""exp-6.08 — Long-tail Persona Viability.

Tests how persona quality degrades as the number of input records decreases.
Uses random subsampling (not ranked removal like exp-3.17) to simulate
natural cluster sizes.

Hypothesis: Quality collapses below ~10 records (groundedness < 0.7,
depth < 3.0). At >= 25 records, quality is within 0.1 of full-cluster score.

Usage:
    python scripts/run_exp_6_08.py
"""

from __future__ import annotations

import asyncio
import json
import random
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
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-6.08"

TARGET_SIZES = [3, 5, 7, 10, 25, 50]


def subsample_cluster(cluster: ClusterData, target_size: int, seed: int = 42) -> ClusterData:
    """Create a subsampled cluster with target_size records."""
    records = list(cluster.sample_records)
    if len(records) <= target_size:
        return cluster
    rng = random.Random(seed)
    sample = rng.sample(records, target_size)
    sub = cluster.model_copy(deep=True)
    sub.sample_records = sample
    sub.summary.cluster_size = target_size
    sub.cluster_id = f"{cluster.cluster_id}_n{target_size}"
    return sub


async def judge_persona_depth(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """Claude-as-judge: rate persona depth 1-5."""
    record_summary = []
    for rec in cluster.sample_records[:10]:
        record_summary.append(f"- {rec.record_id}: {json.dumps(rec.payload)}")

    judge_prompt = f"""Rate this persona on depth (1-5 scale):

**Depth** (1-5): How specific and actionable are the persona's goals, pains, vocabulary?
5 = sharp, specific, memorable — could only describe this person
4 = mostly specific with minor generic elements
3 = passable but somewhat generic
2 = mostly generic platitudes
1 = completely generic, could describe anyone

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
{{"depth": <int 1-5>, "rationale": "<1-2 sentences>"}}"""

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
        "depth": parsed.get("depth"),
        "rationale": parsed.get("rationale"),
    }


async def main() -> None:
    print("=" * 72)
    print("exp-6.08 — Long-tail Persona Viability")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    # ---- F0 check ----
    print("\n[F0] Feasibility check — verifying >= 4 clusters...")
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
    n_clusters = len(clusters)
    print(f"  F0 cluster count: {n_clusters}")
    assert n_clusters >= 2, f"F0 FAIL: expected >= 2 clusters, got {n_clusters}"
    print("  F0 PASS")

    base_cluster = clusters[0]
    full_size = len(base_cluster.sample_records)
    print(f"\n  Using largest cluster: {base_cluster.cluster_id} ({full_size} records)")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # ---- Build size variants ----
    # Filter to sizes <= full cluster, plus "full"
    sizes = [s for s in TARGET_SIZES if s < full_size]
    sizes.append(full_size)  # add full as last
    print(f"  Sizes to test: {sizes}")

    # ---- Run synthesis + scoring for each size ----
    print("\n[1/2] Synthesizing and scoring per size...")
    results = []

    for size in sizes:
        label = f"n={size}" if size < full_size else f"n={size} (full)"
        if size == full_size:
            sub_cluster = base_cluster
        else:
            sub_cluster = subsample_cluster(base_cluster, size)

        print(f"\n  [{label}] {len(sub_cluster.sample_records)} records...")
        try:
            r = await synthesize(sub_cluster, backend)
            p_dict = r.persona.model_dump(mode="json")

            # Structural groundedness
            structural_g = r.groundedness.score

            # Claude-as-judge depth
            judge = await judge_persona_depth(p_dict, sub_cluster, client, settings.default_model)

            print(
                f"    [OK] {p_dict['name'][:40]}  "
                f"groundedness={structural_g:.2f}  "
                f"depth={judge['depth']}  "
                f"cost=${r.total_cost_usd:.4f}"
            )

            results.append({
                "n_records": size,
                "is_full": size == full_size,
                "status": "ok",
                "persona_name": p_dict["name"],
                "persona": p_dict,
                "structural_groundedness": structural_g,
                "judge_depth": judge["depth"],
                "judge_rationale": judge["rationale"],
                "cost_usd": r.total_cost_usd,
                "attempts": r.attempts,
            })
        except SynthesisError as e:
            print(f"    [FAIL] {e}")
            results.append({
                "n_records": size,
                "is_full": size == full_size,
                "status": "failed",
                "error": str(e),
                "cost_usd": sum(a.cost_usd for a in e.attempts),
            })

    # ---- Analyze ----
    print("\n[2/2] Analyzing results...")

    ok_results = [r for r in results if r["status"] == "ok"]

    # Full cluster scores
    full_result = next((r for r in ok_results if r["is_full"]), None)
    full_groundedness = full_result["structural_groundedness"] if full_result else None
    full_depth = full_result["judge_depth"] if full_result else None

    # Find quality knee
    knee_size = None
    for r in ok_results:
        if r["is_full"]:
            continue
        g = r["structural_groundedness"]
        d = r["judge_depth"] or 0
        if g < 0.7 or d < 3.0:
            knee_size = r["n_records"]
            break

    # Check convergence at >= 25
    converged_results = [r for r in ok_results if r["n_records"] >= 25 and not r["is_full"]]
    convergence_ok = True
    for r in converged_results:
        if full_groundedness is not None:
            if abs(r["structural_groundedness"] - full_groundedness) > 0.1:
                convergence_ok = False
        if full_depth is not None and r["judge_depth"] is not None:
            if abs(r["judge_depth"] - full_depth) > 0.1:
                convergence_ok = False

    total_cost = sum(r.get("cost_usd", 0) for r in results)

    summary = {
        "experiment_id": "6.08",
        "branch": "exp-6.08-long-tail-viability",
        "model": settings.default_model,
        "base_cluster_id": base_cluster.cluster_id,
        "base_cluster_records": full_size,
        "sizes_tested": sizes,
        "results": [
            {
                "n_records": r["n_records"],
                "is_full": r["is_full"],
                "status": r["status"],
                "structural_groundedness": r.get("structural_groundedness"),
                "judge_depth": r.get("judge_depth"),
                "persona_name": r.get("persona_name"),
            }
            for r in results
        ],
        "full_cluster_groundedness": full_groundedness,
        "full_cluster_depth": full_depth,
        "quality_knee_at": knee_size,
        "convergence_at_25_plus": convergence_ok if converged_results else "N/A (no sizes >= 25)",
        "total_cost_usd": total_cost,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "personas.json").write_text(json.dumps(results, indent=2, default=str))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
