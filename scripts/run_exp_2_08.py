"""exp-2.08 — Synthetic-Data Warmstart.

Tests whether prepending synthetic prior personas as context improves
persona quality for sparse tenants without degrading dense-tenant
groundedness.

Four conditions:
  - sparse x baseline (standard system prompt)
  - sparse x treatment (warmstart system prompt)
  - dense  x baseline (standard system prompt)
  - dense  x treatment (warmstart system prompt)

Uses Claude-as-judge to score each persona on groundedness (1-5) and
depth (1-5).

Usage:
    python scripts/run_exp_2_08.py
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
from synthesis.engine.prompt_builder import (  # noqa: E402
    SYSTEM_PROMPT,
    build_warmstart_system_prompt,
)
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-2.08"


def make_sparse_cluster(cluster: ClusterData, n: int = 5) -> ClusterData:
    """Create a sparse variant by subsampling records."""
    sparse = cluster.model_copy(deep=True)
    sparse.sample_records = sparse.sample_records[:n]
    sparse.summary.cluster_size = len(sparse.sample_records)
    sparse.cluster_id = f"{cluster.cluster_id}_sparse"
    return sparse


async def judge_persona(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """Claude-as-judge: rate persona on groundedness and depth."""
    record_summary = []
    for rec in cluster.sample_records[:10]:
        record_summary.append(f"- {rec.record_id}: {rec.payload}")

    judge_prompt = f"""Rate this persona on two dimensions (1-5 scale):

**Groundedness** (1-5): How well is each claim traceable to the source records below?
5 = every claim clearly supported, 1 = mostly fabricated

**Depth** (1-5): How specific and actionable are the goals, pains, vocabulary, and quotes?
5 = sharp, specific, memorable, 1 = generic platitudes

Persona:
- Name: {persona_dict.get('name')}
- Summary: {persona_dict.get('summary')}
- Goals: {persona_dict.get('goals')}
- Pains: {persona_dict.get('pains')}
- Vocabulary: {persona_dict.get('vocabulary')}
- Quotes: {persona_dict.get('sample_quotes')}

Source records (from cluster):
{chr(10).join(record_summary)}

Respond with STRICT JSON only:
{{"groundedness": <int 1-5>, "depth": <int 1-5>, "hallucination_flags": [<list of claims not traceable to records>], "rationale": "<1-2 sentences>"}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=500,
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
        "hallucination_flags": parsed.get("hallucination_flags", []),
        "hallucination_rate": len(parsed.get("hallucination_flags", [])) / max(
            len(persona_dict.get("goals", [])) + len(persona_dict.get("pains", []))
            + len(persona_dict.get("vocabulary", [])),
            1,
        ),
        "rationale": parsed.get("rationale"),
    }


async def run_condition(
    label: str,
    cluster: ClusterData,
    backend: AnthropicBackend,
    client: AsyncAnthropic,
    model: str,
    system_prompt: str | None = None,
) -> dict:
    """Synthesize a persona under one condition and judge it."""
    print(f"\n  [{label}] Synthesizing...")
    try:
        result = await synthesize(
            cluster,
            backend,
            system_prompt=system_prompt,
        )
        p_dict = result.persona.model_dump(mode="json")
        print(
            f"    [OK] {p_dict['name'][:40]:40s}  "
            f"${result.total_cost_usd:.4f}  grounded={result.groundedness.score:.2f}  "
            f"attempts={result.attempts}"
        )

        print(f"    Judging...")
        judge = await judge_persona(p_dict, cluster, client, model)
        print(
            f"    groundedness_judge={judge['groundedness_judge']}  "
            f"depth_judge={judge['depth_judge']}  "
            f"hallucination_flags={len(judge['hallucination_flags'])}"
        )

        return {
            "label": label,
            "cluster_id": cluster.cluster_id,
            "status": "ok",
            "persona": p_dict,
            "cost_usd": result.total_cost_usd,
            "groundedness_score": result.groundedness.score,
            "attempts": result.attempts,
            **judge,
        }
    except SynthesisError as e:
        print(f"    [FAIL] {e}")
        return {
            "label": label,
            "cluster_id": cluster.cluster_id,
            "status": "failed",
            "error": str(e),
            "cost_usd": sum(a.cost_usd for a in e.attempts),
            "groundedness_judge": None,
            "depth_judge": None,
            "hallucination_flags": [],
            "hallucination_rate": None,
            "rationale": None,
        }


async def main() -> None:
    print("=" * 72)
    print("exp-2.08 — Synthetic-Data Warmstart")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    # ---- 1. Ingest + segment ----
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
        key=lambda c: c.cluster_id,
    )
    print(f"  Got {len(clusters)} clusters: {[c.cluster_id for c in clusters]}")

    if not clusters:
        raise RuntimeError("No clusters produced — cannot run experiment")

    # Use the first cluster as the base
    dense_cluster = clusters[0]
    sparse_cluster = make_sparse_cluster(dense_cluster, n=5)
    print(
        f"  Dense cluster: {dense_cluster.cluster_id} "
        f"({len(dense_cluster.sample_records)} records)"
    )
    print(
        f"  Sparse cluster: {sparse_cluster.cluster_id} "
        f"({len(sparse_cluster.sample_records)} records)"
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    warmstart_prompt = build_warmstart_system_prompt()

    # ---- 2. Run 4 conditions ----
    print("\n[2/3] Running 4 conditions...")

    sparse_baseline = await run_condition(
        "sparse_baseline", sparse_cluster, backend, client,
        settings.default_model, system_prompt=None,
    )
    sparse_treatment = await run_condition(
        "sparse_treatment", sparse_cluster, backend, client,
        settings.default_model, system_prompt=warmstart_prompt,
    )
    dense_baseline = await run_condition(
        "dense_baseline", dense_cluster, backend, client,
        settings.default_model, system_prompt=None,
    )
    dense_treatment = await run_condition(
        "dense_treatment", dense_cluster, backend, client,
        settings.default_model, system_prompt=warmstart_prompt,
    )

    conditions = {
        "sparse_baseline": sparse_baseline,
        "sparse_treatment": sparse_treatment,
        "dense_baseline": dense_baseline,
        "dense_treatment": dense_treatment,
    }

    # ---- 3. Summarize ----
    print("\n[3/3] Writing results...")

    def safe_get(d: dict, key: str) -> float | None:
        v = d.get(key)
        return float(v) if v is not None else None

    sb_ground = safe_get(sparse_baseline, "groundedness_judge")
    st_ground = safe_get(sparse_treatment, "groundedness_judge")
    sb_depth = safe_get(sparse_baseline, "depth_judge")
    st_depth = safe_get(sparse_treatment, "depth_judge")
    db_ground = safe_get(dense_baseline, "groundedness_judge")
    dt_ground = safe_get(dense_treatment, "groundedness_judge")
    db_depth = safe_get(dense_baseline, "depth_judge")
    dt_depth = safe_get(dense_treatment, "depth_judge")

    summary = {
        "experiment_id": "2.08",
        "branch": "exp-2.08-synthetic-warmstart",
        "model": settings.default_model,
        "conditions": conditions,
        "delta_sparse_depth": (
            (st_depth - sb_depth) if st_depth is not None and sb_depth is not None else None
        ),
        "delta_sparse_groundedness": (
            (st_ground - sb_ground) if st_ground is not None and sb_ground is not None else None
        ),
        "delta_dense_groundedness": (
            (dt_ground - db_ground) if dt_ground is not None and db_ground is not None else None
        ),
        "delta_dense_depth": (
            (dt_depth - db_depth) if dt_depth is not None and db_depth is not None else None
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Also dump individual condition results
    for name, cond in conditions.items():
        (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(cond, indent=2, default=str))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Model: {settings.default_model}")
    print(f"Sparse baseline:  groundedness={sb_ground}  depth={sb_depth}")
    print(f"Sparse treatment: groundedness={st_ground}  depth={st_depth}")
    print(f"Dense baseline:   groundedness={db_ground}  depth={db_depth}")
    print(f"Dense treatment:  groundedness={dt_ground}  depth={dt_depth}")
    print(f"Delta sparse depth:        {summary['delta_sparse_depth']}")
    print(f"Delta sparse groundedness: {summary['delta_sparse_groundedness']}")
    print(f"Delta dense groundedness:  {summary['delta_dense_groundedness']}")
    print(f"Delta dense depth:         {summary['delta_dense_depth']}")
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
