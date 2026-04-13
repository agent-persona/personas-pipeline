"""exp-5.17 — Inter-annotator agreement.

Measures whether LLM-simulated raters agree on persona quality dimensions,
using Krippendorff's alpha as the agreement metric.

Hypothesis: Groundedness and schema validity achieve alpha >= 0.7 (reliable),
while distinctiveness and voice authenticity fall below alpha < 0.5
(inherently subjective).

Approach:
  1. Fetch + segment + synthesize personas from ALL clusters.
  2. For each persona, have 7 independent LLM "raters" score on 6 dimensions
     (1-5 Likert): schema_validity, groundedness, distinctiveness,
     voice_authenticity, depth, actionability.
  3. Compute Krippendorff's alpha per dimension.
  4. Flag dimensions with alpha < 0.4 as "low agreement".

Usage:
    python scripts/run_exp_5_17.py
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
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from evals.human_protocols.agreement import krippendorff_alpha  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.17"

N_RATERS = 7

DIMENSIONS = [
    "schema_validity",
    "groundedness",
    "distinctiveness",
    "voice_authenticity",
    "depth",
    "actionability",
]

LOW_AGREEMENT_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Rater
# ---------------------------------------------------------------------------

async def rate_persona(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
    rater_id: int,
) -> dict:
    """One simulated rater scores a persona on all 6 dimensions."""
    record_summary = []
    for rec in cluster.sample_records[:8]:
        record_summary.append(f"- {rec.record_id}: {json.dumps(rec.payload)}")

    prompt = f"""You are Rater #{rater_id} in a persona quality study. Score this persona on 6 dimensions using a 1-5 Likert scale.

Persona:
- Name: {persona_dict.get('name')}
- Summary: {persona_dict.get('summary')}
- Goals: {persona_dict.get('goals')}
- Pains: {persona_dict.get('pains')}
- Motivations: {persona_dict.get('motivations')}
- Vocabulary: {persona_dict.get('vocabulary')}
- Quotes: {persona_dict.get('sample_quotes')}

Source records from which this persona was synthesized:
{chr(10).join(record_summary)}

Rate on these dimensions (1=poor, 5=excellent):
1. **schema_validity**: Are all required fields present and well-formed?
2. **groundedness**: Is every claim traceable to the source records?
3. **distinctiveness**: Would you confuse this persona with a generic archetype?
4. **voice_authenticity**: Do the quotes and vocabulary sound like a real person?
5. **depth**: Are goals, pains, motivations specific and actionable?
6. **actionability**: Could a product team make decisions based on this persona?

Respond with STRICT JSON only:
{{"schema_validity": <1-5>, "groundedness": <1-5>, "distinctiveness": <1-5>, "voice_authenticity": <1-5>, "depth": <1-5>, "actionability": <1-5>, "rationale": "<1-2 sentences>"}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {}
    return parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 72)
    print("exp-5.17 -- Inter-annotator agreement")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # ---- Step 1: Fetch + segment + synthesize ALL clusters ----
    print("\n[1/4] Fetching and segmenting records...")
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
    print(f"  Got {len(clusters)} clusters, synthesizing all...")

    personas: list[dict] = []
    persona_clusters: list[ClusterData] = []
    synth_results: list[dict] = []

    for i, cluster in enumerate(clusters):
        print(f"  [{i + 1}/{len(clusters)}] Synthesizing {cluster.cluster_id}...")
        try:
            r = await synthesize(cluster, backend)
            p_dict = r.persona.model_dump(mode="json")
            personas.append(p_dict)
            persona_clusters.append(cluster)
            synth_results.append({
                "cluster_id": cluster.cluster_id,
                "name": p_dict["name"],
                "status": "ok",
                "cost_usd": r.total_cost_usd,
                "groundedness": r.groundedness.score,
            })
            print(f"    [OK] {p_dict['name']}  cost=${r.total_cost_usd:.4f}  grounded={r.groundedness.score:.2f}")
        except SynthesisError as e:
            total_cost = sum(a.cost_usd for a in e.attempts)
            synth_results.append({
                "cluster_id": cluster.cluster_id,
                "status": "failed",
                "error": str(e),
                "cost_usd": total_cost,
            })
            print(f"    [FAIL] {e}  cost=${total_cost:.4f}")

    if not personas:
        raise RuntimeError("No personas synthesized successfully")

    # ---- Step 2: Run raters ----
    print(f"\n[2/4] Running {N_RATERS} raters on {len(personas)} personas...")
    # all_scores: persona_name -> {rater_id_str: {dim: score, ...}}
    all_scores: dict[str, dict[str, dict]] = {}

    for idx, (p, cluster) in enumerate(zip(personas, persona_clusters)):
        name = p["name"]
        print(f"  [{idx + 1}/{len(personas)}] Rating {name}...")
        rater_tasks = [
            rate_persona(p, cluster, client, settings.default_model, rater_id=r)
            for r in range(N_RATERS)
        ]
        rater_results = await asyncio.gather(*rater_tasks)
        all_scores[name] = {str(r): result for r, result in enumerate(rater_results)}

        # Print summary for this persona
        for dim in DIMENSIONS:
            dim_scores = [
                result.get(dim) for result in rater_results if result.get(dim) is not None
            ]
            if dim_scores:
                mean_s = sum(dim_scores) / len(dim_scores)
                print(f"    {dim}: mean={mean_s:.1f}  scores={dim_scores}")

    # ---- Step 3: Compute Krippendorff's alpha per dimension ----
    print(f"\n[3/4] Computing Krippendorff's alpha per dimension...")
    alpha_results: dict[str, dict] = {}

    for dim in DIMENSIONS:
        ratings = []
        for persona_name in all_scores:
            item: dict[str, int] = {}
            for rater_id, scores in all_scores[persona_name].items():
                val = scores.get(dim)
                if val is not None:
                    item[rater_id] = val
            if item:
                ratings.append(item)

        alpha = krippendorff_alpha(ratings, level="ordinal", ordinal_order=[1, 2, 3, 4, 5])
        low_agreement = alpha < LOW_AGREEMENT_THRESHOLD

        alpha_results[dim] = {
            "alpha": alpha,
            "low_agreement": low_agreement,
            "n_items": len(ratings),
            "n_raters": N_RATERS,
        }

        flag = " ** LOW **" if low_agreement else ""
        print(f"  {dim:25s}  alpha={alpha:.3f}{flag}")

    # ---- Step 4: Compute per-dimension score distributions ----
    print(f"\n[4/4] Computing score distributions...")
    dim_distributions: dict[str, dict] = {}
    for dim in DIMENSIONS:
        all_dim_scores = []
        for persona_name in all_scores:
            for rater_id, scores in all_scores[persona_name].items():
                val = scores.get(dim)
                if val is not None:
                    all_dim_scores.append(val)
        if all_dim_scores:
            mean_val = sum(all_dim_scores) / len(all_dim_scores)
            min_val = min(all_dim_scores)
            max_val = max(all_dim_scores)
            dim_distributions[dim] = {
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
                "n": len(all_dim_scores),
            }
            print(f"  {dim:25s}  mean={mean_val:.2f}  range=[{min_val}, {max_val}]  n={len(all_dim_scores)}")

    # ---- Build summary ----
    total_cost = sum(r.get("cost_usd", 0) for r in synth_results)
    high_agreement = [d for d, r in alpha_results.items() if r["alpha"] >= 0.7]
    low_agreement_dims = [d for d, r in alpha_results.items() if r["low_agreement"]]

    summary = {
        "experiment_id": "5.17",
        "branch": "exp-5.17-inter-annotator-agreement",
        "model": settings.default_model,
        "n_personas": len(personas),
        "n_raters": N_RATERS,
        "dimensions": DIMENSIONS,
        "alpha_per_dimension": alpha_results,
        "score_distributions": dim_distributions,
        "high_agreement_dims": high_agreement,
        "low_agreement_dims": low_agreement_dims,
        "low_agreement_threshold": LOW_AGREEMENT_THRESHOLD,
        "total_cost_usd": total_cost,
    }

    # ---- Write outputs ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "ratings.json").write_text(
        json.dumps(all_scores, indent=2, default=str)
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
