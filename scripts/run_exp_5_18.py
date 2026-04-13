"""exp-5.18 — Crowd worker vs Expert reviewer.

Hypothesis: "Crowd" (general-purpose LLM judge) labels suffice (kappa >= 0.6
vs "experts") for schema validity and groundedness, but diverge (kappa < 0.4)
on distinctiveness and voice authenticity.

We simulate "crowd workers" as standard LLM judge calls and "expert reviewers"
as LLM calls with an expert system prompt that includes domain expertise framing.

Approach:
  1. Fetch + segment + synthesize personas from all clusters.
  2. Crowd arm: 5 LLM judge calls per persona, standard prompt, temperature=0.8.
  3. Expert arm: 3 LLM judge calls per persona, expert-framed prompt, temperature=0.5.
  4. Both rate on 6 dimensions (1-5 Likert).
  5. Aggregate each arm's scores per persona per dimension (mean -> round to int).
  6. Compute Cohen's kappa between crowd-aggregate and expert-aggregate per dimension.

Usage:
    python scripts/run_exp_5_18.py
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
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from evals.human_protocols.agreement import cohen_kappa  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.18"

DIMENSIONS = [
    "schema_validity",
    "groundedness",
    "distinctiveness",
    "voice_authenticity",
    "depth",
    "actionability",
]

N_CROWD_RATERS = 5
N_EXPERT_RATERS = 3


# ---------------------------------------------------------------------------
# Rater functions
# ---------------------------------------------------------------------------

async def crowd_rate_persona(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
    rater_id: int,
) -> dict:
    """One simulated crowd rater judges a single persona."""
    record_summary = [
        f"- {rec.record_id}: {json.dumps(rec.payload)}"
        for rec in cluster.sample_records[:8]
    ]
    prompt = f"""You are participant #{rater_id} in a persona quality study. Score this persona on 6 dimensions (1-5 scale).

Persona:
- Name: {persona_dict.get('name')}
- Summary: {persona_dict.get('summary')}
- Goals: {persona_dict.get('goals')}
- Pains: {persona_dict.get('pains')}
- Vocabulary: {persona_dict.get('vocabulary')}
- Quotes: {persona_dict.get('sample_quotes')}

Source records:
{chr(10).join(record_summary)}

Dimensions (1=poor, 5=excellent):
1. schema_validity: Are all fields present and well-formed?
2. groundedness: Is every claim traceable to source records?
3. distinctiveness: Is this persona unique, not a generic archetype?
4. voice_authenticity: Do quotes/vocabulary sound like a real person?
5. depth: Are goals/pains specific and actionable?
6. actionability: Could a product team use this persona for decisions?

Respond with STRICT JSON only:
{{"schema_validity": <1-5>, "groundedness": <1-5>, "distinctiveness": <1-5>, "voice_authenticity": <1-5>, "depth": <1-5>, "actionability": <1-5>}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


async def expert_rate_persona(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
    rater_id: int,
) -> dict:
    """One simulated expert rater judges a single persona."""
    record_summary = [
        f"- {rec.record_id}: {json.dumps(rec.payload)}"
        for rec in cluster.sample_records[:8]
    ]
    prompt = f"""You are Expert Reviewer #{rater_id} — a senior UX researcher with 10+ years building buyer personas for enterprise SaaS products. You have deep expertise in qualitative research methods, persona development frameworks (Cooper, Pruitt & Adlin), and have reviewed hundreds of persona artifacts.

Apply your expert judgment to score this persona on 6 dimensions (1-5 scale). Be more critical than a general audience — you know what good personas look like.

Persona:
- Name: {persona_dict.get('name')}
- Summary: {persona_dict.get('summary')}
- Goals: {persona_dict.get('goals')}
- Pains: {persona_dict.get('pains')}
- Vocabulary: {persona_dict.get('vocabulary')}
- Quotes: {persona_dict.get('sample_quotes')}

Source records:
{chr(10).join(record_summary)}

Expert scoring criteria:
1. schema_validity: Structural completeness against industry-standard persona frameworks
2. groundedness: Evidence traceability — can you point to specific records for each claim?
3. distinctiveness: Would this persona survive a "squint test" against 50 others in your portfolio?
4. voice_authenticity: Could you use these quotes in a stakeholder presentation without someone calling them out as synthetic?
5. depth: Enough specificity to drive feature prioritization, not just marketing copy
6. actionability: Would a PM change their roadmap based on this persona's goals and pains?

Respond with STRICT JSON only:
{{"schema_validity": <1-5>, "groundedness": <1-5>, "distinctiveness": <1-5>, "voice_authenticity": <1-5>, "depth": <1-5>, "actionability": <1-5>}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_scores(
    all_ratings: list[dict], dimensions: list[str]
) -> dict[str, int]:
    """Mean score per dimension, rounded to int."""
    totals = {d: [] for d in dimensions}
    for rating in all_ratings:
        for d in dimensions:
            if d in rating:
                totals[d].append(rating[d])
    return {d: round(sum(vals) / len(vals)) if vals else 0 for d, vals in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 72)
    print("exp-5.18 — Crowd worker vs Expert reviewer")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)
    model = settings.default_model

    # ---- Step 1: Fetch + segment + synthesize ----
    print("\n[1/5] Fetching and segmenting records...")
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

    print("\n[2/5] Synthesizing personas from all clusters...")
    persona_results = []
    for i, cluster in enumerate(clusters):
        print(f"  [{i + 1}/{len(clusters)}] {cluster.cluster_id}...")
        try:
            r = await synthesize(cluster, backend)
            p_dict = r.persona.model_dump(mode="json")
            persona_results.append({
                "cluster_id": cluster.cluster_id,
                "status": "ok",
                "persona": p_dict,
                "cluster": cluster,
                "cost_usd": r.total_cost_usd,
                "groundedness": r.groundedness.score,
            })
            print(
                f"    [OK] {p_dict['name']}  "
                f"cost=${r.total_cost_usd:.4f}  "
                f"grounded={r.groundedness.score:.2f}"
            )
        except SynthesisError as e:
            total_cost = sum(a.cost_usd for a in e.attempts)
            persona_results.append({
                "cluster_id": cluster.cluster_id,
                "status": "failed",
                "error": str(e),
                "cost_usd": total_cost,
            })
            print(f"    [FAIL] {e}  cost=${total_cost:.4f}")

    ok_results = [r for r in persona_results if r["status"] == "ok"]
    if not ok_results:
        raise RuntimeError("No personas synthesized successfully")
    print(f"  {len(ok_results)} personas synthesized successfully")

    # ---- Step 2: Crowd arm (5 raters per persona) ----
    print(f"\n[3/5] Running crowd arm ({N_CROWD_RATERS} raters per persona)...")
    crowd_ratings: dict[str, list[dict]] = {}  # cluster_id -> list of rating dicts
    for pr in ok_results:
        cid = pr["cluster_id"]
        print(f"  Crowd-rating {cid}...")
        tasks = [
            crowd_rate_persona(pr["persona"], pr["cluster"], client, model, rid)
            for rid in range(1, N_CROWD_RATERS + 1)
        ]
        ratings = await asyncio.gather(*tasks)
        crowd_ratings[cid] = [r for r in ratings if r]
        print(f"    Got {len(crowd_ratings[cid])}/{N_CROWD_RATERS} valid ratings")

    # ---- Step 3: Expert arm (3 raters per persona) ----
    print(f"\n[4/5] Running expert arm ({N_EXPERT_RATERS} raters per persona)...")
    expert_ratings: dict[str, list[dict]] = {}  # cluster_id -> list of rating dicts
    for pr in ok_results:
        cid = pr["cluster_id"]
        print(f"  Expert-rating {cid}...")
        tasks = [
            expert_rate_persona(pr["persona"], pr["cluster"], client, model, rid)
            for rid in range(1, N_EXPERT_RATERS + 1)
        ]
        ratings = await asyncio.gather(*tasks)
        expert_ratings[cid] = [r for r in ratings if r]
        print(f"    Got {len(expert_ratings[cid])}/{N_EXPERT_RATERS} valid ratings")

    # ---- Step 4: Aggregate and compute kappa ----
    print("\n[5/5] Computing agreement metrics...")

    # Aggregate per persona per dimension
    crowd_agg: dict[str, dict[str, int]] = {}
    expert_agg: dict[str, dict[str, int]] = {}
    for pr in ok_results:
        cid = pr["cluster_id"]
        crowd_agg[cid] = aggregate_scores(crowd_ratings.get(cid, []), DIMENSIONS)
        expert_agg[cid] = aggregate_scores(expert_ratings.get(cid, []), DIMENSIONS)

    # Compute Cohen's kappa per dimension
    persona_ids = [pr["cluster_id"] for pr in ok_results]
    kappa_results = {}
    for dim in DIMENSIONS:
        crowd_labels = [crowd_agg[pid][dim] for pid in persona_ids]
        expert_labels = [expert_agg[pid][dim] for pid in persona_ids]
        k = cohen_kappa(crowd_labels, expert_labels)
        kappa_results[dim] = k
        print(f"  {dim:25s}  kappa={k:+.3f}")

    # Evaluate hypothesis
    high_agreement_dims = ["schema_validity", "groundedness"]
    low_agreement_dims = ["distinctiveness", "voice_authenticity"]

    high_pass = all(kappa_results.get(d, -1) >= 0.6 for d in high_agreement_dims)
    low_pass = all(kappa_results.get(d, 1) < 0.4 for d in low_agreement_dims)
    hypothesis_pass = high_pass and low_pass

    print(f"\n  High-agreement dims (>= 0.6): {high_pass}")
    print(f"  Low-agreement dims (< 0.4):  {low_pass}")
    print(f"  Hypothesis supported:         {hypothesis_pass}")

    # ---- Write outputs ----
    total_cost = sum(r.get("cost_usd", 0) for r in persona_results)

    # Build detailed ratings output
    ratings_output = {}
    for pr in ok_results:
        cid = pr["cluster_id"]
        ratings_output[cid] = {
            "persona_name": pr["persona"].get("name"),
            "crowd_raw": crowd_ratings.get(cid, []),
            "expert_raw": expert_ratings.get(cid, []),
            "crowd_aggregated": crowd_agg[cid],
            "expert_aggregated": expert_agg[cid],
        }

    summary = {
        "experiment_id": "5.18",
        "branch": "exp-5.18-crowd-vs-expert",
        "model": model,
        "n_personas": len(ok_results),
        "n_crowd_raters": N_CROWD_RATERS,
        "n_expert_raters": N_EXPERT_RATERS,
        "kappa_per_dimension": kappa_results,
        "hypothesis": {
            "description": "Crowd labels suffice (kappa >= 0.6) for schema_validity and groundedness, but diverge (kappa < 0.4) on distinctiveness and voice_authenticity",
            "high_agreement_pass": high_pass,
            "low_agreement_pass": low_pass,
            "overall_pass": hypothesis_pass,
        },
        "total_cost_usd": total_cost,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "ratings.json").write_text(
        json.dumps(ratings_output, indent=2, default=str)
    )

    # ---- Print summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
