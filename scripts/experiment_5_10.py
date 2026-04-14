"""Experiment 5.10: Pairwise vs absolute scoring.

Hypothesis: Pairwise preference judging produces higher inter-judge agreement
and cleaner persona rankings than absolute 1-5 scoring.

Uses:
  - evaluation/judges.py LLMJudge (real impl from exp-5.13)
  - evals/pairwise_judging.py (new, this experiment)
  - metrics.spearman_correlation (from yash/eval-infra PR #66 if landed)
    — fallback to a local rank correlation if not available

Method:
  1. Generate N personas by running synthesis multiple times on tenant_acme_corp
  2. Score every persona with absolute mode (real exp-5.13 judge)
  3. Run pairwise tournament on all N*(N-1)/2 pairs
  4. Convert pairwise results into a ranking via win counts + Bradley-Terry
  5. Compute Spearman correlation between the absolute ranking and both
     pairwise rankings (win count + BT)
  6. Also run the absolute scoring with a second model to measure inter-judge
     agreement in each mode (if haiku is available)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for pkg in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / pkg))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")
load_dotenv(Path("/Users/yash/Orchestrator/discord-personas/.env"))

import anthropic
from anthropic import AsyncAnthropic

from synthesis.engine.synthesizer import synthesize
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.models.cluster import (
    ClusterData, ClusterSummary, EnrichmentPayload, SampleRecord, TenantContext,
)
from evaluation.judges import LLMJudge, JudgeBackend
from evals.pairwise_judging import (
    PairwiseVerdict, pairwise_judge, win_count_ranking, bradley_terry_ranking,
    aggregate_dimensions, DIMENSIONS,
)


OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.10-pairwise-vs-absolute"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


JUDGE_MODEL_A = "claude-sonnet-4-20250514"   # primary judge
JUDGE_MODEL_B = "claude-haiku-4-5-20251001"  # secondary for inter-judge agreement
SYNTH_MODEL = "claude-haiku-4-5-20251001"


def make_cluster() -> ClusterData:
    return ClusterData(
        cluster_id="cluster_001",
        tenant=TenantContext(
            tenant_id="tenant_acme_corp",
            industry="B2B SaaS",
            product_description="Project management tool for engineering teams",
            existing_persona_names=[],
        ),
        summary=ClusterSummary(
            cluster_size=150,
            top_behaviors=["views pricing page", "downloads whitepaper", "requests demo"],
            top_pages=["/pricing", "/features", "/case-studies"],
            conversion_rate=0.12,
            avg_session_duration_seconds=340,
            top_referrers=["google", "linkedin", "direct"],
        ),
        sample_records=[
            SampleRecord(record_id="rec_001", source="ga4", timestamp="2026-03-15",
                         payload={"page": "/pricing", "duration": 120, "action": "scroll_to_bottom"}),
            SampleRecord(record_id="rec_002", source="hubspot", timestamp="2026-03-16",
                         payload={"type": "form_submit", "form": "demo_request", "company_size": "50-200"}),
            SampleRecord(record_id="rec_003", source="intercom", timestamp="2026-03-17",
                         payload={"message": "How does your tool integrate with Jira?", "sentiment": "curious"}),
            SampleRecord(record_id="rec_004", source="ga4", timestamp="2026-03-18",
                         payload={"page": "/case-studies/enterprise", "duration": 95, "referrer": "linkedin"}),
            SampleRecord(record_id="rec_005", source="hubspot", timestamp="2026-03-19",
                         payload={"type": "email_open", "campaign": "q1_nurture", "clicks": 2}),
        ],
        enrichment=EnrichmentPayload(
            firmographic={"median_company_size": "50-200"},
            intent_signals=["evaluating project management tools", "comparing vendors"],
            technographic={"tools": ["Jira", "Slack", "GitHub"]},
        ),
    )


def spearman(x: list[float], y: list[float]) -> float:
    """Inline Spearman so we don't depend on yash/eval-infra being merged."""
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0

    def _rank(vs: list[float]) -> list[float]:
        # Rank with tie-averaging
        indexed = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[indexed[j + 1]] == vs[indexed[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((r - mean_rx) ** 2 for r in rx) ** 0.5
    den_y = sum((r - mean_ry) ** 2 for r in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


async def synth_n_personas(n: int) -> list[dict]:
    """Run the real synthesizer N times. Each call produces one persona."""
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    backend = AnthropicBackend(client=client, model=SYNTH_MODEL)
    personas = []
    total_cost = 0.0
    for i in range(n):
        try:
            result = await synthesize(make_cluster(), backend)
            personas.append(result.persona.model_dump())
            total_cost += result.total_cost_usd
            print(f"  synth {i+1}/{n}: cost=${result.total_cost_usd:.4f}", flush=True)
        except Exception as e:
            print(f"  synth {i+1}/{n}: FAILED ({type(e).__name__})", flush=True)
    return personas, total_cost


async def absolute_scores(
    personas: list[dict],
    judge_model: str,
) -> list[dict]:
    """Score every persona in absolute mode with the real exp-5.13 judge."""
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    backend = JudgeBackend(client=client, model=judge_model)
    judge = LLMJudge(backend=backend, model=judge_model, calibration="none")

    scores = []
    for i, p in enumerate(personas):
        try:
            result = await judge.score_persona(p)
            scores.append({
                "persona_id": f"P{i+1}",
                "overall": result.overall,
                "dimensions": result.dimensions,
            })
            print(f"  abs {judge_model} P{i+1}: {result.overall:.2f}", flush=True)
        except Exception as e:
            print(f"  abs {judge_model} P{i+1}: FAILED ({type(e).__name__})", flush=True)
            scores.append({"persona_id": f"P{i+1}", "overall": 0.0, "dimensions": {}})
    return scores


async def pairwise_tournament(
    personas: list[dict],
    judge_model: str,
) -> list[PairwiseVerdict]:
    """Run all N*(N-1)/2 pair comparisons with the pairwise judge."""
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    verdicts = []
    n = len(personas)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                v = await pairwise_judge(
                    client=client,
                    model=judge_model,
                    persona_a=personas[i],
                    persona_b=personas[j],
                    persona_a_id=f"P{i+1}",
                    persona_b_id=f"P{j+1}",
                )
                verdicts.append(v)
                tally = {k: v.debiased[k] for k in DIMENSIONS}
                print(f"  pair P{i+1}vsP{j+1} {judge_model}: {tally}", flush=True)
            except Exception as e:
                print(f"  pair P{i+1}vsP{j+1}: FAILED ({type(e).__name__})", flush=True)
    return verdicts


async def main():
    print("=" * 70, flush=True)
    print("exp-5.10: pairwise vs absolute scoring", flush=True)
    print("=" * 70, flush=True)

    start = time.time()

    # Step 1: generate N personas
    N_PERSONAS = 4
    print(f"\n[1] generating {N_PERSONAS} personas via real synthesizer", flush=True)
    personas, synth_cost = await synth_n_personas(N_PERSONAS)
    if len(personas) < 3:
        print(f"insufficient personas ({len(personas)}) — cannot compare rankings", flush=True)
        _write_results({
            "status": "insufficient_personas",
            "num_personas": len(personas),
            "synth_cost_usd": synth_cost,
        })
        return

    n = len(personas)
    persona_ids = [f"P{i+1}" for i in range(n)]

    # Step 2: absolute scoring with Sonnet (primary judge)
    print(f"\n[2a] absolute scoring — {JUDGE_MODEL_A}", flush=True)
    abs_sonnet = await absolute_scores(personas, JUDGE_MODEL_A)

    print(f"\n[2b] absolute scoring — {JUDGE_MODEL_B}", flush=True)
    abs_haiku = await absolute_scores(personas, JUDGE_MODEL_B)

    # Step 3: pairwise tournament with Sonnet
    print(f"\n[3a] pairwise tournament — {JUDGE_MODEL_A}", flush=True)
    verdicts_sonnet = await pairwise_tournament(personas, JUDGE_MODEL_A)

    print(f"\n[3b] pairwise tournament — {JUDGE_MODEL_B}", flush=True)
    verdicts_haiku = await pairwise_tournament(personas, JUDGE_MODEL_B)

    # Step 4: convert pairwise -> rankings
    wc_sonnet = aggregate_dimensions(persona_ids, win_count_ranking, verdicts_sonnet)
    bt_sonnet = aggregate_dimensions(persona_ids, bradley_terry_ranking, verdicts_sonnet)
    wc_haiku = aggregate_dimensions(persona_ids, win_count_ranking, verdicts_haiku)
    bt_haiku = aggregate_dimensions(persona_ids, bradley_terry_ranking, verdicts_haiku)

    # Step 5: compute correlations
    abs_sonnet_vec = [s["overall"] for s in abs_sonnet]
    abs_haiku_vec = [s["overall"] for s in abs_haiku]
    wc_sonnet_vec = [wc_sonnet[pid] for pid in persona_ids]
    bt_sonnet_vec = [bt_sonnet[pid] for pid in persona_ids]
    wc_haiku_vec = [wc_haiku[pid] for pid in persona_ids]
    bt_haiku_vec = [bt_haiku[pid] for pid in persona_ids]

    # Within-model agreement: does pairwise rank the same as absolute?
    same_model_abs_vs_wc = spearman(abs_sonnet_vec, wc_sonnet_vec)
    same_model_abs_vs_bt = spearman(abs_sonnet_vec, bt_sonnet_vec)

    # Inter-judge agreement in each mode
    inter_judge_absolute = spearman(abs_sonnet_vec, abs_haiku_vec)
    inter_judge_pairwise_wc = spearman(wc_sonnet_vec, wc_haiku_vec)
    inter_judge_pairwise_bt = spearman(bt_sonnet_vec, bt_haiku_vec)

    # Score distribution tightness
    abs_sonnet_range = max(abs_sonnet_vec) - min(abs_sonnet_vec)
    abs_haiku_range = max(abs_haiku_vec) - min(abs_haiku_vec)
    wc_range = max(wc_sonnet_vec) - min(wc_sonnet_vec)

    # Position bias rate: how often do the two orderings disagree (become TIE)?
    position_bias_count = 0
    total_dim_votes = 0
    for v in verdicts_sonnet + verdicts_haiku:
        for d in DIMENSIONS:
            total_dim_votes += 1
            # Check if debiased is "tie" but both orderings picked a winner
            if v.debiased[d] == "tie" and v.ab_order[d] != "tie":
                # Unflip ba to check if it was actually a disagreement
                ba = v.ba_order[d]
                flipped = "b" if ba == "a" else "a" if ba == "b" else "tie"
                if v.ab_order[d] != flipped and flipped != "tie":
                    position_bias_count += 1
    position_bias_rate = position_bias_count / max(total_dim_votes, 1)

    duration_ms = int((time.time() - start) * 1000)

    results = {
        "experiment_id": "5.10",
        "title": "Pairwise vs absolute scoring",
        "num_personas": n,
        "judge_models": [JUDGE_MODEL_A, JUDGE_MODEL_B],
        "synth_cost_usd": round(synth_cost, 4),
        "duration_ms": duration_ms,
        "absolute": {
            JUDGE_MODEL_A: [
                {"id": persona_ids[i], "overall": abs_sonnet[i]["overall"],
                 "dimensions": abs_sonnet[i]["dimensions"]}
                for i in range(n)
            ],
            JUDGE_MODEL_B: [
                {"id": persona_ids[i], "overall": abs_haiku[i]["overall"],
                 "dimensions": abs_haiku[i]["dimensions"]}
                for i in range(n)
            ],
        },
        "pairwise": {
            JUDGE_MODEL_A: {
                "win_count_ranking": wc_sonnet,
                "bradley_terry_ranking": bt_sonnet,
                "verdicts": [
                    {
                        "a": v.persona_a_id, "b": v.persona_b_id,
                        "debiased": v.debiased,
                        "ab_order": v.ab_order,
                        "ba_order": v.ba_order,
                    }
                    for v in verdicts_sonnet
                ],
            },
            JUDGE_MODEL_B: {
                "win_count_ranking": wc_haiku,
                "bradley_terry_ranking": bt_haiku,
            },
        },
        "correlations": {
            "within_model_absolute_vs_winCount_sonnet": round(same_model_abs_vs_wc, 3),
            "within_model_absolute_vs_bradleyTerry_sonnet": round(same_model_abs_vs_bt, 3),
            "inter_judge_absolute_spearman": round(inter_judge_absolute, 3),
            "inter_judge_pairwise_winCount_spearman": round(inter_judge_pairwise_wc, 3),
            "inter_judge_pairwise_bradleyTerry_spearman": round(inter_judge_pairwise_bt, 3),
        },
        "distribution": {
            "absolute_sonnet_range": round(abs_sonnet_range, 3),
            "absolute_haiku_range": round(abs_haiku_range, 3),
            "pairwise_winCount_range": round(wc_range, 3),
        },
        "position_bias": {
            "disagreement_rate": round(position_bias_rate, 3),
            "disagreements": position_bias_count,
            "total_dim_votes": total_dim_votes,
        },
    }

    # Decision
    # Adopt if: pairwise inter-judge agreement > absolute inter-judge agreement by >= 0.1
    #           AND position bias rate < 0.3
    best_pairwise_inter = max(inter_judge_pairwise_wc, inter_judge_pairwise_bt)
    inter_judge_lift = best_pairwise_inter - inter_judge_absolute
    if inter_judge_lift >= 0.1 and position_bias_rate < 0.3:
        decision = "adopt"
    elif inter_judge_lift <= -0.1 or position_bias_rate > 0.5:
        decision = "reject"
    else:
        decision = "defer"

    results["decision"] = decision
    results["decision_rationale"] = (
        f"inter-judge pairwise Spearman = {best_pairwise_inter:.2f}, "
        f"inter-judge absolute = {inter_judge_absolute:.2f}, "
        f"lift = {inter_judge_lift:+.2f}, "
        f"position bias rate = {position_bias_rate:.2f}"
    )

    print(f"\n{'=' * 70}", flush=True)
    print(f"DECISION: {decision}", flush=True)
    print(f"Inter-judge absolute Spearman: {inter_judge_absolute:.3f}", flush=True)
    print(f"Inter-judge pairwise (win count) Spearman: {inter_judge_pairwise_wc:.3f}", flush=True)
    print(f"Inter-judge pairwise (Bradley-Terry) Spearman: {inter_judge_pairwise_bt:.3f}", flush=True)
    print(f"Position bias disagreement rate: {position_bias_rate:.3f}", flush=True)
    print(f"Duration: {duration_ms}ms", flush=True)
    print(f"Synth cost: ${synth_cost:.4f}", flush=True)

    _write_results(results)


def _write_results(results: dict) -> None:
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUTPUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
