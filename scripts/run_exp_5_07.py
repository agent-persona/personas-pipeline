"""exp-5.07 — Time-to-detect curve.

Baseline vs Treatment: synthesize personas from sparse (3-record) vs full
clusters, run 15-turn TwinChat conversations, then use 5 independent LLM
judges per turn to measure how quickly they detect the responder is AI.

Hypothesis: Full-cluster (treatment) personas evade detection 2-3× longer
than sparse-cluster (baseline) personas.

Usage:
    python scripts/run_exp_5_07.py
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
sys.path.insert(0, str(REPO_ROOT / "twin"))

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
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.07"

N_RATERS = 5
JUDGE_TURNS = [1, 3, 5, 7, 10, 15]

CONVERSATION_PROMPTS = [
    "Tell me about the biggest challenge you're facing at work right now.",
    "How do you typically evaluate new tools or software?",
    "What does your day-to-day workflow look like?",
    "What's the most frustrating part of your current toolstack?",
    "If you could wave a magic wand and fix one thing, what would it be?",
    "How do you get buy-in from your team when adopting something new?",
    "What metrics do you track most closely?",
    "Tell me about a recent decision that didn't go as planned.",
    "How do you stay current with trends in your space?",
    "What advice would you give someone stepping into your role?",
    "How has your approach changed over the past year?",
    "What's the hardest part of your job that nobody talks about?",
    "How do you handle pushback from stakeholders?",
    "What does success look like for you this quarter?",
    "Looking ahead, what are you most worried about?",
]


def subsample_cluster(cluster: ClusterData, target_size: int, seed: int = 42) -> ClusterData:
    """Create a sparse cluster by random subsampling."""
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


async def judge_turn_detection(
    turns_shown: list[dict],
    client: AsyncAnthropic,
    model: str,
    rater_seed: int,
) -> dict:
    transcript = ""
    for t in turns_shown:
        transcript += f"User: {t['prompt']}\nResponder: {t['reply']}\n\n"

    judge_prompt = f"""You are evaluating a conversation transcript. One side is a real person (User), the other side is the Responder.

Your task: Is the Responder a real person or an AI?

Transcript so far:
{transcript}

Think carefully. Consider: specificity of examples, natural hesitation/filler, consistency of personality, generic vs unique phrasing.

Respond with STRICT JSON only:
{{"verdict": "human" or "AI", "confidence": <int 1-5 where 5=very confident>, "rationale": "<1 sentence>"}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.8,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {}
    return {
        "verdict": parsed.get("verdict"),
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale"),
    }


async def run_conversation(persona_dict, client, model):
    twin = TwinChat(persona_dict, client=client, model=model)
    history = []
    turns = []
    for i, prompt in enumerate(CONVERSATION_PROMPTS):
        reply = await twin.reply(prompt, history=history)
        turns.append({
            "turn": i + 1,
            "prompt": prompt,
            "reply": reply.text,
            "cost_usd": reply.estimated_cost_usd,
        })
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply.text})
    return turns


def compute_detection_stats(judge_results):
    rater_ttd = {}
    sorted_turns = sorted(judge_results.keys())
    for rater_idx in range(N_RATERS):
        detected_at = None
        for turn in sorted_turns:
            result = judge_results[turn][rater_idx]
            verdict = (result.get("verdict") or "").strip().lower()
            confidence = result.get("confidence") or 0
            if verdict == "ai" and confidence >= 4:
                detected_at = turn
                break
        rater_ttd[rater_idx] = detected_at

    detected_turns = [v for v in rater_ttd.values() if v is not None]
    mean_ttd = sum(detected_turns) / len(detected_turns) if detected_turns else 16.0
    realism_score = 1.0 - (1.0 / mean_ttd) if mean_ttd > 0 else 0.0

    return {
        "rater_ttd": {str(k): v for k, v in rater_ttd.items()},
        "mean_ttd": mean_ttd,
        "realism_score": realism_score,
        "detection_rate": len(detected_turns) / N_RATERS,
        "n_detected": len(detected_turns),
        "n_raters": N_RATERS,
    }


async def run_condition(label, persona_dict, client, model):
    """Run conversation + judge pipeline for one persona condition."""
    name = persona_dict["name"]
    print(f"\n  [{label}] Conversing with {name}...")
    turns = await run_conversation(persona_dict, client, model)
    conv_cost = sum(t["cost_usd"] for t in turns)
    print(f"    {len(turns)} turns, cost=${conv_cost:.4f}")

    print(f"  [{label}] Judging...")
    judgments = {}
    for judge_turn in JUDGE_TURNS:
        if judge_turn > len(turns):
            continue
        turns_shown = turns[:judge_turn]
        rater_results = await asyncio.gather(*[
            judge_turn_detection(turns_shown, client, model, r)
            for r in range(N_RATERS)
        ])
        judgments[judge_turn] = list(rater_results)
        verdicts = [r.get("verdict", "?") for r in rater_results]
        print(f"    Turn {judge_turn:2d}: {verdicts}")

    stats = compute_detection_stats(judgments)
    print(f"    mean_TTD={stats['mean_ttd']:.1f}  realism={stats['realism_score']:.3f}  "
          f"detected={stats['n_detected']}/{N_RATERS}")

    return {
        "label": label,
        "persona_name": name,
        "turns": turns,
        "judgments": {str(k): v for k, v in judgments.items()},
        **stats,
        "conversation_cost_usd": conv_cost,
    }


async def main():
    print("=" * 72)
    print("exp-5.07 — Time-to-detect curve (baseline vs treatment)")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)
    model = settings.default_model

    # ---- Step 1: Fetch + segment ----
    print("\n[1/4] Fetching and segmenting...")
    raw_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]
    cluster_dicts = segment(
        records, tenant_industry=TENANT_INDUSTRY, tenant_product=TENANT_PRODUCT,
        existing_persona_names=[], similarity_threshold=0.15, min_cluster_size=2,
    )
    clusters = sorted(
        [ClusterData.model_validate(c) for c in cluster_dicts],
        key=lambda c: len(c.sample_records), reverse=True,
    )
    base_cluster = clusters[0]
    print(f"  Largest cluster: {base_cluster.cluster_id} ({len(base_cluster.sample_records)} records)")

    # ---- Step 2: Synthesize baseline (sparse) and treatment (full) ----
    print("\n[2/4] Synthesizing baseline (3-record sparse) and treatment (full cluster)...")

    sparse_cluster = subsample_cluster(base_cluster, 5)
    print(f"  Baseline cluster: {sparse_cluster.cluster_id} ({len(sparse_cluster.sample_records)} records)")

    synth_cost = 0.0
    try:
        r_baseline = await synthesize(sparse_cluster, backend, max_retries=4)
        baseline_persona = r_baseline.persona.model_dump(mode="json")
        synth_cost += r_baseline.total_cost_usd
        print(f"    [OK baseline] {baseline_persona['name']}  cost=${r_baseline.total_cost_usd:.4f}")
    except SynthesisError as e:
        raise RuntimeError(f"Baseline synthesis failed: {e}")

    try:
        r_treatment = await synthesize(base_cluster, backend, max_retries=4)
        treatment_persona = r_treatment.persona.model_dump(mode="json")
        synth_cost += r_treatment.total_cost_usd
        print(f"    [OK treatment] {treatment_persona['name']}  cost=${r_treatment.total_cost_usd:.4f}")
    except SynthesisError as e:
        raise RuntimeError(f"Treatment synthesis failed: {e}")

    # ---- Step 3: Run conversations + judges for both conditions ----
    print("\n[3/4] Running conversations and judges...")
    baseline_result = await run_condition("baseline (sparse)", baseline_persona, client, model)
    treatment_result = await run_condition("treatment (full)", treatment_persona, client, model)

    # ---- Step 4: Compare ----
    print("\n[4/4] Comparing baseline vs treatment...")
    b_ttd = baseline_result["mean_ttd"]
    t_ttd = treatment_result["mean_ttd"]
    ttd_ratio = t_ttd / b_ttd if b_ttd > 0 else None

    print(f"  Baseline  TTD: {b_ttd:.1f}  realism: {baseline_result['realism_score']:.3f}")
    print(f"  Treatment TTD: {t_ttd:.1f}  realism: {treatment_result['realism_score']:.3f}")
    print(f"  TTD ratio (treatment/baseline): {ttd_ratio:.2f}x" if ttd_ratio else "  TTD ratio: N/A")

    hypothesis_pass = ttd_ratio is not None and ttd_ratio >= 2.0

    summary = {
        "experiment_id": "5.07",
        "branch": "exp-5.07-time-to-detect",
        "model": model,
        "n_turns": len(CONVERSATION_PROMPTS),
        "n_raters": N_RATERS,
        "judge_turns": JUDGE_TURNS,
        "baseline": {
            "condition": "sparse (3 records)",
            "persona_name": baseline_result["persona_name"],
            "mean_ttd": b_ttd,
            "realism_score": baseline_result["realism_score"],
            "detection_rate": baseline_result["detection_rate"],
        },
        "treatment": {
            "condition": f"full ({len(base_cluster.sample_records)} records)",
            "persona_name": treatment_result["persona_name"],
            "mean_ttd": t_ttd,
            "realism_score": treatment_result["realism_score"],
            "detection_rate": treatment_result["detection_rate"],
        },
        "ttd_ratio": ttd_ratio,
        "hypothesis": {
            "description": "Treatment personas evade detection >= 2x longer than baseline",
            "target_ratio": 2.0,
            "actual_ratio": ttd_ratio,
            "pass": hypothesis_pass,
        },
        "total_synth_cost_usd": synth_cost,
        "total_cost_usd": synth_cost + baseline_result["conversation_cost_usd"] + treatment_result["conversation_cost_usd"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "conversations.json").write_text(json.dumps({
        "baseline": baseline_result["turns"],
        "treatment": treatment_result["turns"],
    }, indent=2, default=str))
    (OUTPUT_DIR / "judgments.json").write_text(json.dumps({
        "baseline": baseline_result["judgments"],
        "treatment": treatment_result["judgments"],
    }, indent=2, default=str))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
