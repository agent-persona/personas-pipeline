"""exp-5.07 — Time-to-detect curve.

Measures how many conversational turns it takes for LLM-simulated "raters"
to detect that a TwinChat persona is AI rather than human.

Hypothesis: High-quality personas evade detection 2-3x longer than baseline
personas. We simulate "human raters" with multiple independent LLM judge calls.

Approach:
  1. Fetch + segment + synthesize personas from first 3 clusters.
  2. For each persona, run a 15-turn conversation via TwinChat.
  3. At judge turns (1, 3, 5, 7, 10, 15), ask 5 independent LLM judges:
     "Is this responder a real person or an AI? Rate confidence 1-5."
  4. Detection point: first turn where verdict="AI" and confidence >= 4.
  5. Compute mean turns-to-detection per persona.
  6. Realism score: 1 - (1 / mean_turns_to_detect).

Usage:
    python scripts/run_exp_5_07.py
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
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.07"

N_CLUSTERS = 3
N_RATERS = 5
JUDGE_TURNS = [1, 3, 5, 7, 10, 15]  # 1-indexed turn numbers to judge

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


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

async def judge_turn_detection(
    turns_shown: list[dict],
    client: AsyncAnthropic,
    model: str,
    rater_seed: int,
) -> dict:
    """Simulate one rater judging whether the responder is human or AI."""
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

    # Add rater variation by tweaking temperature
    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.8,  # variation between "raters"
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


# ---------------------------------------------------------------------------
# Conversation runner
# ---------------------------------------------------------------------------

async def run_conversation(
    persona_dict: dict,
    client: AsyncAnthropic,
    model: str,
) -> list[dict]:
    """Run a 15-turn conversation with a TwinChat persona."""
    twin = TwinChat(persona_dict, client=client, model=model)
    history: list[dict] = []
    turns: list[dict] = []

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


# ---------------------------------------------------------------------------
# Detection analysis
# ---------------------------------------------------------------------------

def compute_detection_stats(
    judge_results: dict[int, list[dict]],
) -> dict:
    """Compute per-rater detection point and mean TTD.

    judge_results: {turn_number: [rater_0_result, rater_1_result, ...]}
    """
    rater_ttd: dict[int, int | None] = {}
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
    mean_ttd = sum(detected_turns) / len(detected_turns) if detected_turns else 16.0  # 16 = never detected
    realism_score = 1.0 - (1.0 / mean_ttd) if mean_ttd > 0 else 0.0
    detection_rate = len(detected_turns) / N_RATERS

    return {
        "rater_ttd": {str(k): v for k, v in rater_ttd.items()},
        "mean_ttd": mean_ttd,
        "realism_score": realism_score,
        "detection_rate": detection_rate,
        "n_detected": len(detected_turns),
        "n_raters": N_RATERS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 72)
    print("exp-5.07 -- Time-to-detect curve")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # ---- Step 1: Fetch + segment + synthesize ----
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
    print(f"  Got {len(clusters)} clusters, using first {N_CLUSTERS}")
    synth_clusters = clusters[:N_CLUSTERS]

    personas: list[dict] = []
    synth_results: list[dict] = []
    for i, cluster in enumerate(synth_clusters):
        print(f"  [{i + 1}/{len(synth_clusters)}] Synthesizing {cluster.cluster_id}...")
        try:
            r = await synthesize(cluster, backend)
            p_dict = r.persona.model_dump(mode="json")
            personas.append(p_dict)
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

    # ---- Step 2: Run 15-turn conversations ----
    print(f"\n[2/4] Running 15-turn conversations for {len(personas)} personas...")
    all_conversations: dict[str, list[dict]] = {}
    for p in personas:
        name = p["name"]
        print(f"  Conversing with {name}...")
        turns = await run_conversation(p, client, settings.default_model)
        all_conversations[name] = turns
        conv_cost = sum(t["cost_usd"] for t in turns)
        print(f"    Done: {len(turns)} turns, cost=${conv_cost:.4f}")

    # ---- Step 3: Judge at selected turns ----
    print(f"\n[3/4] Running judges at turns {JUDGE_TURNS} with {N_RATERS} raters each...")
    all_judgments: dict[str, dict] = {}  # persona_name -> {turn: [rater_results]}
    total_judge_cost = 0.0

    for p in personas:
        name = p["name"]
        turns = all_conversations[name]
        print(f"  Judging {name}...")
        persona_judgments: dict[int, list[dict]] = {}

        for judge_turn in JUDGE_TURNS:
            if judge_turn > len(turns):
                continue
            turns_shown = turns[:judge_turn]
            rater_tasks = [
                judge_turn_detection(turns_shown, client, settings.default_model, rater_seed=r)
                for r in range(N_RATERS)
            ]
            rater_results = await asyncio.gather(*rater_tasks)
            persona_judgments[judge_turn] = list(rater_results)

            # Summarize this turn
            verdicts = [r.get("verdict", "?") for r in rater_results]
            print(f"    Turn {judge_turn:2d}: {verdicts}")

        all_judgments[name] = {str(k): v for k, v in persona_judgments.items()}

    # ---- Step 4: Compute detection stats ----
    print("\n[4/4] Computing detection statistics...")
    persona_stats: dict[str, dict] = {}
    for p in personas:
        name = p["name"]
        # Convert string keys back to int for compute
        jdata = {int(k): v for k, v in all_judgments[name].items()}
        stats = compute_detection_stats(jdata)
        persona_stats[name] = stats
        print(
            f"  {name}: mean_TTD={stats['mean_ttd']:.1f}  "
            f"realism={stats['realism_score']:.3f}  "
            f"detection_rate={stats['detection_rate']:.0%}"
        )

    # ---- Compute aggregate stats ----
    all_ttds = [s["mean_ttd"] for s in persona_stats.values()]
    all_realism = [s["realism_score"] for s in persona_stats.values()]
    overall_mean_ttd = sum(all_ttds) / len(all_ttds) if all_ttds else 0
    overall_realism = sum(all_realism) / len(all_realism) if all_realism else 0

    total_synth_cost = sum(r.get("cost_usd", 0) for r in synth_results)
    total_conv_cost = sum(
        sum(t["cost_usd"] for t in turns) for turns in all_conversations.values()
    )

    summary = {
        "experiment_id": "5.07",
        "branch": "exp-5.07-time-to-detect",
        "model": settings.default_model,
        "n_personas": len(personas),
        "n_turns": len(CONVERSATION_PROMPTS),
        "n_raters": N_RATERS,
        "judge_turns": JUDGE_TURNS,
        "per_persona": persona_stats,
        "overall_mean_ttd": overall_mean_ttd,
        "overall_realism_score": overall_realism,
        "total_synth_cost_usd": total_synth_cost,
        "total_conversation_cost_usd": total_conv_cost,
        "total_cost_usd": total_synth_cost + total_conv_cost,
    }

    # ---- Write outputs ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "conversations.json").write_text(
        json.dumps(all_conversations, indent=2, default=str)
    )
    (OUTPUT_DIR / "judgments.json").write_text(
        json.dumps(all_judgments, indent=2, default=str)
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
