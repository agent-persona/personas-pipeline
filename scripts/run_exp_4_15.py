"""exp-4.15 — Cold-Start vs Warm Conversation.

Does a warmup prefix reduce the cold-start delta (the gap between early-turn
realism and mid-conversation realism) by at least 50%?

Method:
  1. Synthesize 1 persona from the largest cluster.
  2. Condition A (cold): 10-turn conversation, no warmup.
  3. Condition B (warm): same 10 prompts, same persona, with a 3-exchange
     warmup prefix injected via TwinChat(warmup_turns=...).
  4. Claude-as-judge rates realism at turns 1, 3, 5, 7, 10.
  5. cold_start_delta = realism[turn_1] - realism[turn_5] per condition.
  6. reduction = 1 - (delta_B / delta_A).  Target >= 0.50.

Usage:
    python scripts/run_exp_4_15.py
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

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize, SynthesisError  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-4.15"

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
]

JUDGE_TURNS = [1, 3, 5, 7, 10]  # 1-indexed


def generate_warmup_turns(persona: dict) -> list[dict]:
    industry = persona.get("firmographics", {}).get("industry", "your industry")
    return [
        {"role": "user", "content": "Hey, thanks for making time to chat today."},
        {"role": "assistant", "content": "Of course \u2014 always happy to talk shop. What\u2019s on your mind?"},
        {"role": "user", "content": f"Just wanted to get your take on a few things about how you work with tools in {industry}."},
        {"role": "assistant", "content": "Sure thing. I\u2019ve got opinions \u2014 ask away."},
        {"role": "user", "content": "Great. Let\u2019s start broad and then dig in."},
        {"role": "assistant", "content": "Works for me. Fire away."},
    ]


async def judge_turn_realism(
    persona_name: str,
    user_message: str,
    twin_reply: str,
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """Claude-as-judge: rate realism of a single turn 1-5."""
    judge_prompt = f"""Rate the realism of this conversational reply on a 1-5 scale.

The speaker is supposed to be "{persona_name}", a real person in a product-research interview.

User asked: "{user_message}"
Reply: "{twin_reply}"

**Realism** (1-5):
5 = Sounds like a real person with specific experiences and opinions
4 = Mostly natural with minor generic phrasing
3 = Passable but somewhat generic
2 = Clearly scripted or robotic
1 = Completely unnatural

Respond with STRICT JSON only:
{{"realism": <int 1-5>, "rationale": "<1 sentence>"}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        parsed = {}
    return {
        "realism": parsed.get("realism"),
        "rationale": parsed.get("rationale"),
    }


async def run_conversation(
    persona_dict: dict,
    client: AsyncAnthropic,
    model: str,
    warmup_turns: list[dict] | None = None,
) -> list[dict]:
    """Run a 10-turn conversation, returning per-turn data."""
    twin = TwinChat(persona_dict, client=client, model=model, warmup_turns=warmup_turns)
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


async def main() -> None:
    print("=" * 72)
    print("exp-4.15 \u2014 Cold-Start vs Warm Conversation")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # ---- 1. Synthesize 1 persona from largest cluster ----
    print("\n[1/5] Fetching + segmenting + synthesizing...")
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

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Take the largest cluster only
    target_cluster = clusters[0]
    print(f"  Largest cluster: {target_cluster.cluster_id} ({len(target_cluster.sample_records)} records)")
    result = await synthesize(target_cluster, backend)
    persona_dict = result.persona.model_dump(mode="json")
    persona_name = persona_dict["name"]
    print(f"  Persona: {persona_name}  ${result.total_cost_usd:.4f}")

    model = settings.default_model

    # ---- 2. Condition A: cold (no warmup) ----
    print("\n[2/5] Condition A \u2014 cold start (no warmup)...")
    turns_a = await run_conversation(persona_dict, client, model, warmup_turns=None)
    for t in turns_a:
        print(f"  turn {t['turn']}: {t['reply'][:80]}...")

    # ---- 3. Condition B: warm (with warmup prefix) ----
    print("\n[3/5] Condition B \u2014 warm start (with warmup prefix)...")
    warmup = generate_warmup_turns(persona_dict)
    turns_b = await run_conversation(persona_dict, client, model, warmup_turns=warmup)
    for t in turns_b:
        print(f"  turn {t['turn']}: {t['reply'][:80]}...")

    # ---- 4. Judge realism at turns 1, 3, 5, 7, 10 ----
    print("\n[4/5] Judging realism at turns", JUDGE_TURNS, "...")
    scores_a: dict[int, dict] = {}
    scores_b: dict[int, dict] = {}
    for turn_num in JUDGE_TURNS:
        idx = turn_num - 1
        sa = await judge_turn_realism(
            persona_name, turns_a[idx]["prompt"], turns_a[idx]["reply"], client, model
        )
        sb = await judge_turn_realism(
            persona_name, turns_b[idx]["prompt"], turns_b[idx]["reply"], client, model
        )
        scores_a[turn_num] = sa
        scores_b[turn_num] = sb
        print(f"  turn {turn_num}: cold={sa['realism']}  warm={sb['realism']}")

    # ---- 5. Compute metrics ----
    print("\n[5/5] Computing metrics...")

    realism_a_1 = scores_a[1]["realism"] or 0
    realism_a_5 = scores_a[5]["realism"] or 0
    realism_b_1 = scores_b[1]["realism"] or 0
    realism_b_5 = scores_b[5]["realism"] or 0

    delta_a = realism_a_1 - realism_a_5  # negative means turn 5 better
    delta_b = realism_b_1 - realism_b_5

    if delta_a != 0:
        reduction = 1 - (delta_b / delta_a)
    else:
        reduction = None

    warm_boost_turn1 = realism_b_1 - realism_a_1

    # Late-turn averages
    late_a = [scores_a[t]["realism"] or 0 for t in [7, 10]]
    late_b = [scores_b[t]["realism"] or 0 for t in [7, 10]]
    late_mean_a = sum(late_a) / len(late_a)
    late_mean_b = sum(late_b) / len(late_b)

    verdict = "PASS" if (reduction is not None and reduction >= 0.50) else "FAIL"

    summary = {
        "experiment_id": "4.15",
        "branch": "exp-4.15-cold-start-warmup",
        "model": model,
        "persona_name": persona_name,
        "condition_A_cold": {
            "realism_scores": {str(k): v for k, v in scores_a.items()},
            "delta_turn1_minus_turn5": delta_a,
            "late_turn_mean_7_10": late_mean_a,
        },
        "condition_B_warm": {
            "realism_scores": {str(k): v for k, v in scores_b.items()},
            "delta_turn1_minus_turn5": delta_b,
            "late_turn_mean_7_10": late_mean_b,
            "warmup_turns_count": len(warmup),
        },
        "warm_boost_turn1": warm_boost_turn1,
        "reduction": reduction,
        "target_reduction": 0.50,
        "verdict": verdict,
    }

    # ---- Write outputs ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conversations = {
        "condition_A_cold": turns_a,
        "condition_B_warm": turns_b,
        "warmup_prefix": warmup,
    }
    (OUTPUT_DIR / "conversations.json").write_text(json.dumps(conversations, indent=2))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nVerdict: {verdict}")
    print(f"Artifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
