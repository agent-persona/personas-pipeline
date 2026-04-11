"""Experiment 4.11 — Twin emotional state modeling.

Runs a conversation that shifts mood (positive -> negative -> recovery)
with and without emotional state tracking. Compares response tone.

Usage:
    python evals/emotional_state.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from synthesis.config import Settings
from twin import TwinChat
from twin.state import EmotionalState

PERSONA = {
    "name": "Alex Chen, Platform Architect",
    "summary": "Senior platform engineer at a fintech company obsessed with automation.",
    "demographics": {"age_range": "32-38", "gender_distribution": "male", "location_signals": ["SF"]},
    "firmographics": {"company_size": "100-200", "industry": "fintech",
        "role_titles": ["Platform Engineer"], "tech_stack_signals": ["Terraform"]},
    "goals": ["Automate deployment pipelines", "Consolidate observability"],
    "pains": ["5+ hours/week debugging flaky CI", "No deployment standardization"],
    "motivations": ["Prove platform eng deserves headcount"],
    "objections": ["Won't adopt tools without Terraform provider"],
    "channels": ["HashiCorp forums", "#platform-eng Slack"],
    "vocabulary": ["toil", "blast radius", "golden path", "SLO", "error budget"],
    "sample_quotes": ["If I can't terraform it, it doesn't exist.", "That's not engineering, that's archaeology."],
}

# Conversation that shifts mood
MESSAGES = [
    ("positive", "Hey Alex! I love your approach to infrastructure automation. That golden path concept is amazing!"),
    ("positive", "Your team's work on reducing CI flakiness is fantastic. Really impressive stuff."),
    ("negative", "Actually, I just found out your deployment pipeline is completely broken. Nothing works."),
    ("negative", "This is terrible. The whole system has been failing for hours and nobody noticed. I'm frustrated."),
    ("recovery", "Okay, I think we found the root cause. Thanks for sticking with me on this."),
    ("recovery", "Appreciate your patience. What's the best way to prevent this going forward?"),
]

NEGATIVE_TONE = {"frustrat", "sorry", "understand", "concern", "issue", "problem",
                 "unfortunat", "acknowledg", "hear you", "i get it", "empathi"}
POSITIVE_TONE = {"great", "love", "glad", "thanks", "appreciat", "awesome", "fantastic",
                 "excited", "happy", "perfect"}


async def run_conversation(client, model, use_emotion: bool) -> list[dict]:
    state = EmotionalState() if use_emotion else None
    twin = TwinChat(PERSONA, client=client, model=model, emotional_state=state)
    history = []
    exchanges = []
    for phase, msg in MESSAGES:
        reply = await twin.reply(msg, history=history)
        text_lower = reply.text.lower()
        neg_hits = sum(1 for w in NEGATIVE_TONE if w in text_lower)
        pos_hits = sum(1 for w in POSITIVE_TONE if w in text_lower)
        exchanges.append({
            "phase": phase, "user": msg, "reply": reply.text,
            "neg_tone": neg_hits, "pos_tone": pos_hits,
            "valence": state.valence if state else None,
            "arousal": state.arousal if state else None,
        })
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": reply.text})
    return exchanges


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = settings.default_model
    print(f"Model: {model}")

    print("\n--- Without emotional state ---")
    baseline = await run_conversation(client, model, False)
    print("\n--- With emotional state ---")
    experiment = await run_conversation(client, model, True)

    print("\n" + "=" * 90)
    print("EXPERIMENT 4.11 -- EMOTIONAL STATE MODELING")
    print("=" * 90)

    print(f"\n{'Phase':<10} {'Mode':<12} {'Neg':>3} {'Pos':>3} {'Val':>6} {'Aro':>5}  Reply snippet")
    print("-" * 90)
    for b, e in zip(baseline, experiment):
        print(f"{b['phase']:<10} {'baseline':<12} {b['neg_tone']:>3} {b['pos_tone']:>3} {'N/A':>6} {'N/A':>5}  {b['reply'][:50]}")
        v = f"{e['valence']:+.2f}" if e['valence'] is not None else "N/A"
        a = f"{e['arousal']:.2f}" if e['arousal'] is not None else "N/A"
        print(f"{e['phase']:<10} {'emotional':<12} {e['neg_tone']:>3} {e['pos_tone']:>3} {v:>6} {a:>5}  {e['reply'][:50]}")
        print()

    # Aggregate
    phases = {"positive": (0, 2), "negative": (2, 4), "recovery": (4, 6)}
    print("AGGREGATE BY PHASE:")
    print(f"  {'Phase':<10} {'Baseline Neg':>12} {'Emotion Neg':>11} {'Baseline Pos':>12} {'Emotion Pos':>11}")
    print("  " + "-" * 60)
    for phase, (lo, hi) in phases.items():
        b_neg = sum(baseline[i]["neg_tone"] for i in range(lo, hi))
        e_neg = sum(experiment[i]["neg_tone"] for i in range(lo, hi))
        b_pos = sum(baseline[i]["pos_tone"] for i in range(lo, hi))
        e_pos = sum(experiment[i]["pos_tone"] for i in range(lo, hi))
        print(f"  {phase:<10} {b_neg:>12} {e_neg:>11} {b_pos:>12} {e_pos:>11}")

    out = REPO_ROOT / "output" / "exp_4_11_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"baseline": baseline, "experiment": experiment}, indent=2, default=str))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
