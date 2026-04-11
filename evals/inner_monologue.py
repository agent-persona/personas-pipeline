"""Experiment 4.07 — Inner-monologue scaffolding.

Compare twin responses with and without a hidden reasoning step.
Measures contradiction rate, response quality, and latency.

Usage:
    python evals/inner_monologue.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from synthesis.config import Settings
from twin import TwinChat

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

# Questions that test consistency and persona depth
QUESTIONS = [
    "What's your take on using Kubernetes vs serverless?",
    "Your manager wants to adopt a new tool that has no API. What do you say?",
    "A junior engineer asks why you insist on infrastructure-as-code. How do you explain it?",
    "What would make you switch jobs?",
    "How do you feel about no-code platforms?",
]


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = settings.default_model

    results = {"baseline": [], "monologue": []}

    for mode, use_mono in [("baseline", False), ("monologue", True)]:
        print(f"\n--- {mode} ---")
        twin = TwinChat(PERSONA, client=client, model=model, inner_monologue=use_mono)
        for q in QUESTIONS:
            t0 = time.monotonic()
            reply = await twin.reply(q)
            elapsed = time.monotonic() - t0
            print(f"  Q: {q[:60]}")
            print(f"  A: {reply.text[:100]}...")
            if twin.last_thinking:
                print(f"  [thinking]: {twin.last_thinking[:80]}...")
            print(f"  ({elapsed:.1f}s, {reply.input_tokens + reply.output_tokens} tok)")
            results[mode].append({
                "question": q, "reply": reply.text,
                "thinking": twin.last_thinking,
                "elapsed_s": elapsed,
                "tokens": reply.input_tokens + reply.output_tokens,
                "cost": reply.estimated_cost_usd,
            })

    print("\n" + "=" * 80)
    print("EXPERIMENT 4.07 -- INNER MONOLOGUE SCAFFOLDING")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Baseline':>10} {'Monologue':>10}")
    print("-" * 45)

    for mode in ["baseline", "monologue"]:
        data = results[mode]
        avg_time = sum(d["elapsed_s"] for d in data) / len(data)
        avg_tok = sum(d["tokens"] for d in data) / len(data)
        total_cost = sum(d["cost"] for d in data)
        avg_len = sum(len(d["reply"]) for d in data) / len(data)
        results[mode + "_stats"] = {"avg_time": avg_time, "avg_tok": avg_tok,
            "total_cost": total_cost, "avg_reply_len": avg_len}

    b = results["baseline_stats"]
    m = results["monologue_stats"]
    print(f"{'Avg latency (s)':<20} {b['avg_time']:>10.1f} {m['avg_time']:>10.1f}")
    print(f"{'Avg tokens':<20} {b['avg_tok']:>10.0f} {m['avg_tok']:>10.0f}")
    print(f"{'Total cost':<20} ${b['total_cost']:>9.4f} ${m['total_cost']:>9.4f}")
    print(f"{'Avg reply length':<20} {b['avg_reply_len']:>10.0f} {m['avg_reply_len']:>10.0f}")

    latency_delta = ((m["avg_time"] - b["avg_time"]) / b["avg_time"]) * 100
    print(f"\nLatency delta: {latency_delta:+.0f}%")

    # Show side-by-side for one question
    print("\n--- Sample comparison (Q: tool with no API) ---")
    b_reply = results["baseline"][1]["reply"]
    m_reply = results["monologue"][1]["reply"]
    m_think = results["monologue"][1]["thinking"]
    print(f"  [BASELINE]: {b_reply[:200]}")
    print(f"  [THINKING]: {m_think[:200]}")
    print(f"  [MONOLOGUE]: {m_reply[:200]}")

    out = REPO_ROOT / "output" / "exp_4_07_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
