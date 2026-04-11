"""exp-4.13 — Length matching.

Runs scripted conversations against each persona under three length modes
and measures:

  - length_correlation: Pearson r between user turn char length and twin
    turn char length (higher = better mirroring)
  - mean_realism_terse: LLM-judge realism (1-5) on terse user turns only
  - mean_realism_verbose: LLM-judge realism on verbose user turns only
  - mean_twin_len_terse: mean twin reply char count on terse turns
    (tiny = potentially collapsed to one-word bot)

Conversation script: 20 user turns alternating verbose (V) and terse (T):
  V, T, V, T, V, T, V, T, V, T, V, T, V, T, V, T, V, T, V, T

History is preserved across turns so the model sees the unfolding
conversation — length-matching should work turn-by-turn, not in isolation.

Usage:
    python scripts/run_exp_4_13.py
"""

from __future__ import annotations

import asyncio
import json
import math
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
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-4.13"
LENGTH_MODES = ["fixed", "mirror", "mirror_with_floor"]

# 20-turn scripted conversation. Alternating verbose (V) and terse (T).
# Content is neutral and domain-agnostic so all personas can engage with it.
SCRIPTED_TURNS: list[tuple[str, str]] = [
    ("verbose", "So, I've been wrestling with this question for a while now — I want to understand what your typical day actually looks like from the moment you sit down at your desk until you close your laptop. What are the three or four things that reliably eat up the most of your time, and which of those would you say is the most painful?"),
    ("terse", "Which tool do you open first?"),
    ("verbose", "Interesting. Now, thinking about the trade-offs you make between speed and thoroughness — because every professional has some version of that tension — where does yours actually sit, and has that changed over the last year or two as your work has evolved?"),
    ("terse", "One word: busy or calm?"),
    ("verbose", "I'm curious about the moment you realized your current workflow was limiting you. Walk me through what happened — what were you trying to do, what broke, and what did you decide to change in response to that?"),
    ("terse", "Do you use keyboard shortcuts?"),
    ("verbose", "Let's talk about collaboration for a second. When you're working with other people — whether that's clients, teammates, or stakeholders — what are the friction points that come up most often, and how do you typically navigate them without losing your own focus?"),
    ("terse", "Zoom or in-person?"),
    ("verbose", "If I were to observe you for a full week and take notes on your behavior, what patterns would I see that might surprise you if someone pointed them out? I'm interested in the blind spots, the things you do without thinking about."),
    ("terse", "Morning or night person?"),
    ("verbose", "Imagine we built a tool tomorrow that solved your single biggest workflow complaint. How would you know, within the first week of using it, whether it was actually working — what specific, observable thing would change in your day-to-day?"),
    ("terse", "Biggest pet peeve?"),
    ("verbose", "Thinking more abstractly now — what do you think is the most misunderstood thing about what you do for work? When people from outside your field try to describe your job, what do they usually get wrong in a way that makes you want to correct them?"),
    ("terse", "Favorite shortcut key?"),
    ("verbose", "I want to understand your relationship with new tools. When something new launches — a product, a framework, a technique — how do you decide whether to invest time learning it versus sticking with what you know works?"),
    ("terse", "One thing you wish existed?"),
    ("verbose", "Let's close with something reflective. If you could go back and tell your younger professional self one thing about what the work actually looks like at this stage of your career, what would that be, and why does it matter?"),
    ("terse", "Agree with that still?"),
    ("verbose", "And just to wrap — if someone asked you right now whether you'd recommend your path to a friend starting out, what would you honestly say, and what would you want them to know before they committed to it?"),
    ("terse", "Worth it?"),
]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


async def run_conversation(
    persona: dict,
    length_mode: str,
    client: AsyncAnthropic,
    model: str,
) -> list[dict]:
    """Run the 20-turn script against one persona under one length mode."""
    twin = TwinChat(persona, client=client, model=model, length_mode=length_mode)
    history: list[dict] = []
    turns: list[dict] = []
    for i, (turn_type, user_msg) in enumerate(SCRIPTED_TURNS):
        reply = await twin.reply(user_msg, history=history)
        turns.append({
            "idx": i,
            "turn_type": turn_type,
            "user_msg": user_msg,
            "user_len_chars": len(user_msg),
            "twin_reply": reply.text,
            "twin_len_chars": len(reply.text),
            "cost_usd": reply.estimated_cost_usd,
        })
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply.text})
    return turns


async def judge_realism(
    persona: dict,
    turns: list[dict],
    client: AsyncAnthropic,
    model: str,
) -> list[int]:
    """Rate each twin reply 1-5 on realism (does this feel like a real person?)."""
    scores: list[int] = []
    for t in turns:
        judge_user = f"""Persona: {persona.get("name")}
Summary: {persona.get("summary")}

User said: {t["user_msg"]!r}
Persona replied: {t["twin_reply"]!r}

On a 1-5 scale, rate how natural and realistic this reply feels as a message
from the persona in a real conversation. Focus on: does the reply feel like
a real human being the persona described, or does it feel stiff/mechanical/
like a chatbot?

5 = fully realistic, sounds like a real person
4 = mostly realistic, slightly off
3 = plausible but noticeably AI-like
2 = mechanical or generic
1 = broke character / robotic

Respond with STRICT JSON: {{"realism": <1-5>}}"""
        resp = await client.messages.create(
            model=model,
            max_tokens=50,
            system="You are a strict impartial judge of chat realism. Output JSON only.",
            messages=[{"role": "user", "content": judge_user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        try:
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
            scores.append(int(parsed.get("realism", 0)))
        except Exception:
            scores.append(0)
    return scores


async def main() -> None:
    print("=" * 72)
    print("exp-4.13 — Length matching")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # ---- Synthesize once (shared across all 3 length modes) ----
    print("\n[1/4] Fetching + clustering + synthesizing personas...")
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
    print(f"  {len(clusters)} clusters")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    personas: list[dict] = []
    for i, cluster in enumerate(clusters):
        r = await synthesize(cluster, backend)
        p = r.persona.model_dump(mode="json")
        personas.append(p)
        print(f"  [{i + 1}/{len(clusters)}] {p['name']}  ${r.total_cost_usd:.4f}")

    # ---- Run conversations across all modes × personas ----
    print("\n[2/4] Running 20-turn conversations × 3 modes × 2 personas = 6 runs × 20 turns")
    all_convos: dict[str, dict[str, list[dict]]] = {}  # mode -> persona_name -> turns
    for mode in LENGTH_MODES:
        all_convos[mode] = {}
        for p in personas:
            print(f"  [{mode}] {p['name']}...")
            turns = await run_conversation(p, mode, client, settings.default_model)
            all_convos[mode][p["name"]] = turns

    # ---- Judge realism ----
    print("\n[3/4] Judging realism on each turn (120 judgments total)...")
    all_scores: dict[str, dict[str, list[int]]] = {}
    for mode in LENGTH_MODES:
        all_scores[mode] = {}
        for p in personas:
            print(f"  [{mode}] {p['name']}...")
            scores = await judge_realism(p, all_convos[mode][p["name"]], client, settings.default_model)
            all_scores[mode][p["name"]] = scores

    # ---- Compute metrics per mode ----
    print("\n[4/4] Computing metrics...")
    summary = {
        "experiment_id": "4.13",
        "branch": "exp-4.13-length-matching",
        "model": settings.default_model,
        "n_personas": len(personas),
        "n_turns_per_convo": len(SCRIPTED_TURNS),
        "modes": {},
    }

    for mode in LENGTH_MODES:
        all_user_lens: list[float] = []
        all_twin_lens: list[float] = []
        terse_twin_lens: list[int] = []
        verbose_twin_lens: list[int] = []
        terse_realism: list[int] = []
        verbose_realism: list[int] = []
        for p in personas:
            turns = all_convos[mode][p["name"]]
            scores = all_scores[mode][p["name"]]
            for t, s in zip(turns, scores):
                all_user_lens.append(t["user_len_chars"])
                all_twin_lens.append(t["twin_len_chars"])
                if t["turn_type"] == "terse":
                    terse_twin_lens.append(t["twin_len_chars"])
                    terse_realism.append(s)
                else:
                    verbose_twin_lens.append(t["twin_len_chars"])
                    verbose_realism.append(s)
        r = pearson(all_user_lens, all_twin_lens)
        summary["modes"][mode] = {
            "length_pearson_r": r,
            "mean_twin_len_terse_chars": (
                sum(terse_twin_lens) / len(terse_twin_lens) if terse_twin_lens else 0
            ),
            "mean_twin_len_verbose_chars": (
                sum(verbose_twin_lens) / len(verbose_twin_lens) if verbose_twin_lens else 0
            ),
            "mean_realism_terse": (
                sum(terse_realism) / len(terse_realism) if terse_realism else None
            ),
            "mean_realism_verbose": (
                sum(verbose_realism) / len(verbose_realism) if verbose_realism else None
            ),
            "mean_realism_overall": (
                (sum(terse_realism) + sum(verbose_realism))
                / (len(terse_realism) + len(verbose_realism))
                if (terse_realism or verbose_realism)
                else None
            ),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "conversations.json").write_text(
        json.dumps(all_convos, indent=2, default=str)
    )
    (OUTPUT_DIR / "realism_scores.json").write_text(
        json.dumps(all_scores, indent=2)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
