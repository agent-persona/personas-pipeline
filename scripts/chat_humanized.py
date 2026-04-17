"""Interactive local twin chat with the humanization layer on.

This is the closest thing to running the app locally. You type, the twin
replies with real async timing between chunks — you see the typing pauses,
the grammar slips, the separate filler messages, exactly as a Discord user
would experience them.

Setup:
  1. cp synthesis/.env.example synthesis/.env   (already done by default)
  2. Edit synthesis/.env and paste your ANTHROPIC_API_KEY after the =
  3. python3 scripts/chat_humanized.py

Usage:
  python3 scripts/chat_humanized.py                  # default: Alex fixture
  python3 scripts/chat_humanized.py --persona maya   # or maya, riley, jordan
  python3 scripts/chat_humanized.py --no-humanize    # disable for A/B comparison
  python3 scripts/chat_humanized.py --seed 42        # deterministic (for demos)

At the prompt:
  type a message, hit enter
  /switch <name>   — swap persona mid-session
  /humanize on|off — toggle the post-processor
  /quit            — exit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "twin"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from twin.chat import TwinChat  # noqa: E402
from twin.humanize import HumanizeConfig  # noqa: E402


# Personas you can pick from the CLI. Real fixtures + the two synthetic
# contrast personas from the offline smoke script.
def _load_fixture(name: str) -> dict:
    return json.loads((REPO / "tests" / "fixtures" / f"{name}.json").read_text())["persona"]


def _synthetic_riley() -> dict:
    return {
        "name": "Riley the Community Streamer",
        "summary": "A Twitch streamer who built a 10k-person Discord around indie games.",
        "demographics": {"age_range": "22-28", "gender_distribution": "any",
                         "location_signals": ["US"]},
        "firmographics": {},
        "goals": ["Grow the community", "Ship good streams"],
        "pains": ["Mods burning out", "Trolls during raids"],
        "motivations": ["Community first"],
        "objections": ["Not selling out to sponsors"],
        "not_this": ["Never scold a new member", "Never pretend to know something I don't"],
        "vocabulary": ["vibe", "based", "goated", "wild", "fr"],
        "sample_quotes": ["that's so goated", "the vibes are immaculate"],
        "communication_style": {
            "tone": "warm", "formality": "casual", "vocabulary_level": "basic",
            "preferred_channels": ["Discord", "Twitch chat"],
        },
        "emotional_profile": {
            "baseline_mood": "enthusiastic",
            "stress_triggers": ["stream going offline"],
            "coping_mechanisms": ["memes"],
        },
    }


def _synthetic_jordan() -> dict:
    return {
        "name": "Jordan the New PM",
        "summary": "A first-time product manager anxious about shipping on time.",
        "demographics": {"age_range": "26-32", "gender_distribution": "any",
                         "location_signals": ["US"]},
        "firmographics": {"company_size": "50-200", "industry": "SaaS",
                          "role_titles": ["PM"]},
        "goals": ["Ship Q2 roadmap on time"],
        "pains": ["Engineers pushing back", "Scope creep"],
        "motivations": ["Prove I can do this job"],
        "objections": ["Last-minute requirements"],
        "not_this": ["Never commit without checking with eng"],
        "vocabulary": ["blocker", "scope", "risk", "mitigation", "eng"],
        "sample_quotes": ["what's the blocker?", "let's mitigate that"],
        "communication_style": {
            "tone": "analytical", "formality": "professional", "vocabulary_level": "intermediate",
            "preferred_channels": ["Slack"],
        },
        "emotional_profile": {
            "baseline_mood": "anxious",
            "stress_triggers": ["missed deadlines"],
            "coping_mechanisms": ["over-documents"],
        },
    }


PERSONAS: dict[str, callable] = {
    "alex":   lambda: _load_fixture("persona_00"),
    "maya":   lambda: _load_fixture("persona_01"),
    "riley":  _synthetic_riley,
    "jordan": _synthetic_jordan,
}


def _describe(p: dict) -> str:
    cs = p.get("communication_style", {})
    ep = p.get("emotional_profile", {})
    return (
        f"{p['name']}\n"
        f"  formality={cs.get('formality')}  vocab={cs.get('vocabulary_level')}  "
        f"tone={cs.get('tone')}  mood={ep.get('baseline_mood')}"
    )


async def send_chunks(reply) -> None:
    """Play back chunks with their real timing, one by one."""
    if not reply.human_chunks:
        # Humanization off — just dump the whole reply
        print(f"  {reply.text}\n")
        return

    for i, chunk in enumerate(reply.human_chunks, 1):
        # pre_delay includes typing time; show a "typing..." indicator while we wait
        sys.stdout.write(f"  [{i}] (thinking/typing for {chunk.pre_delay_ms/1000:.1f}s) ")
        sys.stdout.flush()
        await asyncio.sleep(chunk.pre_delay_ms / 1000.0)
        # Carriage return + clear, then print the actual chunk
        sys.stdout.write("\r" + " " * 60 + "\r")
        print(f"  [{i}] {chunk.text}")
    print()


async def repl(args: argparse.Namespace) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not found.\n"
            "  1. cp synthesis/.env.example synthesis/.env\n"
            "  2. Edit synthesis/.env and paste your key after ANTHROPIC_API_KEY=\n"
            "  3. Re-run this script.",
            file=sys.stderr,
        )
        return 1

    model = os.environ.get("default_model", "claude-haiku-4-5-20251001")
    client = AsyncAnthropic(api_key=api_key)

    persona_key = args.persona.lower()
    if persona_key not in PERSONAS:
        print(f"Unknown persona {persona_key!r}. Options: {', '.join(PERSONAS)}",
              file=sys.stderr)
        return 1
    persona = PERSONAS[persona_key]()

    humanize_on = not args.no_humanize
    cfg = HumanizeConfig(seed=args.seed) if humanize_on else None
    twin = TwinChat(persona, client=client, model=model, humanize_config=cfg)
    history: list[dict] = []

    print(f"\nModel: {model}   Humanize: {'ON' if humanize_on else 'OFF'}"
          f"   Seed: {args.seed or 'random'}")
    print("Persona:")
    print("  " + _describe(persona).replace("\n", "\n  "))
    print("\nCommands: /switch <name>, /humanize on|off, /quit\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/quit":
            return 0
        if user.startswith("/switch "):
            name = user.split(maxsplit=1)[1].lower()
            if name not in PERSONAS:
                print(f"  unknown persona. options: {', '.join(PERSONAS)}")
                continue
            persona = PERSONAS[name]()
            twin = TwinChat(persona, client=client, model=model, humanize_config=cfg)
            history = []
            print(f"  switched to {persona['name']}. history cleared.")
            print("  " + _describe(persona).replace("\n", "\n  "))
            continue
        if user == "/humanize on":
            cfg = HumanizeConfig(seed=args.seed)
            twin = TwinChat(persona, client=client, model=model, humanize_config=cfg)
            print("  humanize ON")
            continue
        if user == "/humanize off":
            cfg = None
            twin = TwinChat(persona, client=client, model=model, humanize_config=cfg)
            print("  humanize OFF")
            continue

        t0 = time.monotonic()
        reply = await twin.reply(user, history=history)
        dt = time.monotonic() - t0
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply.text})

        print(f"{persona['name']}> (model {dt:.1f}s, {reply.input_tokens}in/{reply.output_tokens}out)")
        await send_chunks(reply)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="alex", help="alex | maya | riley | jordan")
    ap.add_argument("--no-humanize", action="store_true",
                    help="Disable post-processor (for A/B comparison)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Deterministic humanization (same rolls every run)")
    args = ap.parse_args()
    return asyncio.run(repl(args))


if __name__ == "__main__":
    raise SystemExit(main())
