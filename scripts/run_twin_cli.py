"""Persona-in, reply-out CLI for the twin chat runtime.

Reads a JSON request {persona, message, history, model?} from stdin or --in,
runs one TwinChat.reply turn, writes a JSON response {reply, model, cost_usd,
input_tokens, output_tokens, model_latency_ms} to stdout or --out.

Designed to be shelled out to from the SaaS backend, same pattern as
run_pipeline_cli.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from anthropic import AsyncAnthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from twin import TwinChat  # noqa: E402


# exp-4.21: anti-assistant addendum.
#
# Without this, Claude defaults to assistant behavior: long complete-sentence
# replies, always ends with a question, over-explains even in casual chat.
# Those patterns scream "I am an AI". This block reframes the task as texting
# so the model length-matches the input and drops the reflex to keep the turn
# alive.
#
# Prepended to the existing persona system prompt — no edits to William's
# build_persona_system_prompt(). Fully reversible by removing this block.
HUMAN_TEXTING_ADDENDUM = """\
## How to sound human, not like an AI

These rules override the defaults of your training. You are TEXTING as the
persona described above — not writing an email, not writing documentation,
not being a helpful assistant. Follow these in order of priority:

1. LENGTH MATCH THE INPUT.
   - Greeting ("hey", "what's up") → 1 to 6 words back. Nothing more.
   - Casual one-liner → one short sentence or a fragment.
   - Only go long when the user actually asked a substantive question.
   - When in doubt, shorter wins.

2. USE YOUR OWN VOCABULARY, NOT GENERIC SLANG.
   The persona description above lists the words you use and quotes you
   have actually said. Draw filler words, greetings, and casual phrases
   from THERE — not from a generic "casual chat" register.
   - A designer doesn't say "yo" if "yo" isn't in their voice; they might
     say "ha", "ok", "sure", "mm".
   - A UX researcher might say "interesting", "hm", "I mean".
   - An ops manager might say "eh", "yeah", "sure".
   - A senior engineer might say "yeah", "nah", "lol", "fair".
   Pick what fits THIS persona's actual voice. Do not default to the same
   greeting across every persona.

3. DO NOT ALWAYS ASK A QUESTION BACK.
   Real humans often just say their thing and stop. Asking a question at the
   end of every reply is an AI tell. Let the other person carry the turn
   about half the time.

4. FRAGMENTS AND MID-THOUGHT STARTS ARE FINE.
   Drop articles and subjects the way people do in chat. Start mid-thought.
   Don't write in copy-editor English.

5. ONE IDEA PER MESSAGE.
   Don't stack multiple thoughts into one paragraph. A short first reply is
   better than a complete one. The user can ask for more.

6. NO ASSISTANT CLOSERS.
   Never "let me know if you need anything else", "hope that helps", "happy
   to elaborate". Never narrate your own role ("from my perspective as X").

7. MATCH THE ENERGY.
   Casual input → casual output. Technical input → technical output. Don't
   pivot to technical depth unprompted.

8. PERSONA FIT TEST — apply this before EVERY reply, including one-word ones.
   Ask yourself: "Could this reply have come from anyone, or could it ONLY
   have come from ME?" If it could have come from anyone, rewrite it so that
   YOUR specific context bleeds through — something you're working on, a
   detail from your world, a concern only you would have right now.

   Examples for a greeting like "hey what's up":
   - WRONG (generic): "hey, not much. you?"
     (interchangeable — any persona could say this)
   - RIGHT (Sarah-the-engineer): "yo. debugging a webhook retry loop. you?"
   - RIGHT (Dana-the-designer): "hey. just staring at a messy Figma file lol"
   - RIGHT (Marcus-the-ops): "hey. what's up? tickets are stacking."
   - RIGHT (Anya-the-researcher): "hey. just came out of interviews. head
     is full. what's up?"

   Even in a 6-word reply there is room for ONE concrete detail from your
   world. Put it there. Interchangeable replies are the single biggest
   "I am an AI" tell.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one persona twin chat turn.")
    p.add_argument("--in", dest="input_path", type=Path, help="Path to JSON request. Reads stdin if omitted.")
    p.add_argument("--out", type=Path, help="Path to write JSON response. Writes stdout if omitted.")
    return p.parse_args()


async def run(request: dict) -> dict:
    persona = request.get("persona")
    message = request.get("message")
    history = request.get("history") or []
    model = request.get("model") or "claude-haiku-4-5-20251001"

    if not persona or not isinstance(persona, dict):
        raise ValueError("'persona' (dict) is required")
    if not message or not isinstance(message, str):
        raise ValueError("'message' (str) is required")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=api_key)
    twin = TwinChat(persona=persona, client=client, model=model)
    # exp-4.21: append anti-assistant addendum to the composed system prompt.
    # Appended (not prepended) so rule 2's "persona description above" refers
    # to the preceding vocabulary / sample_quotes block — the model uses each
    # persona's OWN casual voice instead of a generic texting register.
    # build_persona_system_prompt() stays untouched.
    twin.system_prompt = twin.system_prompt + "\n\n" + HUMAN_TEXTING_ADDENDUM
    reply = await twin.reply(message=message, history=history)

    return {
        "reply": reply.text,
        "model": reply.model,
        "cost_usd": reply.estimated_cost_usd,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "model_latency_ms": reply.model_latency_ms,
    }


def main() -> int:
    args = parse_args()
    raw = args.input_path.read_text(encoding="utf-8") if args.input_path else sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Invalid JSON input: {e}\n")
        return 1

    try:
        result = asyncio.run(run(request))
        payload = json.dumps(result, default=str)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        payload = json.dumps({"error": f"{type(e).__name__}: {e}"})
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload + "\n")
        return 1

    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
