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
