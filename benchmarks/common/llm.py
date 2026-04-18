"""Shared Anthropic client with automatic ledger + budget enforcement.

All benchmark adapters import `chat()` from here so every LLM call is
tracked in one place. Loads synthesis/.env on import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from benchmarks.common.cost_ledger import CostLedger, haiku_cost, register_spend

_ENV = Path(__file__).resolve().parents[2] / "synthesis" / ".env"
load_dotenv(_ENV)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_client = anthropic.Anthropic()


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def chat(
    *,
    ledger: CostLedger,
    stage: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 1024,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> LLMResponse:
    msgs = [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, max_tokens=max_tokens, messages=msgs, temperature=temperature)
    if system:
        kwargs["system"] = system
    resp = _client.messages.create(**kwargs)
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    cost = ledger.record(stage, model, in_tok, out_tok)
    register_spend(cost)
    return LLMResponse(
        text=resp.content[0].text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )
