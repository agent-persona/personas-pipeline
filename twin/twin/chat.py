from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from anthropic import AsyncAnthropic


@dataclass
class TwinReply:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    # exp-4.14: wall-clock timing so callers can measure perceived latency
    model_latency_ms: int = 0  # time spent inside the Anthropic API call
    artificial_delay_ms: int = 0  # additional sleep injected before return
    total_latency_ms: int = 0  # model_latency_ms + artificial_delay_ms

    @property
    def estimated_cost_usd(self) -> float:
        if "opus" in self.model:
            return (self.input_tokens * 15 + self.output_tokens * 75) / 1_000_000
        if "haiku" in self.model:
            return (self.input_tokens * 1 + self.output_tokens * 5) / 1_000_000
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


def _verbatim_samples_block(samples: list) -> list[str]:
    """Render the voice-grounding section of the system prompt.

    Returns an empty list when no samples are present — the caller splices
    this into the prompt's line list, so an empty return is a clean no-op
    that preserves prior prompt output byte-for-byte.

    When present, frames the samples as "match this register" (not as
    "examples this persona has said") so the model imitates register,
    length, punctuation, and rhythm directly rather than treating them as
    optional flavor to draw from.
    """
    if not samples:
        return []
    lines = [
        "## How your messages actually read (match this register)",
        "The quotes below are VERBATIM text from messages this kind of "
        "person actually writes. Match their register, length, punctuation, "
        "casing, and rhythm when you reply. Do not clean them up; do not "
        "default to assistant-register.",
    ]
    lines.extend(f"- {s}" for s in samples)
    lines.append("")
    return lines


def build_persona_system_prompt(persona: dict) -> str:
    """Construct a system prompt that puts Claude in character as the persona.

    Accepts a persona dict (the output of `PersonaV1.model_dump()`) so the
    twin runtime has no compile-time dependency on the synthesis package.
    """
    name = persona.get("name", "the persona")
    summary = persona.get("summary", "")
    demo = persona.get("demographics", {})
    firmo = persona.get("firmographics", {})
    goals = persona.get("goals", [])
    pains = persona.get("pains", [])
    motivations = persona.get("motivations", [])
    objections = persona.get("objections", [])
    not_this = persona.get("not_this", [])
    vocabulary = persona.get("vocabulary", [])
    sample_quotes = persona.get("sample_quotes", [])
    verbatim_samples = persona.get("verbatim_samples") or []

    not_this_block: list[str] = []
    if not_this:
        not_this_block = [
            "## Things you would never do or say",
            *(f"- {n}" for n in not_this),
            "",
        ]

    lines = [
        f"You are {name}. Stay in character at all times.",
        "",
        f"## About you",
        summary,
        "",
        "## Background",
        f"- Age: {demo.get('age_range', 'unknown')}",
        f"- Location: {', '.join(demo.get('location_signals', []))}",
        f"- Industry: {firmo.get('industry', 'unknown')}",
        f"- Company size: {firmo.get('company_size', 'unknown')}",
        f"- Roles: {', '.join(firmo.get('role_titles', []))}",
        "",
        "## What you want",
        *(f"- {g}" for g in goals),
        "",
        "## What frustrates you",
        *(f"- {p}" for p in pains),
        "",
        "## What drives you",
        *(f"- {m}" for m in motivations),
        "",
        "## Things you would push back on",
        *(f"- {o}" for o in objections),
        "",
        *not_this_block,
        "## How you talk",
        f"You use words like: {', '.join(vocabulary)}.",
        "",
        "Examples of things you have said before:",
        *(f'- "{q}"' for q in sample_quotes),
        "",
        # Voice grounding: verbatim text carried from segmentation, not
        # LLM-rewritten. Shown as "match this register" rather than "examples"
        # so the model imitates register/length/rhythm directly rather than
        # treating it as further persona-flavoring hints. Skipped when absent
        # (personas from non-text-bearing corpora).
        *(_verbatim_samples_block(verbatim_samples)),
        "## Rules",
        "- Answer in first person, in character.",
        "- Use your vocabulary naturally — don't sound like a chatbot.",
        "- Keep responses under 4 sentences unless asked to elaborate.",
        "- If asked to do something that conflicts with your 'never do or say' "
        "list, refuse or redirect in character rather than complying out of "
        "politeness.",
        "- If asked something outside your knowledge or experience, react the way "
        "this persona would (curiosity, dismissal, deflection — whatever fits).",
        "- Do not break character to mention you are an AI.",
    ]
    return "\n".join(lines)


class TwinChat:
    """Stateless persona-driven chat. Caller manages history.

    exp-4.14: supports an `artificial_delay_ms` parameter that injects a
    pre-return sleep so the caller experiences a configurable total latency.
    Accepts either an int (constant delay) or a callable
    (reply_text, output_tokens) -> int for length-scaled delays.
    """

    def __init__(
        self,
        persona: dict,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
        artificial_delay_ms: int | Callable[[str, int], int] | None = None,
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.artificial_delay_ms = artificial_delay_ms
        self.system_prompt = build_persona_system_prompt(persona)

    def _resolve_delay_ms(self, text: str, output_tokens: int) -> int:
        d = self.artificial_delay_ms
        if d is None:
            return 0
        if callable(d):
            return max(0, int(d(text, output_tokens)))
        return max(0, int(d))

    async def reply(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> TwinReply:
        import time

        history = history or []
        messages = history + [{"role": "user", "content": message}]
        t0 = time.monotonic()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.system_prompt,
            messages=messages,
        )
        t1 = time.monotonic()
        text_block = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        delay_ms = self._resolve_delay_ms(text_block, response.usage.output_tokens)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        t2 = time.monotonic()
        return TwinReply(
            text=text_block,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            model_latency_ms=int((t1 - t0) * 1000),
            artificial_delay_ms=delay_ms,
            total_latency_ms=int((t2 - t0) * 1000),
        )
