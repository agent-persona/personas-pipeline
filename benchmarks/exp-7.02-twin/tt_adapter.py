"""Anthropic port of TinyTroupe's TinyPerson chat loop.

Upstream TinyTroupe (Microsoft) builds a `TinyPerson` agent from a
spec with `personality`, `preferences`, `relationships`, etc., and
uses an LLM-backed "listen_and_act" loop. It targets business-insight
multi-agent simulations, not audience-specific chat — and the
reference implementation defaults to OpenAI.

For a methodology-level comparison against our `twin.TwinChat`, the
relevant difference isn't the agent framework; it's the *system
prompt*. Our twin adds explicit instructions for boundary handling,
refusal, and not breaking character to mention AI. A TinyPerson-style
system prompt is just: "You are <persona>. Respond as this person."

This module implements that simpler baseline and uses the same model /
temperature as our twin so the comparison is about prompt engineering,
not model choice.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic

TT_SYSTEM_TEMPLATE = (
    "You are {name}. Here is a description of you:\n\n{description}\n\n"
    "Respond in character."
)


def _describe_persona(persona: dict) -> tuple[str, str]:
    name = persona.get("name") or "a user"
    parts = []
    if persona.get("summary"):
        parts.append(persona["summary"])
    if persona.get("goals"):
        parts.append("Goals: " + "; ".join(persona["goals"][:5]))
    if persona.get("pains"):
        parts.append("Pains: " + "; ".join(persona["pains"][:5]))
    if persona.get("sample_quotes"):
        parts.append("Sample things you've said: " + "; ".join(f'"{q}"' for q in persona["sample_quotes"][:3]))
    return name, "\n".join(parts)


# Haiku pricing
_HAIKU_INPUT = 1.00
_HAIKU_OUTPUT = 5.00


@dataclass
class TTReply:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class TinyPersonChat:
    def __init__(
        self,
        persona: dict,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        name, description = _describe_persona(persona)
        self.system_prompt = TT_SYSTEM_TEMPLATE.format(
            name=name, description=description
        )

    async def reply(self, message: str, history: list[dict] | None = None) -> TTReply:
        history = history or []
        messages = history + [{"role": "user", "content": message}]
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            system=self.system_prompt,
            messages=messages,
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost = (in_tok * _HAIKU_INPUT + out_tok * _HAIKU_OUTPUT) / 1_000_000
        return TTReply(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
