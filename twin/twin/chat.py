from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AsyncAnthropic

# exp-4.13: length-matching modes for the twin's response-length instruction.
#
# - "fixed": current behavior — "Keep responses under 4 sentences unless asked to elaborate."
# - "mirror": match the user's message length tightly.
# - "mirror_with_floor": match the user's length but never drop below one full sentence.
#
# The relevant line is substituted into build_persona_system_prompt's "## Rules"
# section so the model sees exactly one length instruction with no conflict.
LengthMode = Literal["fixed", "mirror", "mirror_with_floor"]

_LENGTH_RULES: dict[str, str] = {
    "fixed": "Keep responses under 4 sentences unless asked to elaborate.",
    "mirror": (
        "Match the user's message length. If they send a single word, answer "
        "in a single word. If they send a paragraph, match it with a paragraph. "
        "Length is a social signal — don't over-explain when they're terse, and "
        "don't under-explain when they're taking time to write."
    ),
    "mirror_with_floor": (
        "Match the user's message length, but never drop below one complete sentence. "
        "If they send a single word, still respond with at least one full sentence — "
        "short but complete. If they send a paragraph, match the paragraph. "
        "Length is a social signal; use it, but don't become a one-word bot."
    ),
}


@dataclass
class TwinReply:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def estimated_cost_usd(self) -> float:
        if "opus" in self.model:
            return (self.input_tokens * 15 + self.output_tokens * 75) / 1_000_000
        if "haiku" in self.model:
            return (self.input_tokens * 1 + self.output_tokens * 5) / 1_000_000
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


def build_persona_system_prompt(
    persona: dict,
    length_mode: LengthMode = "fixed",
) -> str:
    """Construct a system prompt that puts Claude in character as the persona.

    Accepts a persona dict (the output of `PersonaV1.model_dump()`) so the
    twin runtime has no compile-time dependency on the synthesis package.

    exp-4.13: length_mode controls the response-length rule. "fixed" is the
    current production behavior; "mirror" and "mirror_with_floor" are
    experimental variants.
    """
    name = persona.get("name", "the persona")
    summary = persona.get("summary", "")
    demo = persona.get("demographics", {})
    firmo = persona.get("firmographics", {})
    goals = persona.get("goals", [])
    pains = persona.get("pains", [])
    motivations = persona.get("motivations", [])
    objections = persona.get("objections", [])
    vocabulary = persona.get("vocabulary", [])
    sample_quotes = persona.get("sample_quotes", [])

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
        "## How you talk",
        f"You use words like: {', '.join(vocabulary)}.",
        "",
        "Examples of things you have said before:",
        *(f'- "{q}"' for q in sample_quotes),
        "",
        "## Rules",
        "- Answer in first person, in character.",
        "- Use your vocabulary naturally — don't sound like a chatbot.",
        f"- {_LENGTH_RULES[length_mode]}",
        "- If asked something outside your knowledge or experience, react the way "
        "this persona would (curiosity, dismissal, deflection — whatever fits).",
        "- Do not break character to mention you are an AI.",
    ]
    return "\n".join(lines)


class TwinChat:
    """Stateless persona-driven chat. Caller manages history."""

    def __init__(
        self,
        persona: dict,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
        length_mode: LengthMode = "fixed",
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.length_mode = length_mode
        self.system_prompt = build_persona_system_prompt(persona, length_mode=length_mode)

    async def reply(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> TwinReply:
        history = history or []
        messages = history + [{"role": "user", "content": message}]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.system_prompt,
            messages=messages,
        )
        text_block = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        return TwinReply(
            text=text_block,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )
