from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic


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


INTENSITY_BANDS: list[tuple[float, str]] = [
    (0.0, "subtle"),
    (0.25, "understated"),
    (0.5, "balanced"),
    (0.75, "strong"),
    (1.0, "vivid"),
]


def intensity_label(intensity: float) -> str:
    """Snap a float in [0,1] to the nearest named band."""
    intensity = max(0.0, min(1.0, intensity))
    return min(INTENSITY_BANDS, key=lambda b: abs(b[0] - intensity))[1]


_INTENSITY_FRAGMENTS: dict[str, str] = {
    "subtle": (
        "Right now your persona traits sit in the background. Your goals, "
        "concerns, and characteristic vocabulary come through only when the "
        "conversation directly touches them. When a topic doesn't invite them, "
        "you sound like a thoughtful professional first and this specific "
        "persona second."
    ),
    "understated": (
        "Right now your persona traits are present but reserved. You mention "
        "your concerns and priorities when they're relevant, but you don't "
        "lead with them, and your characteristic vocabulary appears "
        "occasionally rather than in every answer."
    ),
    "strong": (
        "Right now your persona traits are clearly visible. Your goals, "
        "concerns, and characteristic vocabulary shape how you phrase most "
        "answers, though you remain a coherent and plausible person — you "
        "don't repeat yourself and you don't force traits into places they "
        "don't fit."
    ),
    "vivid": (
        "Right now your persona traits are consistently noticeable. Your "
        "goals, concerns, and characteristic vocabulary show up in almost "
        "every answer. Stay a coherent, plausible human: no catchphrase "
        "stuffing, no forced repetition, no contradictions to your stated "
        "priorities — just a persona whose signature is always audible."
    ),
}

_INTENSITY_VOCAB_RULES: dict[str, str] = {
    "subtle": "- Use your characteristic vocabulary sparingly; let it surface only where it fits.",
    "understated": "- Use your characteristic vocabulary occasionally rather than in every response.",
    "strong": "- Your characteristic vocabulary should shape most answers, without repetition.",
    "vivid": "- Your characteristic vocabulary should be audible in almost every answer, without catchphrase stuffing.",
}

_DEFAULT_VOCAB_RULE = "- Use your vocabulary naturally — don't sound like a chatbot."


def build_persona_system_prompt(persona: dict, intensity: float = 0.5) -> str:
    """Construct a system prompt that puts Claude in character as the persona.

    Accepts a persona dict (the output of `PersonaV1.model_dump()`) so the
    twin runtime has no compile-time dependency on the synthesis package.

    `intensity` is a float in [0, 1] that snaps to one of five named bands
    (subtle/understated/balanced/strong/vivid) controlling how salient the
    persona traits are in the twin's output. The `balanced` band (default)
    is byte-identical to the pre-intensity production prompt.
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

    label = intensity_label(intensity)

    lines: list[str] = [
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
    ]

    # balanced band: byte-identical to pre-intensity prompt. No section, no
    # rule override. This is the experimental control and is load-bearing.
    if label != "balanced":
        lines.extend([
            "## Expressive intensity",
            _INTENSITY_FRAGMENTS[label],
            "",
        ])

    vocab_rule = (
        _DEFAULT_VOCAB_RULE if label == "balanced" else _INTENSITY_VOCAB_RULES[label]
    )

    lines.extend([
        "## Rules",
        "- Answer in first person, in character.",
        vocab_rule,
        "- Keep responses under 4 sentences unless asked to elaborate.",
        "- If asked something outside your knowledge or experience, react the way "
        "this persona would (curiosity, dismissal, deflection — whatever fits).",
        "- Do not break character to mention you are an AI.",
    ])
    return "\n".join(lines)


class TwinChat:
    """Stateless persona-driven chat. Caller manages history."""

    def __init__(
        self,
        persona: dict,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
        intensity: float = 0.5,
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.intensity = intensity
        self._prompt_cache: dict[str, str] = {}

    def _system_for(self, intensity: float) -> str:
        label = intensity_label(intensity)
        if label not in self._prompt_cache:
            self._prompt_cache[label] = build_persona_system_prompt(
                self.persona, intensity
            )
        return self._prompt_cache[label]

    @property
    def system_prompt(self) -> str:
        """Preserve the previous public attribute for any external reader."""
        return self._system_for(self.intensity)

    async def reply(
        self,
        message: str,
        history: list[dict] | None = None,
        intensity: float | None = None,
    ) -> TwinReply:
        system = self._system_for(
            self.intensity if intensity is None else intensity
        )
        history = history or []
        messages = history + [{"role": "user", "content": message}]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
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
