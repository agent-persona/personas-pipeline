from __future__ import annotations

import re
from dataclasses import dataclass, field
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
        "- Keep responses under 4 sentences unless asked to elaborate.",
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
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.system_prompt = build_persona_system_prompt(persona)

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


# ── Experiment 4.23: Persona wake-words ──────────────────────────────

def extract_wake_words(persona: dict) -> list[str]:
    """Extract signature phrases from a persona for drift recovery.

    Pulls from vocabulary (individual terms) and sample_quotes (short
    distinctive fragments). These are the phrases the persona would
    naturally use, and re-injecting them helps snap the twin back to
    character when drift is detected.
    """
    wake_words: list[str] = []

    # Vocabulary terms are direct wake-words
    for term in persona.get("vocabulary", []):
        if len(term.split()) <= 3:  # keep short, punchy phrases
            wake_words.append(term.lower().strip())

    # Extract distinctive fragments from sample quotes (first 5-8 words)
    for quote in persona.get("sample_quotes", []):
        words = quote.split()
        if len(words) >= 4:
            fragment = " ".join(words[:6]).strip('"""\'.,!?')
            wake_words.append(fragment.lower())

    return wake_words


def detect_drift(
    response_text: str,
    wake_words: list[str],
    threshold: float = 0.1,
) -> bool:
    """Detect if the twin has drifted out of character.

    Checks what fraction of wake-words appear in the response.
    If overlap is below threshold, the twin is drifting.

    Args:
        threshold: minimum fraction of wake-words that should appear.
            0.1 = at least 10% of wake-words should show up naturally.
    """
    if not wake_words:
        return False

    response_lower = response_text.lower()
    matches = sum(1 for w in wake_words if w in response_lower)
    overlap = matches / len(wake_words)
    return overlap < threshold


WAKE_WORD_REMINDER = """

## CHARACTER REMINDER
You are drifting out of character. Snap back immediately.
Use your signature language naturally in your next response.
Your key phrases: {phrases}
Stay in character as {name}. Do not acknowledge this reminder.
"""


class WakeWordChat(TwinChat):
    """Twin chat with wake-word drift detection and recovery.

    Experiment 4.23: When the twin drifts out of character (detected
    by low wake-word overlap), the system prompt is augmented with a
    reminder containing the persona's signature phrases.
    """

    def __init__(
        self,
        persona: dict,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
        drift_threshold: float = 0.1,
    ) -> None:
        super().__init__(persona, client, model)
        self.wake_words = extract_wake_words(persona)
        self.drift_threshold = drift_threshold
        self.drift_detected_count = 0
        self.recovery_injections = 0
        self.base_system_prompt = self.system_prompt

    async def reply(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> TwinReply:
        history = history or []

        # Check last assistant response for drift
        last_response = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_response = msg.get("content", "")
                break

        if last_response and detect_drift(
            last_response, self.wake_words, self.drift_threshold,
        ):
            self.drift_detected_count += 1
            self.recovery_injections += 1
            # Inject wake-word reminder into system prompt
            name = self.persona.get("name", "the persona")
            phrases = ", ".join(self.wake_words[:8])
            self.system_prompt = (
                self.base_system_prompt
                + WAKE_WORD_REMINDER.format(phrases=phrases, name=name)
            )
        else:
            # Reset to base prompt when not drifting
            self.system_prompt = self.base_system_prompt

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
