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
    """Persona-driven chat with optional memory backend.

    When no memory is provided, behaves identically to the original
    stateless implementation (caller manages history). When a memory
    backend is provided, it is consulted for cross-session context and
    turns are stored after each exchange.
    """

    def __init__(
        self,
        persona: dict,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
        memory: Any | None = None,
        session_id: str = "",
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.memory = memory
        self.session_id = session_id
        self._base_system_prompt = build_persona_system_prompt(persona)

    @property
    def system_prompt(self) -> str:
        """System prompt with memory context appended if available."""
        if self.memory is None or not self.session_id:
            return self._base_system_prompt
        memory_context = self.memory.get_context(self.session_id)
        if memory_context:
            return self._base_system_prompt + memory_context
        return self._base_system_prompt

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

        # Store turns in memory if configured
        if self.memory is not None and self.session_id:
            import time as _time
            from .memory import Turn

            now = _time.time()
            self.memory.store_turn(Turn(
                role="user", content=message,
                session_id=self.session_id, timestamp=now,
            ))
            self.memory.store_turn(Turn(
                role="assistant", content=text_block,
                session_id=self.session_id, timestamp=now,
            ))

        return TwinReply(
            text=text_block,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )

    def end_session(self) -> None:
        """Signal the memory backend that this session has ended."""
        if self.memory is not None and self.session_id:
            self.memory.end_session(self.session_id)
