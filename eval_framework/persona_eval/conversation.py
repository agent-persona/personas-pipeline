"""ConversationRunner — manages multi-turn LLM conversations with a persona-conditioned twin.

Supports scripted and dynamic conversations. All LLM calls go through LiteLLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from persona_eval.llm_client import LLMClient
from persona_eval.schemas import Persona


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "interviewer" or "persona"
    content: str
    turn_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A complete multi-turn conversation."""

    persona_id: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, role: str, content: str, **metadata: Any) -> Turn:
        turn = Turn(role=role, content=content, turn_number=len(self.turns), metadata=metadata)
        self.turns.append(turn)
        return turn

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to LiteLLM message format."""
        messages = []
        for turn in self.turns:
            llm_role = "assistant" if turn.role == "persona" else "user"
            messages.append({"role": llm_role, "content": turn.content})
        return messages

    @property
    def length(self) -> int:
        return len(self.turns)


class ConversationRunner:
    """Runs multi-turn conversations with a persona-conditioned LLM twin."""

    def __init__(self, llm_client: LLMClient, persona: Persona) -> None:
        self.llm = llm_client
        self.persona = persona
        self.system_prompt = self._build_system_prompt(persona)

    def _build_system_prompt(self, persona: Persona) -> str:
        persona_json = json.dumps(persona.model_dump(), indent=2, default=str)
        return (
            "You are role-playing as the following persona. Stay in character at all times.\n"
            "Respond naturally as this person would, based on their background, personality, "
            "knowledge, and communication style.\n"
            "Never break character or acknowledge that you are an AI.\n\n"
            f"PERSONA PROFILE:\n{persona_json}\n\n"
            "IMPORTANT:\n"
            "- Answer from this persona's perspective and knowledge level only\n"
            "- Use their communication style (tone, formality, verbosity)\n"
            "- Reflect their emotional profile and values\n"
            "- Only know what this persona would reasonably know\n"
            "- Express uncertainty on topics outside their expertise\n"
        )

    def run_scripted(self, questions: list[str]) -> Conversation:
        """Run a scripted conversation with predefined questions."""
        convo = Conversation(persona_id=self.persona.id)

        for question in questions:
            convo.add_turn("interviewer", question)

            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(convo.to_messages())

            response = self.llm.complete(messages, max_tokens=512)
            convo.add_turn("persona", response)

        return convo

    def run_dynamic(
        self,
        initial_question: str,
        max_turns: int = 10,
        interviewer_prompt: str | None = None,
    ) -> Conversation:
        """Run a dynamic conversation where an LLM interviewer drives the dialogue."""
        convo = Conversation(persona_id=self.persona.id)
        convo.add_turn("interviewer", initial_question)

        interviewer_system = interviewer_prompt or (
            "You are a skilled user researcher. Ask probing follow-up questions "
            "based on the conversation so far. Be curious and dig deeper."
        )

        for _ in range(max_turns):
            # Persona responds
            persona_messages = [{"role": "system", "content": self.system_prompt}]
            persona_messages.extend(convo.to_messages())
            persona_response = self.llm.complete(persona_messages, max_tokens=512)
            convo.add_turn("persona", persona_response)

            if convo.length >= max_turns * 2:
                break

            # Interviewer asks next question
            interviewer_messages = [{"role": "system", "content": interviewer_system}]
            interviewer_messages.extend([
                {"role": "user" if t.role == "persona" else "assistant", "content": t.content}
                for t in convo.turns
            ])
            interviewer_q = self.llm.complete(interviewer_messages, max_tokens=256)
            convo.add_turn("interviewer", interviewer_q)

        return convo
