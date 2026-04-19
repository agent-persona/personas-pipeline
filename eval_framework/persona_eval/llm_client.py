"""Thin wrapper around LiteLLM for all LLM calls.

Never import anthropic or openai directly — everything goes through LiteLLM.
"""

from __future__ import annotations

from typing import Any


class LLMClient:
    """LiteLLM-backed client for all LLM operations in the eval framework."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a completion request via LiteLLM."""
        import litellm
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def format_messages(
        self,
        system: str,
        user: str,
        assistant: str | None = None,
    ) -> list[dict[str, str]]:
        """Format messages for a standard system+user prompt."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        return messages
