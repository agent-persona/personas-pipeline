"""Shared utilities for LLM-as-judge rubric scorers."""
from __future__ import annotations

import json
import re

from persona_eval.llm_client import LLMClient

_SCORE_RE = re.compile(r'"score"\s*:\s*([1-5](?:\.\d+)?)(?!\d)')

JUDGE_MODEL = "claude-sonnet-4-20250514"


def make_client() -> LLMClient:
    return LLMClient(model=JUDGE_MODEL, temperature=0.0)


def parse_score(raw: str) -> tuple[float, str, bool]:
    """Return (raw_1_to_5_float, reasoning_str, parse_ok_bool).

    Tries JSON first; falls back to regex scan; returns (3.0, raw[:200], False)
    on total failure so scorers degrade gracefully without raising.
    """
    try:
        data = json.loads(raw)
        return float(data["score"]), data.get("reasoning", ""), True
    except Exception:
        pass
    m = _SCORE_RE.search(raw)
    if m:
        return float(m.group(1)), raw[:200], True
    return 3.0, raw[:200], False


def normalize(raw_score: float) -> float:
    """Map 1–5 to 0.0–1.0, clamped at both ends."""
    return round(max(0.0, min(1.0, (raw_score - 1) / 4)), 4)


def build_persona_block(persona) -> str:
    """Compact persona text block for LLM prompts. Excludes empty fields."""
    lines = [
        f"Name: {persona.name}",
        f"Age: {persona.age}",
        f"Occupation: {persona.occupation}",
        f"Industry: {persona.industry}",
        f"Education: {persona.education}",
        f"Location: {persona.location}",
        f"Personality traits: {', '.join(persona.personality_traits)}",
        f"Values: {', '.join(persona.values)}",
        f"Goals: {', '.join(persona.goals)}",
        f"Pain points: {', '.join(persona.pain_points)}",
        f"Communication tone: {persona.communication_style.tone}",
        f"Communication formality: {persona.communication_style.formality}",
        f"Vocabulary level: {persona.communication_style.vocabulary_level}",
        f"Baseline mood: {persona.emotional_profile.baseline_mood}",
        f"Core values (moral): {', '.join(persona.moral_framework.core_values)}",
        f"Bio: {persona.bio}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))
