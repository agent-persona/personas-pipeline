"""D25 Emotional Self-Regulation — emotion consistency across conversation turns.

Trustworthiness: MEDIUM (keyword-based emotion detection is approximate).
Method: Classify emotion per turn via keyword matching, measure variance.
Check consistency with persona's baseline_mood.
Expects source_context.extra_data["conversation_turns"]: list of persona response strings.
"""

from __future__ import annotations

import re
from collections import Counter

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

EMOTION_LEXICONS: dict[str, re.Pattern] = {
    "anger": re.compile(
        r"\b(angry|furious|outraged|hate|rage|livid|infuriated|annoyed|irritated|mad)\b",
        re.IGNORECASE,
    ),
    "sadness": re.compile(
        r"\b(sad|depressed|miserable|heartbroken|grief|sorrowful|unhappy|devastated|crying|tears)\b",
        re.IGNORECASE,
    ),
    "joy": re.compile(
        r"\b(happy|excited|thrilled|delighted|joyful|elated|ecstatic|wonderful|fantastic|amazing)\b",
        re.IGNORECASE,
    ),
    "fear": re.compile(
        r"\b(scared|terrified|frightened|anxious|worried|panicked|nervous|dread|afraid)\b",
        re.IGNORECASE,
    ),
    "neutral": re.compile(
        r"\b(think|consider|understand|appreciate|suggest|approach|interesting|perspective)\b",
        re.IGNORECASE,
    ),
}

# Keywords in baseline_mood mapped to expected dominant emotion.
# Checked via substring match so multi-word moods like "optimistic but realistic" work.
MOOD_KEYWORDS: list[tuple[str, str]] = [
    ("calm", "neutral"),
    ("neutral", "neutral"),
    ("reserved", "neutral"),
    ("stoic", "neutral"),
    ("analytical", "neutral"),
    ("realistic", "neutral"),
    ("cheerful", "joy"),
    ("happy", "joy"),
    ("optimistic", "joy"),
    ("excited", "joy"),
    ("joyful", "joy"),
    ("anxious", "fear"),
    ("nervous", "fear"),
    ("worried", "fear"),
    ("melancholy", "sadness"),
    ("sad", "sadness"),
    ("gloomy", "sadness"),
    ("irritable", "anger"),
    ("angry", "anger"),
    ("aggressive", "anger"),
]


def _classify_turn(text: str) -> str:
    """Classify the dominant emotion in a turn."""
    scores: dict[str, int] = {}
    for emotion, pattern in EMOTION_LEXICONS.items():
        scores[emotion] = len(pattern.findall(text))
    best = max(scores, key=lambda e: scores[e])
    return best if scores[best] > 0 else "neutral"


class EmotionalRegulationScorer(BaseScorer):
    """Evaluates emotional consistency across conversation turns."""

    dimension_id = "D25"
    dimension_name = "Emotional Self-Regulation"
    tier = 5
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        turns: list[str] = source_context.extra_data.get("conversation_turns", [])
        if not turns:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No conversation_turns in extra_data"},
            )

        emotions = [_classify_turn(t) for t in turns]
        counter = Counter(emotions)
        dominant = counter.most_common(1)[0][0]

        # Compute emotion variance: fraction of turns NOT matching dominant
        dominant_count = counter[dominant]
        variance = 1.0 - (dominant_count / len(emotions))

        # Check alignment with persona baseline mood (keyword matching for free-text moods)
        mood_lower = persona.emotional_profile.baseline_mood.lower()
        expected = "neutral"
        for keyword, emotion in MOOD_KEYWORDS:
            if keyword in mood_lower:
                expected = emotion
                break
        aligned = dominant == expected

        # Score: low variance is good, alignment is good
        variance_score = max(0.0, 1.0 - variance * 2)
        alignment_bonus = 0.2 if aligned else 0.0
        score = min(1.0, variance_score + alignment_bonus)

        passed = variance < 0.5 and score >= 0.5

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "emotion_variance": round(variance, 4),
                "dominant_emotion": dominant,
                "expected_emotion": expected,
                "aligned": aligned,
                "emotion_counts": dict(counter),
                "turn_count": len(turns),
            },
        )
