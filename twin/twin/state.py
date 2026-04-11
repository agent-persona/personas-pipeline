"""Emotional state tracker for twin personas (experiment 4.11).

Tracks (valence, arousal) between turns based on user input keywords.
Valence: -1.0 (upset) to 1.0 (happy)
Arousal: 0.0 (calm) to 1.0 (excited/urgent)
"""

from __future__ import annotations


_POSITIVE = {"love", "great", "amazing", "thanks", "thank", "awesome", "fantastic",
             "perfect", "excellent", "happy", "glad", "appreciate", "wonderful", "nice"}
_NEGATIVE = {"hate", "frustrated", "broken", "terrible", "awful", "angry", "annoyed",
             "disappointing", "useless", "worst", "horrible", "fail", "failed", "bug"}
_HIGH_AROUSAL = {"urgent", "critical", "immediately", "asap", "emergency", "now",
                 "amazing", "incredible", "shocking", "breaking"}
_LOW_AROUSAL = {"whenever", "no rush", "just curious", "wondering", "maybe",
                "eventually", "sometime", "casual"}


class EmotionalState:
    """Simple keyword-based emotional state tracker."""

    def __init__(self, valence: float = 0.0, arousal: float = 0.3) -> None:
        self.valence = valence
        self.arousal = arousal

    def update(self, user_message: str) -> None:
        words = set(user_message.lower().split())
        text_lower = user_message.lower()

        pos = len(words & _POSITIVE)
        neg = len(words & _NEGATIVE)
        hi = len(words & _HIGH_AROUSAL) + sum(1 for p in _HIGH_AROUSAL if p in text_lower)
        lo = len(words & _LOW_AROUSAL) + sum(1 for p in _LOW_AROUSAL if p in text_lower)

        # Shift valence
        self.valence += pos * 0.15 - neg * 0.2
        self.valence = max(-1.0, min(1.0, self.valence))

        # Shift arousal
        self.arousal += hi * 0.15 - lo * 0.1
        self.arousal = max(0.0, min(1.0, self.arousal))

        # Decay toward neutral
        self.valence *= 0.9
        self.arousal = self.arousal * 0.9 + 0.1 * 0.3  # decay toward 0.3

    def describe(self) -> str:
        if self.valence > 0.3:
            mood = "pleased and engaged"
        elif self.valence > 0.1:
            mood = "in a good mood"
        elif self.valence < -0.3:
            mood = "frustrated and impatient"
        elif self.valence < -0.1:
            mood = "slightly annoyed"
        else:
            mood = "neutral"

        if self.arousal > 0.6:
            energy = "highly alert and energized"
        elif self.arousal > 0.4:
            energy = "attentive"
        else:
            energy = "calm and relaxed"

        return f"You're feeling {mood}, and your energy level is {energy}."

    def prompt_section(self) -> str:
        return (
            f"\n\n## Current Emotional State\n"
            f"Valence: {self.valence:+.2f} (negative=upset, positive=happy)\n"
            f"Arousal: {self.arousal:.2f} (0=calm, 1=excited)\n"
            f"{self.describe()}\n"
            f"Let this emotional state subtly influence your tone and word choice."
        )
