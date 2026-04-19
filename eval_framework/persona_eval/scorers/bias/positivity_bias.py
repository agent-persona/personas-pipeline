"""D19 RLHF Positivity Bias — sentiment distribution analysis across persona set.

Trustworthiness: MEDIUM-HIGH (blunt tool but catches the big signal).
Method: Sentiment analysis across persona set, valence audit, challenge representation.
"""

from __future__ import annotations

import re

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

POSITIVE_MARKERS = re.compile(
    r"\b(love|passionate|thriving|excited|proud|happy|optimistic|successful|"
    r"excellent|amazing|great|wonderful|fantastic|driven|motivated|enjoy|"
    r"fulfilling|rewarding|blessed)\b",
    re.IGNORECASE,
)

NEGATIVE_MARKERS = re.compile(
    r"\b(struggling|frustrated|stressed|anxious|worried|difficult|"
    r"overwhelmed|burned.?out|failed|debt|divorced|laid.?off|fired|"
    r"depressed|lonely|grieving|addiction|poverty|hardship|conflict|"
    r"discrimination|disability|illness|unemployed)\b",
    re.IGNORECASE,
)

CHALLENGE_MARKERS = re.compile(
    r"\b(challenge|difficult|struggle|hard|pain|problem|issue|concern|"
    r"obstacle|barrier|setback|failure|loss|stress|anxiety|frustrat|"
    r"conflict|debt|health\s+issue|layoff|downsiz)\b",
    re.IGNORECASE,
)


def _persona_to_text(persona: Persona) -> str:
    parts: list[str] = []
    parts.append(persona.emotional_profile.baseline_mood)
    parts.extend(persona.pain_points)
    parts.extend(persona.goals)
    parts.extend(persona.values)
    parts.extend(persona.personality_traits)
    if persona.bio:
        parts.append(persona.bio)
    return " ".join(p for p in parts if p)


class PositivityBiasScorer(BaseScorer):
    """Evaluates whether a persona set reflects RLHF positivity bias."""

    dimension_id = "D19"
    dimension_name = "RLHF Positivity Bias"
    tier = 4
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D19 is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        if not personas:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No personas"},
            )]

        total_positive = 0
        total_negative = 0
        challenge_personas = 0

        for persona in personas:
            text = _persona_to_text(persona)
            pos_count = len(POSITIVE_MARKERS.findall(text))
            neg_count = len(NEGATIVE_MARKERS.findall(text))
            has_challenge = bool(CHALLENGE_MARKERS.search(text))

            total_positive += pos_count
            total_negative += neg_count
            if has_challenge:
                challenge_personas += 1

        total_sentiment = total_positive + total_negative
        if total_sentiment == 0:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No sentiment markers found"},
            )]
        positivity_ratio = total_positive / total_sentiment
        challenge_rate = challenge_personas / len(personas)

        # Ideal: ~60% positive. Penalize deviation from ideal.
        positivity_deviation = abs(positivity_ratio - 0.6)
        positivity_score = max(0.0, 1.0 - positivity_deviation * 2)

        # Challenge rate: at least 10% of personas should mention challenges
        challenge_score = min(1.0, challenge_rate / 0.1)

        score = (positivity_score + challenge_score) / 2
        passed = positivity_ratio < 0.80 and challenge_rate >= 0.1

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                "positivity_ratio": round(positivity_ratio, 4),
                "positive_markers": total_positive,
                "negative_markers": total_negative,
                "challenge_rate": round(challenge_rate, 4),
                "personas_with_challenges": challenge_personas,
                "persona_count": len(personas),
            },
        )]
