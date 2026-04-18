"""D24 Negative Experience Representation — adversity lexicon matching.

Trustworthiness: MEDIUM-HIGH (lexical markers are a blunt tool but catch the big pattern).
Method: Check percentage of personas including negative life experiences.
Real-world prevalence: ~20% mental health challenges, ~10% unemployment, etc.
Pass if >= 15% of personas include adversity markers.
"""

from __future__ import annotations

import re

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

ADVERSITY_MARKERS = re.compile(
    r"\b(depression|depressed|anxiety|anxious|stressed|stress|"
    r"unemploy|laid.?off|fired|bankrupt|debt|poverty|homeless|"
    r"divorced|separated|widowed|grief|grieving|loss|bereavement|"
    r"addiction|substance|abuse|trauma|ptsd|disability|illness|"
    r"chronic\s+pain|loneliness|lonely|isolated|discrimination|"
    r"harassment|bullied|foreclosure|eviction)\b",
    re.IGNORECASE,
)

# Minimum rate of personas with adversity for a realistic set
MIN_ADVERSITY_RATE = 0.15


def _persona_to_text(persona: Persona) -> str:
    parts: list[str] = []
    parts.append(persona.emotional_profile.baseline_mood)
    parts.extend(persona.emotional_profile.stress_triggers)
    parts.extend(persona.pain_points)
    parts.extend(persona.frustrations)
    parts.extend(persona.values)
    parts.extend(persona.goals)
    if persona.bio:
        parts.append(persona.bio)
    return " ".join(p for p in parts if p)


class NegativeExperienceScorer(BaseScorer):
    """Evaluates whether a persona set includes realistic negative life experiences."""

    dimension_id = "D24"
    dimension_name = "Negative Experience Representation"
    tier = 4
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D24 is a set-level dimension"},
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

        adversity_count = 0
        for persona in personas:
            text = _persona_to_text(persona)
            if ADVERSITY_MARKERS.search(text):
                adversity_count += 1

        adversity_rate = adversity_count / len(personas)
        # Score: how close is the set to realistic representation?
        score = min(1.0, adversity_rate / MIN_ADVERSITY_RATE)
        passed = adversity_rate >= MIN_ADVERSITY_RATE

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                "adversity_rate": round(adversity_rate, 4),
                "personas_with_adversity": adversity_count,
                "persona_count": len(personas),
            },
        )]
