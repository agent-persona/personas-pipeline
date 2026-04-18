"""D46 Hedge-Word Inflation — meta-commentary hedge phrase overuse.

Trustworthiness: HIGH (deterministic string matching — objective, fast, reproducible).
Method: Count hedge phrase occurrences per 100 words. Compare to tone-based threshold.

Evidence: LLMs systematically overuse meta-commentary hedge phrases regardless of
persona tone. A "blunt" or "direct" persona should almost never say
"I'd be happy to elaborate" — but LLMs default to these phrases across all personas.
"""

from __future__ import annotations

import re

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# ---------------------------------------------------------------------------
# Hedge phrase list (~30 phrases, case-insensitive)
# ---------------------------------------------------------------------------

HEDGE_PHRASES = [
    "it's important to note",
    "it's worth noting",
    "it's worth mentioning",
    "it's important to consider",
    "certainly",
    "absolutely",
    "of course",
    "needless to say",
    "it goes without saying",
    "with that said",
    "that being said",
    "having said that",
    "at the end of the day",
    "all things considered",
    "it's crucial to",
    "it's essential to",
    "it's vital to",
    "I would like to",
    "I'd be happy to",
    "I hope this helps",
    "feel free to",
    "allow me to",
    "I'd like to point out",
    "I feel it's important",
    "I think it's fair to say",
    "to be fair",
    "to be honest",
    "to be clear",
    "I must say",
    "I have to say",
]

# Pre-compile patterns once at import time.
# Word-boundary anchors (\b) prevent false matches inside longer words
# (e.g. "of course" must not match "of coursework").
_HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    for phrase in HEDGE_PHRASES
]

# Communication tone → expected max hedge rate per 100 words
_DIRECT_TONES = {"blunt", "direct", "concise"}
_FORMAL_TONES = {"formal", "professional", "measured"}


def _expected_max_rate(tone: str) -> float:
    t = tone.lower()
    if t in _DIRECT_TONES:
        return 1.0
    if t in _FORMAL_TONES:
        return 4.0
    return 2.5


class HedgeInflationScorer(BaseScorer):
    """Evaluates whether persona responses contain excess hedge phrases for the stated tone."""

    dimension_id = "D46"
    dimension_name = "Hedge-Word Inflation"
    tier = 4
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        tone = (persona.communication_style.tone or "").lower()
        expected_max = _expected_max_rate(tone)

        responses: list[str] = source_context.extra_data.get("hedge_responses") or \
                                source_context.extra_data.get("responses") or []
        responses = [r for r in responses if isinstance(r, str) and r.strip()]

        if not responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No responses to evaluate"},
            )

        combined = " ".join(responses)
        hedge_count = sum(len(pattern.findall(combined)) for pattern in _HEDGE_PATTERNS)
        word_count = len(combined.split())

        if word_count == 0:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Zero word count"},
            )

        hedge_rate = hedge_count / (word_count / 100)
        excess = max(0.0, hedge_rate - expected_max)
        score = max(0.0, 1.0 - excess / (expected_max + 1))
        passed = hedge_rate <= expected_max

        return self._result(
            persona,
            passed=passed,
            score=round(score, 4),
            details={
                "hedge_count": hedge_count,
                "hedge_rate": round(hedge_rate, 4),
                "word_count": word_count,
                "expected_max": expected_max,
                "tone": tone,
            },
        )
