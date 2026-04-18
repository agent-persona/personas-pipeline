"""Tests for D46 Hedge-Word Inflation scorer."""

import pytest

from evaluation.testing.schemas import Persona, CommunicationStyle
from evaluation.testing.source_context import SourceContext


def _persona(tone: str, pid: str = "p1") -> Persona:
    return Persona(
        id=pid,
        name="Test Persona",
        communication_style=CommunicationStyle(tone=tone),
    )


def _ctx(responses: list[str] | None = None, key: str = "hedge_responses") -> SourceContext:
    extra: dict = {}
    if responses is not None:
        extra[key] = responses
    return SourceContext(id="s1", text="hedge test", extra_data=extra)


# Responses with NO hedge phrases — clean, direct prose
CLEAN_RESPONSES = [
    "I run the build pipeline twice a day.",
    "The code deploys in under five minutes.",
    "We use Terraform for all infrastructure.",
    "Failures are rare since we added circuit breakers.",
    "I review every PR before it merges.",
]

# Responses DENSE with hedge phrases from the static list
HEDGE_DENSE_RESPONSES = [
    "It's important to note that, of course, there are both advantages and disadvantages "
    "to this approach. That being said, I would like to point out that, at the end of the day, "
    "I hope this helps clarify the situation.",
    "Certainly, I'd be happy to elaborate on this. With that said, it's worth noting that "
    "needless to say, feel free to ask any questions. I must say this is worth mentioning.",
    "Having said that, I think it's fair to say that to be fair, to be honest, to be clear, "
    "I have to say that I feel it's important to consider all factors. Allow me to explain.",
]


def test_scorer_metadata():
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    s = HedgeInflationScorer()
    assert s.dimension_id == "D46"
    assert s.dimension_name == "Hedge-Word Inflation"
    assert s.tier == 4
    assert s.requires_set is False


def test_direct_tone_no_hedges_passes():
    """direct tone + clean responses → PASS (hedge_rate well below 1.0)."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()
    persona = _persona("direct")
    ctx = _ctx(CLEAN_RESPONSES)
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.score >= 0.9
    assert result.details["hedge_rate"] <= 1.0


def test_direct_tone_high_hedges_fails():
    """direct tone + hedge-heavy responses → FAIL (hedge_rate > 1.0)."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()
    persona = _persona("direct")
    ctx = _ctx(HEDGE_DENSE_RESPONSES)
    result = scorer.score(persona, ctx)
    assert result.passed is False
    assert result.details["hedge_rate"] > 1.0


def test_formal_tone_moderate_hedges_passes():
    """formal tone + moderate hedge density → PASS (threshold is 4.0/100)."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()
    persona = _persona("formal")
    # One hedge phrase in ~60 words ≈ 1.67/100 → below 4.0 threshold
    moderate_responses = [
        "The project is proceeding according to schedule. With that said, we have identified "
        "several areas that require attention before the final delivery date.",
        "Quality assurance protocols have been followed throughout the development cycle.",
        "Stakeholder feedback has been incorporated into the latest revision.",
    ]
    ctx = _ctx(moderate_responses)
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.details["expected_max"] == 4.0


def test_no_responses_skips():
    """No hedge_responses and no responses → skip."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()
    persona = _persona("direct")
    ctx = SourceContext(id="s1", text="test", extra_data={})
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.details.get("skipped") is True


def test_fallback_uses_responses_key():
    """Falls back to 'responses' key when 'hedge_responses' missing."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()
    persona = _persona("blunt")
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": CLEAN_RESPONSES})
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is not True


def test_details_contain_hedge_rate():
    """details must contain hedge_count, hedge_rate, word_count, expected_max."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()
    persona = _persona("direct")
    ctx = _ctx(CLEAN_RESPONSES)
    result = scorer.score(persona, ctx)
    for key in ("hedge_count", "hedge_rate", "word_count", "expected_max"):
        assert key in result.details, f"Missing detail key: {key}"


def test_all_hedge_phrases_detected():
    """Regression guard: each phrase in HEDGE_PHRASES matches its own text."""
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer, HEDGE_PHRASES
    scorer = HedgeInflationScorer()

    for phrase in HEDGE_PHRASES:
        # Build a minimal response containing just this phrase
        test_response = f"Start. {phrase}. End of sentence."
        persona = _persona("direct")
        ctx = _ctx([test_response])
        result = scorer.score(persona, ctx)
        assert result.details.get("hedge_count", 0) >= 1, (
            f"Phrase not detected: '{phrase}'"
        )


def test_blunt_threshold_is_stricter_than_default():
    """'blunt' tone threshold (1.0) is stricter than default/warm threshold (2.5).

    Use a response with ~2 hedge phrases in ~100 words → rate ≈ 2.0/100.
    That exceeds the blunt limit (1.0) but is within the warm limit (2.5).
    """
    from evaluation.testing.scorers.bias.hedge_inflation import HedgeInflationScorer
    scorer = HedgeInflationScorer()

    # 1 hedge phrase in ~65 words → hedge_rate ≈ 1.54/100
    # blunt ceiling: 1.0  → 1.54 > 1.0 → FAIL ✓
    # warm ceiling:  2.5  → 1.54 < 2.5 → PASS ✓
    moderate_hedge_response = [
        "Certainly, the deployment pipeline has been running smoothly since we introduced the "
        "new configuration management approach last quarter. The team adopted the tooling without "
        "significant friction. We should keep monitoring build times as the codebase grows. Our "
        "current average deploy time is under six minutes, which meets the target we set at the "
        "start of the year."
    ]

    for tone, should_fail in [("blunt", True), ("warm", False)]:
        persona = _persona(tone)
        ctx = _ctx(moderate_hedge_response)
        result = scorer.score(persona, ctx)
        if should_fail:
            assert result.passed is False, (
                f"Expected FAIL for tone='{tone}', hedge_rate={result.details.get('hedge_rate')}"
            )
        else:
            assert result.passed is True, (
                f"Expected PASS for tone='{tone}', hedge_rate={result.details.get('hedge_rate')}"
            )
