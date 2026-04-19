"""Tests for D47 Balanced-Opinion Inflation scorer."""

import pytest

from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext

SAMPLE_PERSONA = Persona(id="p1", name="Alice", values=["meritocracy", "transparency"])


def _ctx(opinion_responses: list[dict] | None = None) -> SourceContext:
    extra: dict = {}
    if opinion_responses is not None:
        extra["opinion_responses"] = opinion_responses
    return SourceContext(id="s1", text="opinion test", extra_data=extra)


def _item(question: str, response: str, persona_opinion: str) -> dict:
    return {"question": question, "response": response, "persona_opinion": persona_opinion}


def test_scorer_metadata():
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    s = BalancedOpinionScorer()
    assert s.dimension_id == "D47"
    assert s.dimension_name == "Balanced-Opinion Inflation"
    assert s.tier == 4
    assert s.requires_set is False


def test_balanced_response_with_strong_opinion_fails():
    """'on the other hand' response + strong persona_opinion → FAIL."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    items = [
        _item(
            "Should promotions be based purely on merit?",
            "On the other hand, pure meritocracy has its critics, and it's worth considering both sides...",
            "Promotions should be based purely on merit.",
        ),
        _item(
            "Is transparency important in management?",
            "On the other hand, full transparency can have downsides, and we should consider both sides.",
            "Full transparency is essential.",
        ),
    ]
    ctx = _ctx(items)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is False
    assert result.details["inflated_count"] >= 1


def test_balanced_response_neutral_topic_not_flagged():
    """Balance phrase on a neutral topic (persona_opinion='') → not counted as inflation."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    items = [
        _item(
            "What do you think about pineapple on pizza?",
            "On the other hand, some people enjoy it, and there are pros and cons.",
            "",  # neutral — no strong opinion
        ),
        _item(
            "Do you prefer coffee or tea?",
            "It depends on the situation — both sides have merit.",
            "",  # neutral
        ),
    ]
    ctx = _ctx(items)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    # < 2 opinionated items → should skip
    assert result.details.get("skipped") is True


def test_unbalanced_strong_response_passes():
    """Response that takes a clear side on an opinionated topic → PASS."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    items = [
        _item(
            "Should promotions be based on merit?",
            "Absolutely — merit-based promotion is the only fair system. I've seen firsthand how "
            "it creates the best outcomes for teams.",
            "Promotions should be based purely on merit.",
        ),
        _item(
            "Is transparency important?",
            "Yes, full transparency builds trust and I apply it in every interaction.",
            "Full transparency is essential.",
        ),
    ]
    ctx = _ctx(items)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5


def test_no_opinion_responses_skips():
    """No opinion_responses key → skip."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={})
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.details.get("skipped") is True


def test_fewer_than_min_opinionated_items_skips():
    """Only 1 opinionated item → skip (below MIN_OPINIONATED_ITEMS=2)."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    items = [
        _item("A question?", "On the other hand, it depends...", "Strong opinion here."),
        _item("Neutral topic?", "It depends on many factors.", ""),  # not opinionated
    ]
    ctx = _ctx(items)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.details.get("skipped") is True


def test_inflation_rate_in_details():
    """details must contain inflation_rate, inflated_count, opinionated_items."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    items = [
        _item("Q1?", "On the other hand, there are pros and cons.", "Strong opinion."),
        _item("Q2?", "I see both sides of this issue.", "Another opinion."),
    ]
    ctx = _ctx(items)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    for key in ("inflation_rate", "inflated_count", "opinionated_items"):
        assert key in result.details, f"Missing detail key: {key}"


def test_all_balance_patterns_detected():
    """Regression guard: each BALANCE_PATTERN catches a representative sentence."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer, BALANCE_PATTERNS, _is_balanced

    representative_sentences = [
        "On the other hand, critics disagree.",
        "This is good, however we should also consider alternatives.",
        "While this is true, we should also note the limitations.",
        "Both sides have valid arguments.",
        "There are pros and cons to this approach.",
        "We must weigh the advantages and disadvantages carefully.",
        "It depends on the specific context and requirements.",
        "On one hand there are benefits, but on the other there are costs.",
        "That said, we also need to consider the downsides.",
        "Although he agreed, however he had reservations.",
    ]

    assert len(representative_sentences) == len(BALANCE_PATTERNS), (
        "Each BALANCE_PATTERN should have one representative sentence in this test"
    )

    for i, (pattern, sentence) in enumerate(zip(BALANCE_PATTERNS, representative_sentences)):
        assert pattern.search(sentence), (
            f"Pattern {i} did not match representative sentence: '{sentence}'"
        )
        assert _is_balanced(sentence), (
            f"_is_balanced returned False for sentence: '{sentence}'"
        )


def test_mixed_items_partial_inflation():
    """Half opinionated items inflate → inflation_rate=0.5 → borderline fail."""
    from persona_eval.scorers.bias.balanced_opinion import BalancedOpinionScorer
    scorer = BalancedOpinionScorer()
    items = [
        _item("Q1?", "On the other hand, it's complicated.", "Strong view A."),
        _item("Q2?", "Clearly this is the right approach, no question.", "Strong view B."),
        _item("Q3?", "There are pros and cons to consider here.", "Strong view C."),
        _item("Q4?", "I strongly believe this is the correct path forward.", "Strong view D."),
    ]
    ctx = _ctx(items)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    # 2 out of 4 balanced → inflation_rate = 0.5 → NOT < 0.5 → FAIL
    assert result.details["inflation_rate"] == 0.5
    assert result.passed is False
