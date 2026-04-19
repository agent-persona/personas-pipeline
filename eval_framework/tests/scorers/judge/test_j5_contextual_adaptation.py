"""Tests for J5 ContextualAdaptationScorer."""
from __future__ import annotations

import pytest
from persona_eval.scorers.judge.j5_contextual_adaptation import ContextualAdaptationScorer
from persona_eval.source_context import SourceContext

CONTEXTUAL_RESPONSES = [
    {"context": "formal meeting", "response": "Good morning. I'll keep this brief."},
    {"context": "casual chat", "response": "Hey! Yeah this week was brutal honestly."},
    {"context": "technical question", "response": "The issue here is lock contention at scale."},
]


def test_scorer_metadata():
    s = ContextualAdaptationScorer()
    assert s.dimension_id == "J5"
    assert s.dimension_name == "Contextual Adaptation"
    assert s.tier == 3
    assert s.requires_set is False


def test_neither_key_skips(judge_persona):
    """No contextual_responses and no responses → skipped=True."""
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True
    assert result.score == 1.0
    assert result.passed is True


def test_empty_contextual_and_no_flat_skips(judge_persona):
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"contextual_responses": []})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True


def test_contextual_responses_used(judge_persona, mock_llm_score_3):
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"contextual_responses": CONTEXTUAL_RESPONSES})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is not True
    assert result.details["response_count"] == len(CONTEXTUAL_RESPONSES)


def test_contextual_responses_takes_precedence(judge_persona, mock_llm_score_3):
    """When both keys present, contextual_responses is used (not flat responses)."""
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "contextual_responses": CONTEXTUAL_RESPONSES,
        "responses": ["flat1", "flat2", "flat3", "flat4", "flat5", "flat6"],
    })
    result = scorer.score(judge_persona, ctx)
    # response_count must match contextual count (3), not flat count (6)
    assert result.details["response_count"] == len(CONTEXTUAL_RESPONSES)


def test_fallback_to_flat_responses(judge_persona, judge_responses, mock_llm_score_3):
    """Only flat responses key → does not skip, uses them."""
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is not True
    assert result.details["response_count"] == len(judge_responses)


def test_malformed_llm_output_degrades_gracefully(judge_persona):
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"contextual_responses": CONTEXTUAL_RESPONSES})
    result = scorer.score(judge_persona, ctx)
    assert result.details["parse_ok"] is False
    assert result.score == 0.5


def test_score_4_passes(judge_persona, mock_llm_score_4):
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"contextual_responses": CONTEXTUAL_RESPONSES})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is True
    assert result.score == pytest.approx(0.75)


def test_score_2_fails(judge_persona, mock_llm_score_2):
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"contextual_responses": CONTEXTUAL_RESPONSES})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is False


def test_details_contain_required_keys(judge_persona):
    scorer = ContextualAdaptationScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"contextual_responses": CONTEXTUAL_RESPONSES})
    result = scorer.score(judge_persona, ctx)
    for key in ("raw_score", "reasoning", "parse_ok", "response_count", "rubric"):
        assert key in result.details
