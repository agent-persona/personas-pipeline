"""Tests for J1 BehavioralAuthenticityScorer."""
from __future__ import annotations

import pytest
from persona_eval.scorers.judge.j1_behavioral_authenticity import BehavioralAuthenticityScorer
from persona_eval.source_context import SourceContext


def test_scorer_metadata():
    s = BehavioralAuthenticityScorer()
    assert s.dimension_id == "J1"
    assert s.dimension_name == "Behavioral Authenticity"
    assert s.tier == 3
    assert s.requires_set is False


def test_no_responses_skips(judge_persona):
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True
    assert result.score == 1.0
    assert result.passed is True


def test_empty_list_skips(judge_persona):
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": []})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True


def test_malformed_llm_output_degrades_gracefully(judge_persona, judge_responses):
    """Autouse mock returns 'I am a mock LLM response.' → parse_ok=False, score=0.5."""
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.details["parse_ok"] is False
    assert result.score == 0.5   # normalize(3.0) = 0.5
    assert result.passed is False  # 0.5 < 0.6


def test_score_4_passes(judge_persona, judge_responses, mock_llm_score_4):
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is True
    assert result.score == pytest.approx(0.75)  # normalize(4.0)


def test_score_2_fails(judge_persona, judge_responses, mock_llm_score_2):
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is False
    assert result.score == pytest.approx(0.25)  # normalize(2.0)


def test_details_contain_required_keys(judge_persona, judge_responses):
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert "raw_score" in result.details
    assert "reasoning" in result.details
    assert "parse_ok" in result.details
    assert "response_count" in result.details
    assert "rubric" in result.details


def test_caps_at_5_responses(judge_persona, mock_llm_score_3):
    scorer = BehavioralAuthenticityScorer()
    ten_responses = [f"Response number {i}" for i in range(10)]
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": ten_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.details["response_count"] <= 5


def test_dimension_id_in_result(judge_persona, judge_responses):
    scorer = BehavioralAuthenticityScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.dimension_id == "J1"
    assert result.persona_id == judge_persona.id
