"""Tests for J2 VoiceConsistencyScorer."""
from __future__ import annotations

import pytest
from persona_eval.scorers.judge.j2_voice_consistency import VoiceConsistencyScorer
from persona_eval.source_context import SourceContext


def test_scorer_metadata():
    s = VoiceConsistencyScorer()
    assert s.dimension_id == "J2"
    assert s.dimension_name == "Voice Consistency"
    assert s.tier == 3
    assert s.requires_set is False


def test_no_responses_skips(judge_persona):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True
    assert result.score == 1.0
    assert result.passed is True


def test_empty_list_skips(judge_persona):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": []})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True


def test_malformed_llm_output_degrades_gracefully(judge_persona, judge_responses):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.details["parse_ok"] is False
    assert result.score == 0.5
    assert result.passed is False


def test_score_4_passes(judge_persona, judge_responses, mock_llm_score_4):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is True
    assert result.score == pytest.approx(0.75)


def test_score_2_fails(judge_persona, judge_responses, mock_llm_score_2):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is False
    assert result.score == pytest.approx(0.25)


def test_details_contain_required_keys(judge_persona, judge_responses):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    for key in ("raw_score", "reasoning", "parse_ok", "response_count", "rubric"):
        assert key in result.details


def test_caps_at_5_responses(judge_persona, mock_llm_score_3):
    scorer = VoiceConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": [f"R{i}" for i in range(10)]})
    result = scorer.score(judge_persona, ctx)
    assert result.details["response_count"] <= 5
