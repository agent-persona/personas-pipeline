"""Tests for J3 ValueAlignmentScorer."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from persona_eval.scorers.judge.j3_value_alignment import ValueAlignmentScorer
from persona_eval.source_context import SourceContext


def test_scorer_metadata():
    s = ValueAlignmentScorer()
    assert s.dimension_id == "J3"
    assert s.dimension_name == "Value Alignment"
    assert s.tier == 3
    assert s.requires_set is False


def test_no_responses_skips(judge_persona):
    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True
    assert result.score == 1.0


def test_empty_list_skips(judge_persona):
    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": []})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True


def test_malformed_llm_output_degrades_gracefully(judge_persona, judge_responses):
    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.details["parse_ok"] is False
    assert result.score == 0.5


def test_score_4_passes(judge_persona, judge_responses, mock_llm_score_4):
    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is True
    assert result.score == pytest.approx(0.75)


def test_score_2_fails(judge_persona, judge_responses, mock_llm_score_2):
    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is False


def test_values_highlighted_in_prompt(judge_persona, judge_responses):
    """User message must contain 'STATED VALUES:' with persona's values."""
    captured_messages = []

    def capturing_complete(messages, **kwargs):
        captured_messages.extend(messages)
        return "I am a mock LLM response."

    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})

    with patch.object(scorer.__class__, "score", wraps=scorer.score):
        from persona_eval.llm_client import LLMClient
        original_complete = LLMClient.complete

        def capturing_complete_method(self, messages, **kwargs):
            captured_messages.extend(messages)
            return "I am a mock LLM response."

        with patch.object(LLMClient, "complete", capturing_complete_method):
            scorer.score(judge_persona, ctx)

    user_content = next((m["content"] for m in captured_messages if m["role"] == "user"), "")
    assert "STATED VALUES:" in user_content
    for value in judge_persona.values:
        assert value in user_content


def test_ethical_stance_in_prompt(judge_persona, judge_responses):
    """User message must contain 'ETHICAL STANCE:' with persona's ethical stance."""
    captured_messages = []

    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})

    from persona_eval.llm_client import LLMClient

    def capturing_complete_method(self, messages, **kwargs):
        captured_messages.extend(messages)
        return "I am a mock LLM response."

    with patch.object(LLMClient, "complete", capturing_complete_method):
        scorer.score(judge_persona, ctx)

    user_content = next((m["content"] for m in captured_messages if m["role"] == "user"), "")
    assert "ETHICAL STANCE:" in user_content


def test_details_contain_required_keys(judge_persona, judge_responses):
    scorer = ValueAlignmentScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    for key in ("raw_score", "reasoning", "parse_ok", "response_count", "rubric"):
        assert key in result.details
