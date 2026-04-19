"""Tests for J4 PersonaDepthScorer."""
from __future__ import annotations

import pytest
from unittest.mock import patch
from persona_eval.scorers.judge.j4_persona_depth import PersonaDepthScorer
from persona_eval.source_context import SourceContext


def test_scorer_metadata():
    s = PersonaDepthScorer()
    assert s.dimension_id == "J4"
    assert s.dimension_name == "Persona Depth"
    assert s.tier == 3
    assert s.requires_set is False


def test_no_responses_skips(judge_persona):
    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True
    assert result.score == 1.0


def test_empty_list_skips(judge_persona):
    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": []})
    result = scorer.score(judge_persona, ctx)
    assert result.details.get("skipped") is True


def test_malformed_llm_output_degrades_gracefully(judge_persona, judge_responses):
    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.details["parse_ok"] is False
    assert result.score == 0.5


def test_score_4_passes(judge_persona, judge_responses, mock_llm_score_4):
    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is True
    assert result.score == pytest.approx(0.75)


def test_score_2_fails(judge_persona, judge_responses, mock_llm_score_2):
    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    assert result.passed is False


def test_bio_highlighted_in_prompt(judge_persona, judge_responses):
    """User message must contain 'BIO (key source' and the persona's bio text."""
    captured_messages = []

    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})

    from persona_eval.llm_client import LLMClient

    def capturing_complete_method(self, messages, **kwargs):
        captured_messages.extend(messages)
        return "I am a mock LLM response."

    with patch.object(LLMClient, "complete", capturing_complete_method):
        scorer.score(judge_persona, ctx)

    user_content = next((m["content"] for m in captured_messages if m["role"] == "user"), "")
    assert "BIO (key source" in user_content
    assert judge_persona.bio in user_content


def test_details_contain_required_keys(judge_persona, judge_responses):
    scorer = PersonaDepthScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": judge_responses})
    result = scorer.score(judge_persona, ctx)
    for key in ("raw_score", "reasoning", "parse_ok", "response_count", "rubric"):
        assert key in result.details
