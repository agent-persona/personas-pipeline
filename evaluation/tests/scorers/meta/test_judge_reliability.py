"""Tests for M1 LLM-as-Judge Reliability scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="meta test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    assert JudgeReliabilityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    s = JudgeReliabilityScorer()
    assert s.dimension_id == "M1"
    assert s.tier == 8
    assert s.requires_set is False


def test_high_correlation_passes():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    scorer = JudgeReliabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "judge_scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "human_scores": [0.88, 0.82, 0.72, 0.58, 0.52, 0.38, 0.32],
        "dimension_name": "D1",
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert result.details["trust_level"] in ("high", "medium")


def test_low_correlation_fails():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    scorer = JudgeReliabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "judge_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "human_scores": [0.3, 0.9, 0.2, 0.8, 0.4],
        "dimension_name": "D12",
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.details["trust_level"] in ("low", "unreliable")


def test_no_scores_skips():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    scorer = JudgeReliabilityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_too_few_samples_skips():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    scorer = JudgeReliabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "judge_scores": [0.9, 0.8],
        "human_scores": [0.88, 0.82],
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_mismatched_lengths_skips():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    scorer = JudgeReliabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "judge_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "human_scores": [0.88, 0.82],
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_constant_scores_handled():
    from evaluation.testing.scorers.meta.judge_reliability import JudgeReliabilityScorer
    scorer = JudgeReliabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "judge_scores": [0.5, 0.5, 0.5, 0.5, 0.5],
        "human_scores": [0.3, 0.5, 0.7, 0.9, 0.1],
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score == 0.0
