"""Tests for D39 Reproducibility scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    assert ReproducibilityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    s = ReproducibilityScorer()
    assert s.dimension_id == "D39"
    assert s.tier == 6
    assert s.requires_set is False


def test_identical_runs_pass():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    scorer = ReproducibilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "run_outputs": [
            {"occupation": "analyst", "age": 32, "bio": "Data analyst at a tech firm."},
            {"occupation": "analyst", "age": 32, "bio": "Data analyst at a tech firm."},
            {"occupation": "analyst", "age": 32, "bio": "Data analyst at a tech firm."},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["structured_variance"] == 0.0


def test_high_variance_fails():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    scorer = ReproducibilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "run_outputs": [
            {"occupation": "analyst", "age": 25, "bio": "Young data analyst."},
            {"occupation": "engineer", "age": 40, "bio": "Senior software engineer."},
            {"occupation": "designer", "age": 33, "bio": "Creative UX designer."},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_run_outputs_skips():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    scorer = ReproducibilityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_single_run_skips():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    scorer = ReproducibilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "run_outputs": [
            {"occupation": "analyst", "age": 32},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_structured_zero_variance_narrative_bounded():
    from evaluation.testing.scorers.system.reproducibility import ReproducibilityScorer
    scorer = ReproducibilityScorer()
    p = _make_persona()
    # Structured fields identical, narrative fields vary slightly
    ctx = SourceContext(id="s1", text="test", extra_data={
        "run_outputs": [
            {"occupation": "analyst", "age": 32, "bio": "Data analyst who loves numbers."},
            {"occupation": "analyst", "age": 32, "bio": "Analyst focused on data insights."},
            {"occupation": "analyst", "age": 32, "bio": "Data analyst working with metrics."},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.details["structured_variance"] == 0.0
    assert result.details["narrative_variance"] > 0.0
