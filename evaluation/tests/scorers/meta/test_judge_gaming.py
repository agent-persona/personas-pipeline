"""Tests for M2 Judge Gaming Prevention scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="meta test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    assert JudgeGamingPreventionScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    s = JudgeGamingPreventionScorer()
    assert s.dimension_id == "M2"
    assert s.tier == 8
    assert s.requires_set is False


def test_good_detection_passes():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    scorer = JudgeGamingPreventionScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "adversarial_tests": [
            {"is_bad": True, "judge_caught": True},
            {"is_bad": True, "judge_caught": True},
            {"is_bad": True, "judge_caught": True},
            {"is_bad": False, "judge_caught": False},
            {"is_bad": False, "judge_caught": False},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert result.details["detection_rate"] == 1.0


def test_poor_detection_fails():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    scorer = JudgeGamingPreventionScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "adversarial_tests": [
            {"is_bad": True, "judge_caught": False},
            {"is_bad": True, "judge_caught": False},
            {"is_bad": True, "judge_caught": True},
            {"is_bad": False, "judge_caught": False},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.details["detection_rate"] < 0.7


def test_no_adversarial_data_skips():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    scorer = JudgeGamingPreventionScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_cross_family_agreement_reported():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    scorer = JudgeGamingPreventionScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "adversarial_tests": [
            {"is_bad": True, "judge_caught": True},
            {"is_bad": True, "judge_caught": True},
            {"is_bad": False, "judge_caught": False},
        ],
        "cross_family_agreement": 0.85,
    })
    result = scorer.score(p, ctx)
    assert result.details["cross_family_agreement"] == 0.85


def test_false_positive_rate_reported():
    from evaluation.testing.scorers.meta.judge_gaming import JudgeGamingPreventionScorer
    scorer = JudgeGamingPreventionScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "adversarial_tests": [
            {"is_bad": True, "judge_caught": True},
            {"is_bad": False, "judge_caught": True},  # False positive
            {"is_bad": False, "judge_caught": False},
        ]
    })
    result = scorer.score(p, ctx)
    assert "false_positive_rate" in result.details
    assert result.details["false_positive_rate"] == pytest.approx(0.5, abs=0.01)
