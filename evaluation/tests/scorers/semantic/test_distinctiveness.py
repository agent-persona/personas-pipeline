"""Tests for D6 Distinctiveness scorer."""

import pytest
from evaluation.testing.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


CTX = SourceContext(id="s1", text="distributional test")


@pytest.fixture
def diverse_personas():
    return generate_test_persona_set(n=20, seed=42)


@pytest.fixture
def homogeneous_personas():
    return generate_homogeneous_set(n=20)


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
    assert DistinctivenessScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
    s = DistinctivenessScorer()
    assert s.dimension_id == "D6"
    assert s.tier == 2
    assert s.requires_set is True


@pytest.mark.slow
def test_diverse_personas_are_distinct(diverse_personas):
    from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
    scorer = DistinctivenessScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.3
    assert "mean_pairwise_distance" in result.details


@pytest.mark.slow
def test_clones_are_not_distinct(homogeneous_personas):
    from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
    scorer = DistinctivenessScorer()
    ctxs = [CTX] * len(homogeneous_personas)
    results = scorer.score_set(homogeneous_personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.score < 0.3
    assert len(result.details["near_duplicates"]) > 0


def test_small_set_skipped():
    from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
    scorer = DistinctivenessScorer()
    personas = generate_test_persona_set(n=1, seed=1)
    results = scorer.score_set(personas, [CTX])
    assert results[0].details.get("skipped") is True


def test_single_persona_score_skips():
    from evaluation.testing.scorers.semantic.distinctiveness import DistinctivenessScorer
    scorer = DistinctivenessScorer()
    personas = generate_test_persona_set(n=1, seed=1)
    result = scorer.score(personas[0], CTX)
    assert result.details.get("skipped") is True
