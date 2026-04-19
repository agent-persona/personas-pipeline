"""Tests for D16 Minority Viewpoint Preservation scorer."""

import pytest
from persona_eval.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


CTX = SourceContext(id="s1", text="distributional test")


@pytest.fixture
def diverse_personas():
    return generate_test_persona_set(n=50, seed=42)


@pytest.fixture
def homogeneous_personas():
    return generate_homogeneous_set(n=50)


def test_scorer_importable():
    from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
    assert MinorityViewpointScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
    s = MinorityViewpointScorer()
    assert s.dimension_id == "D16"
    assert s.tier == 3
    assert s.requires_set is True


def test_diverse_set_has_minorities(diverse_personas):
    from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
    scorer = MinorityViewpointScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.5
    assert "within_group_entropy" in result.details
    # Verify multiple groups were actually analyzed (fixture has 3 genders)
    assert len(result.details["within_group_entropy"]) >= 2


def test_homogeneous_set_lacks_minorities(homogeneous_personas):
    from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
    scorer = MinorityViewpointScorer()
    ctxs = [CTX] * len(homogeneous_personas)
    results = scorer.score_set(homogeneous_personas, ctxs)
    result = results[0]
    # Homogeneous set: all same gender → only 1 group, but within that group
    # all opinions are identical → low entropy → should fail
    assert result.passed is False
    assert result.score < 0.5


def test_small_set_skipped():
    from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
    scorer = MinorityViewpointScorer()
    # Gate is now n < 30; use n=10 to confirm any sub-threshold set is skipped
    personas = generate_test_persona_set(n=10, seed=1)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    assert results[0].details.get("skipped") is True


def test_single_persona_score_skips():
    from persona_eval.scorers.distributional.minority_viewpoint import MinorityViewpointScorer
    scorer = MinorityViewpointScorer()
    personas = generate_test_persona_set(n=1, seed=1)
    result = scorer.score(personas[0], CTX)
    assert result.details.get("skipped") is True
