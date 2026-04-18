"""Tests for D44 Sparse vs Dense Coverage scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="generation test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    assert SparseDenseCoverageScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    s = SparseDenseCoverageScorer()
    assert s.dimension_id == "D44"
    assert s.tier == 7
    assert s.requires_set is True


def test_good_coverage_passes():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    scorer = SparseDenseCoverageScorer()
    personas = [_make_persona(f"p{i}") for i in range(5)]
    # Each conversation covers most attributes
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "coverage_matrix": {
                "communication_style": True,
                "emotional_tone": True,
                "values": True,
                "knowledge_domains": True,
                "moral_framework": True,
            }
        }) for i in range(5)
    ]
    results = scorer.score_set(personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.8
    assert "coverage_rate" in result.details


def test_sparse_coverage_fails():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    scorer = SparseDenseCoverageScorer()
    personas = [_make_persona(f"p{i}") for i in range(5)]
    # Each conversation only covers 1 attribute
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "coverage_matrix": {
                "communication_style": i == 0,
                "emotional_tone": i == 1,
                "values": i == 2,
                "knowledge_domains": i == 3,
                "moral_framework": i == 4,
            }
        }) for i in range(5)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.score < 0.5


def test_no_coverage_data_skips():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    scorer = SparseDenseCoverageScorer()
    personas = [_make_persona("p1")]
    ctxs = [CTX]
    results = scorer.score_set(personas, ctxs)
    assert results[0].details.get("skipped") is True


def test_single_persona_score_skips():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    scorer = SparseDenseCoverageScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_sparse_dimensions_identified():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    scorer = SparseDenseCoverageScorer()
    personas = [_make_persona(f"p{i}") for i in range(5)]
    # moral_framework only covered once (sparse)
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "coverage_matrix": {
                "communication_style": True,
                "emotional_tone": True,
                "values": True,
                "moral_framework": i == 0,
            }
        }) for i in range(5)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "sparse_dimensions" in result.details
    assert "moral_framework" in result.details["sparse_dimensions"]


def test_empty_personas_skips():
    from evaluation.testing.scorers.generation.sparse_dense_coverage import SparseDenseCoverageScorer
    scorer = SparseDenseCoverageScorer()
    results = scorer.score_set([], [])
    assert results[0].details.get("skipped") is True
