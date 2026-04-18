"""Tests for D35 Role Identifiability scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str, name: str, occupation: str) -> Persona:
    return Persona(id=pid, name=name, occupation=occupation)


def test_scorer_importable():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    assert RoleIdentifiabilityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    s = RoleIdentifiabilityScorer()
    assert s.dimension_id == "D35"
    assert s.tier == 6
    assert s.requires_set is True


def test_high_accuracy_passes():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    scorer = RoleIdentifiabilityScorer()
    personas = [_make_persona(f"p{i}", f"Person {i}", "engineer") for i in range(5)]
    # 4 out of 5 correct = 80% accuracy
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "identification_result": {"true_id": f"p{i}", "predicted_id": f"p{i}"}
        }) for i in range(4)
    ] + [
        SourceContext(id="s4", text="test", extra_data={
            "identification_result": {"true_id": "p4", "predicted_id": "p0"}
        })
    ]
    results = scorer.score_set(personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score == pytest.approx(0.8, abs=0.01)
    assert result.details["accuracy"] == pytest.approx(0.8, abs=0.01)
    assert result.details["correct"] == 4
    assert result.details["total"] == 5


def test_low_accuracy_fails():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    scorer = RoleIdentifiabilityScorer()
    personas = [_make_persona(f"p{i}", f"Person {i}", "engineer") for i in range(5)]
    # 1 out of 5 correct = 20%
    ctxs = [
        SourceContext(id="s0", text="test", extra_data={
            "identification_result": {"true_id": "p0", "predicted_id": "p0"}
        })
    ] + [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "identification_result": {"true_id": f"p{i}", "predicted_id": "p0"}
        }) for i in range(1, 5)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.score < 0.5


def test_no_identification_data_skips():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    scorer = RoleIdentifiabilityScorer()
    personas = [_make_persona("p1", "Alice", "engineer")]
    ctxs = [CTX]
    results = scorer.score_set(personas, ctxs)
    assert results[0].details.get("skipped") is True


def test_single_persona_score_skips():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    scorer = RoleIdentifiabilityScorer()
    p = _make_persona("p1", "Alice", "engineer")
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_perfect_accuracy():
    from evaluation.testing.scorers.system.role_identifiability import RoleIdentifiabilityScorer
    scorer = RoleIdentifiabilityScorer()
    personas = [_make_persona(f"p{i}", f"Person {i}", "engineer") for i in range(3)]
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "identification_result": {"true_id": f"p{i}", "predicted_id": f"p{i}"}
        }) for i in range(3)
    ]
    results = scorer.score_set(personas, ctxs)
    assert results[0].passed is True
    assert results[0].score == 1.0
