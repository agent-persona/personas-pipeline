import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext
from evaluation.testing.scorers.structural.schema_compliance import SchemaComplianceScorer

CTX = SourceContext(id="s1", text="any source")


@pytest.fixture
def scorer():
    return SchemaComplianceScorer()


def test_valid_persona_passes(scorer):
    p = Persona(id="p1", name="Alice", age=30, experience_years=5)
    result = scorer.score(p, CTX)
    assert result.passed is True
    assert result.score == 1.0
    assert result.errors == []


def test_invalid_age_fails(scorer):
    p = Persona(id="p1", name="Alice", age=200)
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("age" in e for e in result.errors)


def test_negative_experience_fails(scorer):
    p = Persona(id="p1", name="Alice", experience_years=-1)
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("experience_years" in e for e in result.errors)


def test_multiple_errors_reduces_score(scorer):
    p = Persona(id="p1", name="Alice", age=999, experience_years=-5)
    result = scorer.score(p, CTX)
    assert result.score < 1.0
    assert len(result.errors) >= 2


@given(
    age=st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
    experience=st.one_of(st.none(), st.integers(min_value=0, max_value=50)),
)
@settings(max_examples=50)
def test_hypothesis_valid_personas_pass(age, experience):
    scorer = SchemaComplianceScorer()
    p = Persona(id="p1", name="Test", age=age, experience_years=experience)
    result = scorer.score(p, CTX)
    assert result.passed is True
