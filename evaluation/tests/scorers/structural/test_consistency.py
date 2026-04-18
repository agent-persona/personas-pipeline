import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext
from evaluation.testing.scorers.structural.consistency import ConsistencyScorer

CTX = SourceContext(id="s1", text="any source")


@pytest.fixture
def scorer():
    return ConsistencyScorer()


def test_consistent_persona_passes(scorer):
    p = Persona(
        id="p1",
        name="Alice",
        occupation="Senior Product Manager",
        age=35,
        experience_years=10,
    )
    result = scorer.score(p, CTX)
    assert result.passed is True


def test_senior_title_low_experience_fails(scorer):
    p = Persona(
        id="p2",
        name="Bob",
        occupation="Senior Engineer",
        age=22,
        experience_years=1,
    )
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("Senior" in e for e in result.errors)


def test_entry_title_high_experience_fails(scorer):
    p = Persona(
        id="p3",
        name="Carol",
        occupation="Junior Developer",
        experience_years=15,
    )
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("Entry-level" in e for e in result.errors)


def test_age_experience_contradiction_fails(scorer):
    p = Persona(
        id="p4",
        name="Dave",
        occupation="Engineer",
        age=20,
        experience_years=12,
    )
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("age 14" in e for e in result.errors)


def test_budget_luxury_contradiction_fails(scorer):
    p = Persona(
        id="p5",
        name="Fiona",
        occupation="Accountant",
        values=["budget-conscious living"],
        behaviors=["tracks every expense"],
        lifestyle="luxury condo downtown",
        bio="Fiona enjoys luxury travel and fine dining.",
    )
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("budget" in e.lower() for e in result.errors)


def test_introvert_public_speaking_contradiction_fails(scorer):
    p = Persona(
        id="p6",
        name="Greg",
        occupation="Researcher",
        personality_traits=["introvert", "reserved"],
        behaviors=["public speaking at conferences"],
        goals=["become a thought leader"],
    )
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("Introvert" in e for e in result.errors)


def test_multiple_violations_lower_score(scorer):
    p = Persona(
        id="p7",
        name="Eve",
        occupation="Senior Manager",
        age=20,
        experience_years=1,
    )
    result = scorer.score(p, CTX)
    assert result.score < 1.0
