import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext
from persona_eval.scorers.structural.completeness import CompletenessScorer

CTX = SourceContext(id="s1", text="any source")


@pytest.fixture
def scorer():
    return CompletenessScorer()


def _full_persona() -> Persona:
    return Persona(
        id="p1",
        name="Alice Nguyen",
        occupation="Product Manager",
        industry="SaaS",
        location="San Francisco, CA",
        education="BS Computer Science",
        lifestyle="Remote-first, active commuter",
        bio="Alice has 8 years of experience shipping B2B SaaS products. She focuses on reducing churn and growing NRR through tight collaboration with engineering.",
        goals=["Ship new onboarding flow", "Reduce support tickets by 30%"],
        pain_points=["Too many stakeholders", "Unclear product vision"],
        values=["User empathy", "Data-driven decisions"],
        knowledge_domains=["Product management", "SaaS metrics", "Agile"],
        personality_traits=["Analytical", "Collaborative"],
        behaviors=["Reads product analytics daily", "Weekly 1:1s with engineers"],
    )


def test_full_persona_passes(scorer):
    result = scorer.score(_full_persona(), CTX)
    assert result.passed is True
    assert result.score == 1.0


def test_empty_bio_fails(scorer):
    p = _full_persona()
    p.bio = ""
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("bio" in e for e in result.errors)


def test_filler_value_fails(scorer):
    p = _full_persona()
    p.occupation = "N/A"
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("occupation" in e for e in result.errors)


def test_filler_case_insensitive(scorer):
    p = _full_persona()
    p.industry = "not specified"
    result = scorer.score(p, CTX)
    assert result.passed is False


def test_empty_goals_list_fails(scorer):
    p = _full_persona()
    p.goals = []
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("goals" in e for e in result.errors)


def test_bio_too_short_fails(scorer):
    p = _full_persona()
    p.bio = "Short bio."
    result = scorer.score(p, CTX)
    assert result.passed is False
    assert any("bio" in e for e in result.errors)


def test_score_degrades_with_errors(scorer):
    p = _full_persona()
    p.goals = []
    p.pain_points = []
    p.bio = ""
    result = scorer.score(p, CTX)
    assert result.score < 1.0
