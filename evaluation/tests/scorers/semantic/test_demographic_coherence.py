"""Tests for D7 Demographic Coherence scorer."""

from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="test")


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    assert DemographicCoherenceScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    s = DemographicCoherenceScorer()
    assert s.dimension_id == "D7"
    assert s.tier == 2
    assert s.requires_set is False


def test_plausible_persona_passes(sample_persona):
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    scorer = DemographicCoherenceScorer()
    result = scorer.score(sample_persona, CTX)
    assert result.passed is True
    assert result.score >= 0.8
    assert len(result.details.get("anomalies", [])) == 0


def test_phd_at_19_detected():
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    scorer = DemographicCoherenceScorer()
    persona = Persona(id="p1", name="Young PhD", age=19, education="PhD in Physics")
    result = scorer.score(persona, CTX)
    assert result.passed is False
    anomalies = result.details["anomalies"]
    assert any("PhD" in a["message"] for a in anomalies)


def test_ceo_at_20_detected():
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    scorer = DemographicCoherenceScorer()
    persona = Persona(id="p2", name="Young CEO", age=20, occupation="CEO")
    result = scorer.score(persona, CTX)
    assert result.passed is False
    anomalies = result.details["anomalies"]
    assert any("C-suite" in a["message"] for a in anomalies)


def test_experience_exceeds_age_detected():
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    scorer = DemographicCoherenceScorer()
    persona = Persona(id="p3", name="Impossible XP", age=25, experience_years=20)
    result = scorer.score(persona, CTX)
    assert result.passed is False
    anomalies = result.details["anomalies"]
    assert any("impossible" in a["message"] for a in anomalies)


def test_multiple_anomalies_lower_score():
    from evaluation.testing.scorers.semantic.demographic_coherence import DemographicCoherenceScorer
    scorer = DemographicCoherenceScorer()
    persona = Persona(
        id="p4", name="Multi-anomaly", age=19,
        education="PhD", occupation="CEO", experience_years=15,
    )
    result = scorer.score(persona, CTX)
    assert result.passed is False
    assert result.score < 0.6
    assert len(result.details["anomalies"]) >= 3
