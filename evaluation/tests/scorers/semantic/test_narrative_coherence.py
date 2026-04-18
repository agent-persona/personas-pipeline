"""Tests for D12 Narrative Coherence scorer."""

from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer
    assert NarrativeCoherenceScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer
    s = NarrativeCoherenceScorer()
    assert s.dimension_id == "D12"
    assert s.tier == 2


def test_high_precomputed_score_passes():
    from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer
    scorer = NarrativeCoherenceScorer()
    persona = Persona(id="p1", name="Alice")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "narrative_score": 4,
        "narrative_reasoning": "Coherent career trajectory and personality.",
    })
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.score == 0.8
    assert result.details["source"] == "pre_computed"
    assert result.details["raw_score"] == 4


def test_low_precomputed_score_fails():
    from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer
    scorer = NarrativeCoherenceScorer()
    persona = Persona(id="p2", name="Bob")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "narrative_score": 2,
    })
    result = scorer.score(persona, ctx)
    assert result.passed is False
    assert result.score == 0.4


def test_no_score_or_llm_skips():
    from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer
    scorer = NarrativeCoherenceScorer()
    persona = Persona(id="p3", name="Charlie")
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True


def test_score_clamped_to_bounds():
    from evaluation.testing.scorers.semantic.narrative_coherence import NarrativeCoherenceScorer
    scorer = NarrativeCoherenceScorer()
    persona = Persona(id="p4", name="Edge")
    # Score of 5 → 1.0, score of 0 → 0.0
    ctx_max = SourceContext(id="s1", text="test", extra_data={"narrative_score": 5})
    result_max = scorer.score(persona, ctx_max)
    assert result_max.score == 1.0

    ctx_zero = SourceContext(id="s2", text="test", extra_data={"narrative_score": 0})
    result_zero = scorer.score(persona, ctx_zero)
    assert result_zero.score == 0.0
