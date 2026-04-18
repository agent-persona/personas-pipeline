"""Tests for D9 Knowledge Boundary Awareness scorer."""

from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX_EMPTY = SourceContext(id="s1", text="test")


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
    assert KnowledgeBoundaryScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
    s = KnowledgeBoundaryScorer()
    assert s.dimension_id == "D9"
    assert s.tier == 2


def test_appropriate_uncertainty_passes():
    from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
    scorer = KnowledgeBoundaryScorer()
    persona = Persona(id="p1", name="PM", knowledge_domains=["product strategy"])
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ood_responses": [
            {"domain": "quantum_physics", "response": "I'm not sure about quantum physics, that's outside my expertise."},
            {"domain": "neurosurgery", "response": "I don't know much about neurosurgery, you'd be better off asking a doctor."},
            {"domain": "marine_biology", "response": "I'm not familiar with marine biology at all."},
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.score >= 0.6


def test_overconfident_responses_fail():
    from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
    scorer = KnowledgeBoundaryScorer()
    persona = Persona(id="p2", name="Overconfident")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ood_responses": [
            {"domain": "quantum_physics", "response": "Quantum entanglement means two particles share state instantaneously."},
            {"domain": "neurosurgery", "response": "The best approach for a craniotomy is a pterional incision."},
            {"domain": "marine_biology", "response": "Coral bleaching occurs when water temperatures rise above 30C."},
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_responses_skips():
    from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
    scorer = KnowledgeBoundaryScorer()
    persona = Persona(id="p3", name="No data")
    result = scorer.score(persona, CTX_EMPTY)
    assert result.details.get("skipped") is True


def test_mixed_responses_partial_score():
    from evaluation.testing.scorers.semantic.knowledge_boundary import KnowledgeBoundaryScorer
    scorer = KnowledgeBoundaryScorer()
    persona = Persona(id="p4", name="Mixed")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ood_responses": [
            {"domain": "quantum_physics", "response": "I'm not sure about that topic."},
            {"domain": "neurosurgery", "response": "The frontal lobe handles executive function."},
            {"domain": "marine_biology", "response": "I wouldn't know anything about marine biology."},
            {"domain": "law", "response": "The constitution guarantees freedom of speech."},
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.score == 0.5
