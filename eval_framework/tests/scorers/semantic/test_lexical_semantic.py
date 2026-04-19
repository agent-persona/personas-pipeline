"""Tests for D10 Lexical vs Semantic Generalization scorer."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


def test_scorer_importable():
    from persona_eval.scorers.semantic.lexical_semantic import LexicalSemanticScorer
    assert LexicalSemanticScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.semantic.lexical_semantic import LexicalSemanticScorer
    s = LexicalSemanticScorer()
    assert s.dimension_id == "D10"
    assert s.tier == 2


@pytest.mark.slow
def test_consistent_paraphrases_score_well():
    from persona_eval.scorers.semantic.lexical_semantic import LexicalSemanticScorer
    scorer = LexicalSemanticScorer()
    persona = Persona(id="p1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "response_pairs": [
            {
                "original": "I prefer data-driven approaches to decision making.",
                "paraphrased": "I like to base my choices on evidence and metrics.",
            },
            {
                "original": "Our team uses agile methodology for product development.",
                "paraphrased": "We follow an iterative development process with sprints.",
            },
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.score >= 0.4
    assert "pair_similarities" in result.details


@pytest.mark.slow
def test_inconsistent_paraphrases_score_poorly():
    from persona_eval.scorers.semantic.lexical_semantic import LexicalSemanticScorer
    scorer = LexicalSemanticScorer()
    persona = Persona(id="p1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "response_pairs": [
            {
                "original": "I love working with data and analytics.",
                "paraphrased": "I prefer outdoor activities and hiking.",
            },
            {
                "original": "My team is small and collaborative.",
                "paraphrased": "Quantum physics describes subatomic particles.",
            },
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.score < 0.5


def test_no_pairs_skips():
    from persona_eval.scorers.semantic.lexical_semantic import LexicalSemanticScorer
    scorer = LexicalSemanticScorer()
    persona = Persona(id="p1", name="Test")
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True
