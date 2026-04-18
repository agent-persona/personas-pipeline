"""Tests for D8 Memory Consistency scorer."""

from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    assert MemoryConsistencyScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    s = MemoryConsistencyScorer()
    assert s.dimension_id == "D8"
    assert s.tier == 2


def test_perfect_recall_passes(sample_persona):
    from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    scorer = MemoryConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "answers": {
            "age": "I am 32 years old.",
            "occupation": "I'm a Product Manager.",
            "industry": "I work in SaaS.",
            "experience_years": "I have 8 years of experience.",
        },
    })
    result = scorer.score(sample_persona, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert result.details["correct"] == result.details["total"]


def test_incorrect_recall_fails(sample_persona):
    from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    scorer = MemoryConsistencyScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "answers": {
            "age": "I am 25 years old.",          # Wrong: should be 32
            "occupation": "I'm a Software Engineer.",  # Wrong: should be Product Manager
            "industry": "I work in Healthcare.",    # Wrong: should be SaaS
        },
    })
    result = scorer.score(sample_persona, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_answers_skips(sample_persona):
    from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    scorer = MemoryConsistencyScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(sample_persona, ctx)
    assert result.details.get("skipped") is True


def test_minimal_persona_skips():
    from evaluation.testing.scorers.semantic.memory_consistency import MemoryConsistencyScorer
    scorer = MemoryConsistencyScorer()
    persona = Persona(id="minimal", name="Minimal")
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True
