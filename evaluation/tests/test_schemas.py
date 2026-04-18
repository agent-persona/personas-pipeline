import pytest
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext


def test_persona_minimal():
    p = Persona(id="p1", name="Alice")
    assert p.id == "p1"
    assert p.goals == []


def test_persona_full():
    p = Persona(
        id="p2",
        name="Bob",
        age=35,
        occupation="Engineer",
        experience_years=10,
        goals=["ship product", "grow team"],
        pain_points=["too many meetings"],
    )
    assert p.experience_years == 10
    assert len(p.goals) == 2


def test_eval_result_score_bounds():
    r = EvalResult(
        dimension_id="D1",
        dimension_name="Schema Compliance",
        persona_id="p1",
        passed=True,
        score=1.0,
    )
    assert r.score == 1.0


def test_eval_result_invalid_score():
    with pytest.raises(Exception):
        EvalResult(
            dimension_id="D1",
            dimension_name="Schema Compliance",
            persona_id="p1",
            passed=False,
            score=1.5,
        )


def test_source_context_chunking():
    ctx = SourceContext(id="s1", text="word " * 1000)
    chunks = ctx.get_chunks(max_chunk_size=100)
    assert len(chunks) == 10
    assert all(len(c.split()) <= 100 for c in chunks)
