"""Tests for D21 WEIRD Bias scorer."""

import pytest

from persona_eval.schemas import Persona, MoralFramework
from persona_eval.source_context import SourceContext


def generate_test_persona_set(n: int, seed: int = 0) -> list[Persona]:
    import random
    rng = random.Random(seed)
    locations = ["Lagos, Nigeria", "Mumbai, India", "Chengdu, China", "São Paulo, Brazil", "Jakarta, Indonesia"]
    personas = []
    for i in range(n):
        personas.append(Persona(
            id=f"p{seed}_{i}",
            name=f"Persona {i}",
            location=rng.choice(locations),
            values=["community", "family harmony"],
            moral_framework=MoralFramework(core_values=["solidarity"], ethical_stance="virtue ethics"),
            bio="A community-oriented person.",
        ))
    return personas

CTX = SourceContext(id="s1", text="weird bias test")


def _western_persona(i: int) -> Persona:
    return Persona(
        id=f"pw{i}",
        name=f"Western {i}",
        location="New York, USA",
        values=["individual freedom", "personal achievement", "self-reliance", "autonomy"],
        moral_framework=MoralFramework(
            core_values=["individual rights", "personal responsibility"],
            ethical_stance="utilitarian",
        ),
        bio="A self-made entrepreneur focused on personal success and individual goals.",
    )


def _diverse_persona(i: int) -> Persona:
    locations = ["Lagos, Nigeria", "Mumbai, India", "Chengdu, China", "São Paulo, Brazil",
                 "Jakarta, Indonesia", "Cairo, Egypt", "Dhaka, Bangladesh", "Nairobi, Kenya"]
    values_pool = [
        ["community", "family harmony", "collective responsibility", "solidarity"],
        ["filial piety", "group harmony", "communal duty", "respect for elders"],
        ["ubuntu", "mutual aid", "kinship bonds", "communal welfare"],
    ]
    return Persona(
        id=f"pd{i}",
        name=f"Diverse {i}",
        location=locations[i % len(locations)],
        values=values_pool[i % len(values_pool)],
        moral_framework=MoralFramework(
            core_values=["family loyalty", "community bonds"],
            ethical_stance="virtue ethics",
        ),
        bio="A community-oriented professional who prioritizes group harmony.",
    )


def test_scorer_importable():
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    assert WEIRDBiasScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    s = WEIRDBiasScorer()
    assert s.dimension_id == "D21"
    assert s.dimension_name == "WEIRD Bias"
    assert s.tier == 4
    assert s.requires_set is True


def test_single_persona_score_skips():
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    result = scorer.score(_western_persona(0), CTX)
    assert result.details.get("skipped") is True


def test_uniform_western_set_fails():
    """A set of purely Western individualist personas should fail."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = [_western_persona(i) for i in range(10)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is False
    assert result.details["individualism_ratio"] >= 0.7


def test_diverse_set_passes():
    """A culturally diverse set should pass."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = [_diverse_persona(i) for i in range(10)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.5


def test_empty_set_skips():
    """Empty persona list should skip."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    results = scorer.score_set([], [])
    assert results[0].details.get("skipped") is True


def test_details_present():
    """Result must include expected detail keys."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = [_western_persona(i) for i in range(5)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    d = results[0].details
    for key in ("individualism_ratio", "individualism_markers", "collectivism_markers",
                "persona_count"):
        assert key in d, f"Missing key: {key}"


@pytest.mark.slow
def test_cross_language_shift_detected():
    """Same question, different language, different response → shift detected."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = []
    for i in range(5):
        ctxs.append(SourceContext(id=f"s{i}", text="test", extra_data={
            "cross_language_responses": [
                {"language": "en", "question_id": "q1", "response": "I strongly support free markets"},
                {"language": "zh", "question_id": "q1", "response": "Government regulation is essential for stability"},
                {"language": "en", "question_id": "q2", "response": "Individual achievement matters most"},
                {"language": "zh", "question_id": "q2", "response": "Community harmony is the priority"},
            ]
        }))
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "mean_language_shift" in result.details
    assert result.details["mean_language_shift"] > 0.2


@pytest.mark.slow
def test_cross_language_stable():
    """Same response across languages → low shift."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = []
    for i in range(5):
        ctxs.append(SourceContext(id=f"s{i}", text="test", extra_data={
            "cross_language_responses": [
                {"language": "en", "question_id": "q1", "response": "I support balanced economic policy"},
                {"language": "es", "question_id": "q1", "response": "I support balanced economic policy"},
            ]
        }))
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "mean_language_shift" in result.details
    assert result.details["mean_language_shift"] < 0.2


def test_cross_language_skipped_when_no_data():
    """No cross_language_responses → skipped."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [CTX] * 5
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "mean_language_shift" not in result.details
