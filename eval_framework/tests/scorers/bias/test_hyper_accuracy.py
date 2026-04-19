"""Tests for D22 Hyper-Accuracy Distortion scorer."""

from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext

CTX = SourceContext(id="s1", text="accuracy test")


def _ctx_with_answers(answers: list[dict]) -> SourceContext:
    """answers: list of {"question": str, "persona_answer": str, "correct_answer": str, "human_accuracy": float}"""
    return SourceContext(id="s1", text="accuracy test", extra_data={"factual_answers": answers})


# Realistic human-like accuracy (~70%)
HUMAN_LIKE_ANSWERS = [
    {"question": "Capital of France?", "persona_answer": "Paris", "correct_answer": "Paris", "human_accuracy": 0.95},
    {"question": "Year WWI started?", "persona_answer": "1914", "correct_answer": "1914", "human_accuracy": 0.65},
    {"question": "Chemical symbol for gold?", "persona_answer": "Ag", "correct_answer": "Au", "human_accuracy": 0.50},
    {"question": "Deepest ocean trench?", "persona_answer": "Mariana Trench", "correct_answer": "Mariana Trench", "human_accuracy": 0.40},
    {"question": "Population of Iceland?", "persona_answer": "500000", "correct_answer": "370000", "human_accuracy": 0.15},
]

# Unrealistically perfect answers (100% accuracy on hard questions)
HYPER_ACCURATE_ANSWERS = [
    {"question": "Capital of France?", "persona_answer": "Paris", "correct_answer": "Paris", "human_accuracy": 0.95},
    {"question": "Year WWI started?", "persona_answer": "1914", "correct_answer": "1914", "human_accuracy": 0.65},
    {"question": "Chemical symbol for gold?", "persona_answer": "Au", "correct_answer": "Au", "human_accuracy": 0.50},
    {"question": "Deepest ocean trench?", "persona_answer": "Mariana Trench", "correct_answer": "Mariana Trench", "human_accuracy": 0.40},
    {"question": "Population of Iceland?", "persona_answer": "370000", "correct_answer": "370000", "human_accuracy": 0.15},
]

SAMPLE_PERSONA = Persona(id="p1", name="Alice")


def test_scorer_importable():
    from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
    assert HyperAccuracyScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
    s = HyperAccuracyScorer()
    assert s.dimension_id == "D22"
    assert s.dimension_name == "Hyper-Accuracy Distortion"
    assert s.tier == 4
    assert s.requires_set is False


def test_human_like_accuracy_passes():
    """Persona with realistic accuracy should pass."""
    from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
    scorer = HyperAccuracyScorer()
    ctx = _ctx_with_answers(HUMAN_LIKE_ANSWERS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "persona_accuracy" in result.details


def test_hyper_accurate_fails():
    """Persona that's too accurate on hard questions should fail."""
    from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
    scorer = HyperAccuracyScorer()
    ctx = _ctx_with_answers(HYPER_ACCURATE_ANSWERS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is False
    assert result.details["persona_accuracy"] > result.details["expected_accuracy"]


def test_no_answers_skips():
    """Missing factual_answers should skip."""
    from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
    scorer = HyperAccuracyScorer()
    result = scorer.score(SAMPLE_PERSONA, CTX)
    assert result.details.get("skipped") is True


def test_details_present():
    """Result details must include expected keys."""
    from persona_eval.scorers.bias.hyper_accuracy import HyperAccuracyScorer
    scorer = HyperAccuracyScorer()
    ctx = _ctx_with_answers(HUMAN_LIKE_ANSWERS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    for key in ("persona_accuracy", "expected_accuracy", "accuracy_gap", "question_count"):
        assert key in result.details, f"Missing key: {key}"
