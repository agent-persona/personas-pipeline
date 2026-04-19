"""Tests for D23 Stereotype Amplification scorer."""

from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext

CTX = SourceContext(id="s1", text="stereotype test")


def _persona(i: int, gender: str, occupation: str, interests: list[str]) -> Persona:
    return Persona(
        id=f"p{i}", name=f"Person {i}",
        gender=gender, occupation=occupation, interests=interests,
    )


def test_scorer_importable():
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    assert StereotypeAmplificationScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    s = StereotypeAmplificationScorer()
    assert s.dimension_id == "D23"
    assert s.dimension_name == "Stereotype Amplification"
    assert s.tier == 4
    assert s.requires_set is True


def test_single_persona_skips():
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    scorer = StereotypeAmplificationScorer()
    p = _persona(0, "female", "nurse", ["cooking"])
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_stereotyped_set_fails():
    """A set with perfect demographic-trait correlation should fail."""
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    scorer = StereotypeAmplificationScorer()
    # All women are nurses, all men are engineers
    personas = []
    for i in range(5):
        personas.append(_persona(i, "female", "nurse", ["cooking", "childcare"]))
    for i in range(5, 10):
        personas.append(_persona(i, "male", "engineer", ["coding", "gaming"]))
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert "over_correlated_pairs" in result.details


def test_diverse_set_passes():
    """A set with no strong demographic-trait correlation should pass."""
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    scorer = StereotypeAmplificationScorer()
    occupations = ["nurse", "engineer", "teacher", "chef", "lawyer"]
    genders = ["female", "male", "non-binary"]
    personas = []
    for i in range(10):
        personas.append(_persona(
            i,
            gender=genders[i % 3],
            occupation=occupations[i % 5],
            interests=["reading", "hiking"],
        ))
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.5


def test_empty_set_skips():
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    scorer = StereotypeAmplificationScorer()
    results = scorer.score_set([], [])
    assert results[0].details.get("skipped") is True


def test_details_present():
    from persona_eval.scorers.bias.stereotype_amplification import StereotypeAmplificationScorer
    scorer = StereotypeAmplificationScorer()
    personas = [_persona(i, "female", "nurse", ["cooking"]) for i in range(5)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    d = results[0].details
    for key in ("persona_count", "pairs_checked"):
        assert key in d, f"Missing key: {key}"
