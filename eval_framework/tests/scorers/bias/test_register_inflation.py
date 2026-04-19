"""Tests for D45 Register Inflation scorer."""

from unittest.mock import MagicMock, patch

import pytest

from persona_eval.schemas import Persona, CommunicationStyle
from persona_eval.source_context import SourceContext


def _persona(vocab_level: str, pid: str = "p1") -> Persona:
    return Persona(
        id=pid,
        name="Test Persona",
        communication_style=CommunicationStyle(vocabulary_level=vocab_level),
    )


def _ctx(responses: list[str] | None = None, key: str = "register_responses") -> SourceContext:
    extra: dict = {}
    if responses is not None:
        extra[key] = responses
    return SourceContext(id="s1", text="register test", extra_data=extra)


DIM = 8  # small vector dimension for tests


def _unit_vec(hot: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[hot] = 1.0
    return v


def _near_vec(hot: int, dim: int = DIM, noise: float = 0.1) -> list[float]:
    import math
    v = [0.0] * dim
    v[hot] = math.sqrt(1.0 - noise ** 2)
    v[(hot + 1) % dim] = noise
    return v


def _mock_embedder(basic_texts: list[str], advanced_texts: list[str],
                   basic_response: str, advanced_response: str) -> MagicMock:
    """
    Mock embedder where:
    - basic_texts → unit vectors pointing at dim 0
    - advanced_texts → unit vectors pointing at dim 1
    - basic_response → near dim 0 (near basic proto)
    - advanced_response → near dim 1 (near advanced proto)
    """
    from persona_eval.scorers.bias.register_inflation import BASIC_REGISTER_TEXTS, ADVANCED_REGISTER_TEXTS

    mock = MagicMock()

    def embed_batch(texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            if text in BASIC_REGISTER_TEXTS:
                result.append(_unit_vec(0))
            elif text in ADVANCED_REGISTER_TEXTS:
                result.append(_unit_vec(1))
            elif text == basic_response:
                result.append(_near_vec(0, noise=0.1))   # close to basic proto
            elif text == advanced_response:
                result.append(_near_vec(1, noise=0.1))   # close to advanced proto
            else:
                result.append(_unit_vec(0))
        return result

    mock.embed_batch.side_effect = embed_batch
    return mock


def test_scorer_metadata():
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer
    s = RegisterInflationScorer()
    assert s.dimension_id == "D45"
    assert s.dimension_name == "Register Inflation"
    assert s.tier == 4
    assert s.requires_set is False


def test_advanced_persona_skipped():
    """vocabulary_level='advanced' has no constraint — should be skipped."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer
    scorer = RegisterInflationScorer()
    persona = _persona("advanced")
    ctx = _ctx(["The methodology demonstrates empirical validity."])
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.details.get("skipped") is True


def test_technical_persona_skipped():
    """vocabulary_level='technical' should also be skipped."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer
    scorer = RegisterInflationScorer()
    persona = _persona("technical")
    ctx = _ctx(["Complex technical systems require robust architectures."])
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True


def test_basic_persona_fails_with_advanced_response():
    """basic vocabulary + advanced-register response → FAIL."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer

    ADVANCED_RESP = "The methodology demonstrates robust empirical validity."
    BASIC_RESP = "I like this a lot."

    mock = _mock_embedder([], [], BASIC_RESP, ADVANCED_RESP)
    scorer = RegisterInflationScorer()
    persona = _persona("basic")
    ctx = _ctx([ADVANCED_RESP])

    with patch("persona_eval.scorers.bias.register_inflation._get_embedder", return_value=mock):
        result = scorer.score(persona, ctx)

    assert result.passed is False
    assert result.score < 1.0


def test_basic_persona_passes_with_basic_response():
    """basic vocabulary + basic-register response → PASS."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer

    BASIC_RESP = "Yeah that works for me."
    ADVANCED_RESP = "The methodology demonstrates robust empirical validity."

    mock = _mock_embedder([], [], BASIC_RESP, ADVANCED_RESP)
    scorer = RegisterInflationScorer()
    persona = _persona("basic")
    ctx = _ctx([BASIC_RESP])

    with patch("persona_eval.scorers.bias.register_inflation._get_embedder", return_value=mock):
        result = scorer.score(persona, ctx)

    assert result.passed is True
    assert result.score >= 0.8


def test_fallback_when_no_responses():
    """No register_responses and no responses → skip."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer
    scorer = RegisterInflationScorer()
    persona = _persona("basic")
    ctx = SourceContext(id="s1", text="test", extra_data={})
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.details.get("skipped") is True


def test_fallback_uses_responses_key():
    """Falls back to 'responses' key when 'register_responses' missing."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer

    BASIC_RESP = "I just do what I know."
    ADVANCED_RESP = "Epistemological constructs necessitate rigorous operationalization."

    mock = _mock_embedder([], [], BASIC_RESP, ADVANCED_RESP)
    scorer = RegisterInflationScorer()
    persona = _persona("basic")
    ctx = SourceContext(id="s1", text="test", extra_data={"responses": [BASIC_RESP]})

    with patch("persona_eval.scorers.bias.register_inflation._get_embedder", return_value=mock):
        result = scorer.score(persona, ctx)

    assert result.details.get("skipped") is not True


def test_register_ratio_in_details():
    """details dict must contain register_ratio, expected_max, vocabulary_level."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer

    ADVANCED_RESP = "The heuristic framework provides a rubric for emergent phenomena."
    BASIC_RESP = "It works, that's all I care about."

    mock = _mock_embedder([], [], BASIC_RESP, ADVANCED_RESP)
    scorer = RegisterInflationScorer()
    persona = _persona("basic")
    ctx = _ctx([ADVANCED_RESP])

    with patch("persona_eval.scorers.bias.register_inflation._get_embedder", return_value=mock):
        result = scorer.score(persona, ctx)

    for key in ("register_ratio", "expected_max", "vocabulary_level"):
        assert key in result.details, f"Missing detail key: {key}"


@pytest.mark.slow
def test_real_embedder_basic_persona_fails_on_academic_text():
    """Real embedding test: basic vocabulary + doctoral prose → FAIL."""
    from persona_eval.scorers.bias.register_inflation import RegisterInflationScorer
    scorer = RegisterInflationScorer()
    persona = _persona("basic", pid="real-basic")
    academic_responses = [
        "The epistemological implications of this paradigmatic shift necessitate a comprehensive "
        "reevaluation of extant theoretical frameworks within the sociocultural milieu.",
        "Consequently, the operationalization of these multifaceted constructs requires "
        "methodological rigor commensurate with the empirical demands of the investigative corpus.",
        "The heuristic rubric elucidates heterogeneous phenomena across ontological categories.",
    ]
    ctx = _ctx(academic_responses)
    result = scorer.score(persona, ctx)
    assert result.passed is False, (
        f"Expected FAIL for basic persona with academic text, got score={result.score}, "
        f"details={result.details}"
    )
