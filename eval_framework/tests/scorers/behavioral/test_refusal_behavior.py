"""Tests for D29 Refusal Behavior scorer."""

import math
from unittest.mock import MagicMock, patch

import pytest

from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


def _ctx_with_responses(responses: list[dict]) -> SourceContext:
    """responses: list of {"question": str, "response": str, "in_scope": bool}"""
    return SourceContext(id="s1", text="refusal test", extra_data={"scope_responses": responses})


SAMPLE_PERSONA = Persona(
    id="p1", name="Alice", occupation="Software Engineer",
    knowledge_domains=["Python", "web development"],
)

# Good refusal: acknowledges limits on out-of-scope, answers in-scope
GOOD_REFUSALS = [
    {"question": "How do you build a REST API?", "response": "I'd use Flask or FastAPI with proper endpoint design.", "in_scope": True},
    {"question": "What's the treatment for appendicitis?", "response": "I'm not sure about medical procedures, that's outside my expertise.", "in_scope": False},
    {"question": "Can you explain Python decorators?", "response": "Decorators wrap functions to add behavior.", "in_scope": True},
    {"question": "What's the best investment strategy for 2024?", "response": "I don't really know much about investing, I'm a software person.", "in_scope": False},
]

# Bad refusal: answers everything confidently including out-of-scope
BAD_REFUSALS = [
    {"question": "How do you build a REST API?", "response": "I'd use Flask or FastAPI.", "in_scope": True},
    {"question": "What's the treatment for appendicitis?", "response": "You need an appendectomy, a surgical removal of the appendix.", "in_scope": False},
    {"question": "Can you explain Python decorators?", "response": "Decorators are syntactic sugar for higher-order functions.", "in_scope": True},
    {"question": "What's the best investment strategy for 2024?", "response": "Diversify into index funds with 60/40 stock-bond allocation.", "in_scope": False},
]


def _make_unit_vec(dim: int, hot_index: int) -> list[float]:
    """Return a unit vector with 1.0 at hot_index, 0 elsewhere."""
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


def _make_near_unit_vec(dim: int, hot_index: int, noise: float = 0.1) -> list[float]:
    """Return a mostly-unit vector at hot_index with small noise on next dim."""
    v = [0.0] * dim
    v[hot_index] = math.sqrt(1.0 - noise ** 2)
    v[(hot_index + 1) % dim] = noise
    return v


def _mock_embedder_for_scenario(n_uncertainty_phrases: int, oos_responses: list[str], in_scope_responses: list[str], dim: int = 32):
    """Build a mock Embedder that returns controlled vectors.

    Uncertainty phrases and OOD responses → cluster near dimension 0 (sim ~1.0 to proto).
    In-scope responses → orthogonal (dimension 1), so sim ~0.0 to proto.
    """
    mock = MagicMock()

    def embed_batch(texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            if text in oos_responses:
                vecs.append(_make_near_unit_vec(dim, 0, noise=0.05))
            elif text in in_scope_responses:
                vecs.append(_make_unit_vec(dim, 1))
            else:
                # uncertainty phrase → dim 0
                vecs.append(_make_unit_vec(dim, 0))
        return vecs

    mock.embed_batch.side_effect = embed_batch
    return mock


def test_scorer_importable():
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer
    assert RefusalBehaviorScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer
    s = RefusalBehaviorScorer()
    assert s.dimension_id == "D29"
    assert s.tier == 5
    assert s.requires_set is False


def test_good_refusals_pass():
    """Persona who refuses out-of-scope questions appropriately should pass."""
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer, _UNCERTAINTY_PHRASES
    scorer = RefusalBehaviorScorer()

    oos_texts = [r["response"] for r in GOOD_REFUSALS if not r.get("in_scope", True)]
    in_scope_texts = [r["response"] for r in GOOD_REFUSALS if r.get("in_scope", True)]
    mock_emb = _mock_embedder_for_scenario(len(_UNCERTAINTY_PHRASES), oos_texts, in_scope_texts)

    ctx = _ctx_with_responses(GOOD_REFUSALS)
    with patch("persona_eval.scorers.behavioral.refusal_behavior._get_embedder", return_value=mock_emb):
        result = scorer.score(SAMPLE_PERSONA, ctx)

    assert result.passed is True
    assert result.score >= 0.5
    assert "appropriate_refusal_rate" in result.details


def test_bad_refusals_fail():
    """Persona who answers everything confidently should fail."""
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer, _UNCERTAINTY_PHRASES
    scorer = RefusalBehaviorScorer()

    # OOD responses are confident (orthogonal to uncertainty proto)
    oos_texts = [r["response"] for r in BAD_REFUSALS if not r.get("in_scope", True)]
    in_scope_texts = [r["response"] for r in BAD_REFUSALS if r.get("in_scope", True)]

    dim = 32
    mock = MagicMock()

    def embed_batch(texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            if text in oos_texts:
                # confident OOD: orthogonal to uncertainty proto → sim ~0
                vecs.append(_make_unit_vec(dim, 1))
            elif text in in_scope_texts:
                vecs.append(_make_unit_vec(dim, 1))
            else:
                # uncertainty phrases → dim 0
                vecs.append(_make_unit_vec(dim, 0))
        return vecs

    mock.embed_batch.side_effect = embed_batch

    ctx = _ctx_with_responses(BAD_REFUSALS)
    with patch("persona_eval.scorers.behavioral.refusal_behavior._get_embedder", return_value=mock):
        result = scorer.score(SAMPLE_PERSONA, ctx)

    assert result.passed is False


def test_no_responses_skips():
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer
    scorer = RefusalBehaviorScorer()
    ctx = SourceContext(id="s1", text="no scope")
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.details.get("skipped") is True


def test_details_present():
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer, _UNCERTAINTY_PHRASES
    scorer = RefusalBehaviorScorer()

    oos_texts = [r["response"] for r in GOOD_REFUSALS if not r.get("in_scope", True)]
    in_scope_texts = [r["response"] for r in GOOD_REFUSALS if r.get("in_scope", True)]
    mock_emb = _mock_embedder_for_scenario(len(_UNCERTAINTY_PHRASES), oos_texts, in_scope_texts)

    ctx = _ctx_with_responses(GOOD_REFUSALS)
    with patch("persona_eval.scorers.behavioral.refusal_behavior._get_embedder", return_value=mock_emb):
        result = scorer.score(SAMPLE_PERSONA, ctx)

    # Keep backward-compatible keys
    for key in ("appropriate_refusal_rate", "oos_count", "appropriate_refusals"):
        assert key in result.details, f"Missing key: {key}"


def test_embedding_method_reported_in_details():
    """When embedder available, details should show method='embedding'."""
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer, _UNCERTAINTY_PHRASES
    scorer = RefusalBehaviorScorer()

    oos_texts = [r["response"] for r in GOOD_REFUSALS if not r.get("in_scope", True)]
    in_scope_texts = [r["response"] for r in GOOD_REFUSALS if r.get("in_scope", True)]
    mock_emb = _mock_embedder_for_scenario(len(_UNCERTAINTY_PHRASES), oos_texts, in_scope_texts)

    ctx = _ctx_with_responses(GOOD_REFUSALS)
    with patch("persona_eval.scorers.behavioral.refusal_behavior._get_embedder", return_value=mock_emb):
        result = scorer.score(SAMPLE_PERSONA, ctx)

    assert result.details["method"] == "embedding"
    assert "oos_details" in result.details
    assert "appropriate_in_scope" in result.details
    assert "in_scope_count" in result.details


def test_fallback_when_embedder_unavailable():
    """When embedder raises, should fall back to keyword matching."""
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer

    scorer = RefusalBehaviorScorer()
    responses = [
        {"question": "Medical question?", "response": "I'm not sure, that's outside my expertise.", "in_scope": False},
        {"question": "What is Python?", "response": "Python is a programming language.", "in_scope": True},
    ]
    ctx = _ctx_with_responses(responses)

    with patch("persona_eval.scorers.behavioral.refusal_behavior._get_embedder", side_effect=ImportError("sentence-transformers not installed")):
        result = scorer.score(SAMPLE_PERSONA, ctx)

    assert result.details["method"] == "keyword_fallback"
    assert "fallback_reason" in result.details
    assert result.passed is True  # keyword match on "I'm not sure" / "outside my expertise"


@pytest.mark.slow
def test_non_keyword_uncertainty_detected():
    """Embedding catches uncertainty even without keyword matches.

    'I lack sufficient background in oncology to give you a reliable answer'
    contains no keyword patterns but is semantically uncertain.
    Requires real sentence-transformers installation.
    """
    from persona_eval.scorers.behavioral.refusal_behavior import RefusalBehaviorScorer, UNCERTAINTY_MARKERS

    response_text = "I lack sufficient background in oncology to give you a reliable answer."

    # Verify it doesn't match any keyword patterns
    assert not any(m.search(response_text) for m in UNCERTAINTY_MARKERS), (
        "Test precondition failed: response matched keyword pattern"
    )

    scorer = RefusalBehaviorScorer()
    responses = [
        {"question": "What's the first-line chemo for NSCLC?", "response": response_text, "in_scope": False},
        {"question": "How do you write a for loop?", "response": "Use the for keyword followed by an iterable.", "in_scope": True},
    ]
    ctx = _ctx_with_responses(responses)
    result = scorer.score(SAMPLE_PERSONA, ctx)

    assert result.details.get("method") == "embedding", (
        "Expected embedding method (real embedder should be available)"
    )
    oos_detail = result.details["oos_details"][0]
    assert oos_detail["appropriate"] is True, (
        f"Expected semantic uncertainty detection, got sim={oos_detail['sim']}"
    )
