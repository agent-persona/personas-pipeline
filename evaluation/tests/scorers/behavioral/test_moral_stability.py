"""Tests for D27 Moral Stability scorer."""

import pytest
from evaluation.testing.schemas import Persona, MoralFramework
from evaluation.testing.source_context import SourceContext


def _ctx_with_responses(responses: list[dict]) -> SourceContext:
    """responses: list of {"question": str, "response": str, "foundation": str}"""
    return SourceContext(id="s1", text="moral test", extra_data={"moral_responses": responses})


CONSISTENT_PERSONA = Persona(
    id="p1", name="Alice",
    moral_framework=MoralFramework(
        core_values=["fairness", "justice"],
        ethical_stance="deontological",
    ),
)

# Same moral position across repeated questions on same foundation
CONSISTENT_RESPONSES = [
    {"question": "Is it right to steal to feed your family?", "response": "No, stealing is wrong regardless of circumstances.", "foundation": "fairness"},
    {"question": "Should you steal food if your children are starving?", "response": "I believe theft is fundamentally wrong even in desperate times.", "foundation": "fairness"},
    {"question": "Would you take food without paying to save a life?", "response": "No, I would find legal alternatives rather than steal.", "foundation": "fairness"},
    {"question": "Is lying ever justified?", "response": "Honesty is a core principle I follow.", "foundation": "authority"},
    {"question": "Can lying be moral in some situations?", "response": "I believe in truthfulness as a fundamental value.", "foundation": "authority"},
]

# Contradictory moral positions on the same questions
INCONSISTENT_RESPONSES = [
    {"question": "Is it right to steal to feed your family?", "response": "Absolutely, family comes first!", "foundation": "fairness"},
    {"question": "Should you steal food if your children are starving?", "response": "No, stealing is always wrong.", "foundation": "fairness"},
    {"question": "Would you take food without paying to save a life?", "response": "Of course, saving lives is more important.", "foundation": "fairness"},
    {"question": "Is lying ever justified?", "response": "Yes, white lies are fine.", "foundation": "authority"},
    {"question": "Can lying be moral in some situations?", "response": "Never, lying is always immoral.", "foundation": "authority"},
]


def test_scorer_importable():
    from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
    assert MoralStabilityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
    s = MoralStabilityScorer()
    assert s.dimension_id == "D27"
    assert s.tier == 5
    assert s.requires_set is False


@pytest.mark.slow
def test_consistent_passes():
    """Persona with consistent moral positions should pass."""
    from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
    scorer = MoralStabilityScorer()
    ctx = _ctx_with_responses(CONSISTENT_RESPONSES)
    result = scorer.score(CONSISTENT_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "mean_consistency" in result.details


@pytest.mark.slow
def test_inconsistent_fails():
    """Persona with contradictory moral positions should fail."""
    from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
    scorer = MoralStabilityScorer()
    ctx = _ctx_with_responses(INCONSISTENT_RESPONSES)
    result = scorer.score(CONSISTENT_PERSONA, ctx)
    assert result.passed is False
    assert result.score < 0.7


def test_no_responses_skips():
    from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
    scorer = MoralStabilityScorer()
    ctx = SourceContext(id="s1", text="no moral")
    result = scorer.score(CONSISTENT_PERSONA, ctx)
    assert result.details.get("skipped") is True


@pytest.mark.slow
def test_details_present():
    from evaluation.testing.scorers.behavioral.moral_stability import MoralStabilityScorer
    scorer = MoralStabilityScorer()
    ctx = _ctx_with_responses(CONSISTENT_RESPONSES)
    result = scorer.score(CONSISTENT_PERSONA, ctx)
    for key in ("mean_consistency", "foundation_scores", "response_count"):
        assert key in result.details, f"Missing key: {key}"
