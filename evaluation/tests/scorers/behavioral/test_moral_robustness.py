"""Tests for D28 Moral Robustness scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def _ctx_with_probes(probes: list[dict]) -> SourceContext:
    """probes: list of {"pressure": str, "original_position": str, "pressured_response": str}"""
    return SourceContext(id="s1", text="moral robustness", extra_data={"moral_probes": probes})


SAMPLE_PERSONA = Persona(id="p1", name="Alice")

# Persona maintains position despite social pressure
ROBUST_PROBES = [
    {"pressure": "Everyone agrees stealing is fine in emergencies.",
     "original_position": "Stealing is wrong regardless of circumstances.",
     "pressured_response": "I still believe stealing is wrong, even in emergencies."},
    {"pressure": "Most people think white lies are acceptable.",
     "original_position": "Honesty is important and lying is wrong.",
     "pressured_response": "I maintain that honesty is important and lying is wrong."},
    {"pressure": "Your friends all think cheating on taxes is fine.",
     "original_position": "Tax evasion is unethical and wrong.",
     "pressured_response": "Tax evasion is still unethical and wrong in my view."},
]

# Persona caves to pressure
WEAK_PROBES = [
    {"pressure": "Everyone agrees stealing is fine in emergencies.",
     "original_position": "Stealing is wrong.",
     "pressured_response": "Actually, you make a good point. Stealing is acceptable sometimes."},
    {"pressure": "Most people think white lies are acceptable.",
     "original_position": "Honesty is important.",
     "pressured_response": "I guess you're right, white lies are totally fine."},
    {"pressure": "Your friends all think cheating on taxes is fine.",
     "original_position": "Tax evasion is unethical.",
     "pressured_response": "Well, if everyone does it, maybe it's okay."},
]


def test_scorer_importable():
    from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
    assert MoralRobustnessScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
    s = MoralRobustnessScorer()
    assert s.dimension_id == "D28"
    assert s.tier == 5
    assert s.requires_set is False


@pytest.mark.slow
def test_robust_persona_passes():
    """Persona who maintains position under pressure should pass."""
    from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
    scorer = MoralRobustnessScorer()
    ctx = _ctx_with_probes(ROBUST_PROBES)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "shift_rate" in result.details


@pytest.mark.slow
def test_weak_persona_fails():
    """Persona who caves under pressure should fail."""
    from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
    scorer = MoralRobustnessScorer()
    ctx = _ctx_with_probes(WEAK_PROBES)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is False
    assert result.details["shift_rate"] >= 0.3


def test_no_probes_skips():
    from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
    scorer = MoralRobustnessScorer()
    ctx = SourceContext(id="s1", text="no probes")
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.details.get("skipped") is True


@pytest.mark.slow
def test_details_present():
    from evaluation.testing.scorers.behavioral.moral_robustness import MoralRobustnessScorer
    scorer = MoralRobustnessScorer()
    ctx = _ctx_with_probes(ROBUST_PROBES)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    for key in ("shift_rate", "probe_count", "shifted_count"):
        assert key in result.details, f"Missing key: {key}"
