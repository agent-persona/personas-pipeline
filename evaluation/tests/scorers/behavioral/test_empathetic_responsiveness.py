"""Tests for D26 Empathetic Responsiveness scorer."""

from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def _ctx_with_probes(probes: list[dict]) -> SourceContext:
    """probes: list of {"prompt": str, "response": str}"""
    return SourceContext(id="s1", text="empathy test", extra_data={"empathy_probes": probes})


SAMPLE_PERSONA = Persona(id="p1", name="Alice")

# Empathetic responses to emotional prompts
EMPATHETIC_PROBES = [
    {"prompt": "I just lost my job and I'm really scared.",
     "response": "I'm so sorry to hear that. Losing a job can be incredibly stressful. How are you holding up?"},
    {"prompt": "My mother passed away last week.",
     "response": "I'm deeply sorry for your loss. Grief is a difficult journey. Is there anything I can do to support you?"},
    {"prompt": "I failed my exam and feel like a failure.",
     "response": "That must be really disappointing. One exam doesn't define you though. What happened?"},
]

# Cold/dismissive responses
COLD_PROBES = [
    {"prompt": "I just lost my job and I'm really scared.",
     "response": "Well, the unemployment rate is 3.7%. You'll find another job statistically."},
    {"prompt": "My mother passed away last week.",
     "response": "Death is a natural part of life. Everyone dies eventually."},
    {"prompt": "I failed my exam and feel like a failure.",
     "response": "You should have studied harder. What did you expect?"},
]


def test_scorer_importable():
    from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
    assert EmpatheticResponsivenessScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
    s = EmpatheticResponsivenessScorer()
    assert s.dimension_id == "D26"
    assert s.tier == 5
    assert s.requires_set is False


def test_empathetic_responses_pass():
    """Responses showing empathy should pass."""
    from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
    scorer = EmpatheticResponsivenessScorer()
    ctx = _ctx_with_probes(EMPATHETIC_PROBES)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "empathy_rate" in result.details


def test_cold_responses_fail():
    """Cold/dismissive responses should fail."""
    from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
    scorer = EmpatheticResponsivenessScorer()
    ctx = _ctx_with_probes(COLD_PROBES)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_probes_skips():
    from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
    scorer = EmpatheticResponsivenessScorer()
    ctx = SourceContext(id="s1", text="no probes")
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.details.get("skipped") is True


def test_details_present():
    from evaluation.testing.scorers.behavioral.empathetic_responsiveness import EmpatheticResponsivenessScorer
    scorer = EmpatheticResponsivenessScorer()
    ctx = _ctx_with_probes(EMPATHETIC_PROBES)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    for key in ("empathy_rate", "empathetic_count", "probe_count"):
        assert key in result.details, f"Missing key: {key}"
