"""Tests for D4 Factual Grounding scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.factual_grounding import FactualGroundingScorer
    assert FactualGroundingScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.factual_grounding import FactualGroundingScorer
    s = FactualGroundingScorer()
    assert s.dimension_id == "D4"
    assert s.tier == 2
    assert s.requires_set is False


@pytest.mark.slow
def test_grounded_persona_scores_well(sample_persona):
    from evaluation.testing.scorers.semantic.factual_grounding import FactualGroundingScorer
    scorer = FactualGroundingScorer()
    # Rich source text that matches the sample_persona's claims
    rich_source = SourceContext(
        id="s-rich",
        text=(
            "Alice Chen is a 32-year-old product manager working at a mid-stage SaaS startup. "
            "She has 8 years of experience leading product teams. "
            "Her main goal is to ship v2 of the platform and grow her team to 10 people. "
            "She often struggles with too many stakeholders wanting different things."
        ),
    )
    result = scorer.score(sample_persona, rich_source)
    assert result.score >= 0.7
    assert result.details["grounded_claims"] >= 5
    assert result.details["total_claims"] > 0


@pytest.mark.slow
def test_ungrounded_persona_scores_poorly():
    from evaluation.testing.scorers.semantic.factual_grounding import FactualGroundingScorer
    scorer = FactualGroundingScorer()
    ungrounded = Persona(
        id="ungrounded-1", name="Basket Weaver",
        occupation="Underwater Basket Weaver", industry="Basket Arts",
        goals=["Win the Olympic basket weaving gold medal"],
    )
    source = SourceContext(
        id="s-unrelated", text="Interview with software engineers about code review practices and CI/CD pipelines.",
    )
    result = scorer.score(ungrounded, source)
    assert result.score < 0.5


def test_no_source_text_skips():
    from evaluation.testing.scorers.semantic.factual_grounding import FactualGroundingScorer
    scorer = FactualGroundingScorer()
    persona = Persona(id="test-1", name="Test")
    ctx = SourceContext(id="empty", text="")
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True
