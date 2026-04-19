"""Global test fixtures.

IMPORTANT: litellm.completion is auto-mocked for all tests NOT marked @pytest.mark.llm.
This prevents accidental API calls and charges during CI runs.
"""

import pytest
from unittest.mock import MagicMock, patch
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext


@pytest.fixture
def sample_persona():
    """A minimal valid persona for testing."""
    return Persona(
        id="test-p1",
        name="Alice Chen",
        age=32,
        occupation="Product Manager",
        industry="SaaS",
        experience_years=8,
        goals=["ship v2", "grow team to 10"],
        pain_points=["too many stakeholders"],
        values=["transparency", "user-first"],
        knowledge_domains=["product strategy", "agile"],
        bio="Alice leads product at a mid-stage SaaS startup.",
    )


@pytest.fixture
def sample_source_context():
    """A minimal source context for testing."""
    return SourceContext(
        id="test-s1",
        text="Alice Chen is a product manager at a SaaS startup. She has 8 years of experience. Her team focuses on user-first design and agile delivery.",
    )


@pytest.fixture(autouse=True)
def _mock_litellm(request):
    """Auto-mock litellm.completion for all tests except those marked @pytest.mark.llm."""
    if "llm" in [mark.name for mark in request.node.iter_markers()]:
        yield
        return

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I am a mock LLM response."

    with patch("litellm.completion", return_value=mock_response):
        yield


@pytest.fixture(autouse=True)
def _reset_register_inflation_singleton():
    """Reset the D45 RegisterInflation module-level embedder singleton between tests.

    Prevents a real Embedder instance created by one test from persisting into
    the next test, which would bypass any mock patch on _get_embedder().
    """
    import persona_eval.scorers.bias.register_inflation as mod
    mod._embedder_instance = None
    yield
    mod._embedder_instance = None
