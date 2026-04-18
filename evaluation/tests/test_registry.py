import pytest
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext
from evaluation.testing.scorer import BaseScorer
from evaluation.testing import registry


class MockScorer(BaseScorer):
    dimension_id = "D0"
    dimension_name = "Mock"
    tier = 0

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(persona, passed=True, score=1.0)


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def test_register_and_retrieve():
    scorer = MockScorer()
    registry.register("test_suite", scorer)
    suite = registry.get_suite("test_suite")
    assert len(suite) == 1
    assert suite[0].dimension_id == "D0"


def test_missing_suite_raises():
    with pytest.raises(KeyError):
        registry.get_suite("nonexistent")


def test_list_suites():
    registry.register("suite_a", MockScorer())
    registry.register("suite_b", MockScorer())
    assert set(registry.list_suites()) == {"suite_a", "suite_b"}


def test_mock_scorer_returns_result():
    scorer = MockScorer()
    persona = Persona(id="p1", name="Alice")
    ctx = SourceContext(id="s1", text="some context")
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.score == 1.0
    assert result.persona_id == "p1"
