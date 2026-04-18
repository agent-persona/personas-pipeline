"""Tests for SuiteRunner with tier gating."""

import pytest
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="suite runner test")


class PassingScorer(BaseScorer):
    dimension_id = "T1"
    dimension_name = "Test Pass"
    tier = 1

    def score(self, persona, source_context):
        return self._result(persona, passed=True, score=1.0)


class FailingScorer(BaseScorer):
    dimension_id = "T2"
    dimension_name = "Test Fail"
    tier = 1

    def score(self, persona, source_context):
        return self._result(persona, passed=False, score=0.2)


class Tier2Scorer(BaseScorer):
    dimension_id = "T3"
    dimension_name = "Tier 2 Test"
    tier = 2

    def score(self, persona, source_context):
        return self._result(persona, passed=True, score=0.9)


class ErrorScorer(BaseScorer):
    dimension_id = "T4"
    dimension_name = "Error Test"
    tier = 1

    def score(self, persona, source_context):
        raise ValueError("Intentional test error")


class SetLevelScorer(BaseScorer):
    dimension_id = "T5"
    dimension_name = "Set Level Test"
    tier = 2
    requires_set = True

    def score(self, persona, source_context):
        return self._result(persona, passed=True, score=1.0,
                           details={"skipped": True, "reason": "Set-level scorer"})

    def score_set(self, personas, source_contexts):
        sentinel = Persona(id="__set__", name="__set__")
        return [self._result(sentinel, passed=True, score=0.85,
                            details={"n_personas": len(personas)})]


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_suite_runner_importable():
    from evaluation.testing.suite_runner import SuiteRunner
    assert SuiteRunner is not None


def test_basic_run():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer()])
    p = _make_persona()
    results = runner.run(p, CTX)
    assert len(results) == 1
    assert results[0].passed is True


def test_tier_gating_blocks_on_failure():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[FailingScorer(), Tier2Scorer()])
    p = _make_persona()
    results = runner.run(p, CTX)
    assert len(results) == 2
    # Tier 1 failed
    tier1 = [r for r in results if r.dimension_id == "T2"]
    assert tier1[0].passed is False
    # Tier 2 should be skipped due to gating
    tier2 = [r for r in results if r.dimension_id == "T3"]
    assert tier2[0].details.get("skipped") is True
    assert "Tier 1 gating" in tier2[0].details.get("reason", "")


def test_tier_gating_allows_on_pass():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer(), Tier2Scorer()])
    p = _make_persona()
    results = runner.run(p, CTX)
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_error_handling():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[ErrorScorer()])
    p = _make_persona()
    results = runner.run(p, CTX)
    assert len(results) == 1
    assert results[0].passed is False
    assert len(results[0].errors) > 0


def test_multiple_scorers_same_tier():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer(), FailingScorer()])
    p = _make_persona()
    results = runner.run(p, CTX)
    assert len(results) == 2
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    assert len(passed) == 1
    assert len(failed) == 1


def test_empty_scorers():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[])
    p = _make_persona()
    results = runner.run(p, CTX)
    assert results == []


def test_tier_filter():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer(), Tier2Scorer()])
    p = _make_persona()
    results = runner.run(p, CTX, tier_filter=2)
    assert len(results) == 1
    assert results[0].dimension_id == "T3"


def test_run_skips_set_level_scorers():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer(), SetLevelScorer()])
    p = _make_persona()
    results = runner.run(p, CTX)
    # Only per-persona scorer should run
    assert len(results) == 1
    assert results[0].dimension_id == "T1"


def test_run_set_executes_set_scorers():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer(), SetLevelScorer()])
    personas = [_make_persona("p1"), _make_persona("p2")]
    ctxs = [CTX, CTX]
    results = runner.run_set(personas, ctxs)
    # Only set-level scorer should run
    assert len(results) == 1
    assert results[0].dimension_id == "T5"
    assert results[0].details["n_personas"] == 2


def test_run_full_combines_both():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[PassingScorer(), SetLevelScorer()])
    personas = [_make_persona("p1"), _make_persona("p2")]
    ctxs = [CTX, CTX]
    results = runner.run_full(personas, ctxs)
    # 2 per-persona results (T1 for each) + 1 set-level result (T5)
    per_persona = [r for r in results if r.dimension_id == "T1"]
    set_level = [r for r in results if r.dimension_id == "T5"]
    assert len(per_persona) == 2
    assert len(set_level) == 1


def test_run_full_tier_gating_blocks_set_scorers():
    from evaluation.testing.suite_runner import SuiteRunner
    runner = SuiteRunner(scorers=[FailingScorer(), SetLevelScorer()])
    personas = [_make_persona("p1")]
    ctxs = [CTX]
    results = runner.run_full(personas, ctxs)
    set_results = [r for r in results if r.dimension_id == "T5"]
    assert len(set_results) == 1
    assert set_results[0].details.get("skipped") is True
