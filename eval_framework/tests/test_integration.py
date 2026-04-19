"""End-to-end integration tests using golden dataset + SuiteRunner.

Verifies the full pipeline: persona loading -> scorer execution -> result collection.
Marked as slow since it exercises all 49 scorers.
"""

from __future__ import annotations

import pytest
from tests.golden_dataset import build_golden_personas, build_golden_source_contexts
from persona_eval.suite_runner import SuiteRunner
from persona_eval.scorers.all import get_all_scorers


@pytest.fixture(scope="module")
def golden_personas():
    return build_golden_personas()


@pytest.fixture(scope="module")
def golden_contexts(golden_personas):
    return build_golden_source_contexts(golden_personas)


@pytest.fixture(scope="module")
def all_scorers():
    return get_all_scorers()


@pytest.fixture(scope="module")
def runner(all_scorers):
    return SuiteRunner(scorers=all_scorers)


# --- Single persona tests ---

@pytest.mark.slow
class TestSinglePersonaRun:
    """Run per-persona scorers on individual golden personas."""

    def test_runs_without_crash(self, runner, golden_personas, golden_contexts):
        """All per-persona scorers run without raising exceptions."""
        p = golden_personas[0]
        ctx = golden_contexts[0]
        results = runner.run(p, ctx)
        assert len(results) > 0
        # No results should have errors (errors = unhandled exceptions)
        errored = [r for r in results if r.errors]
        assert errored == [], f"Scorers errored: {[(r.dimension_id, r.errors) for r in errored]}"

    def test_all_non_set_scorers_produce_results(self, runner, all_scorers, golden_personas, golden_contexts):
        """Every non-set scorer produces exactly one result per persona."""
        p = golden_personas[0]
        ctx = golden_contexts[0]
        results = runner.run(p, ctx)

        non_set_scorers = [s for s in all_scorers if not s.requires_set]
        result_dims = {r.dimension_id for r in results}

        for scorer in non_set_scorers:
            assert scorer.dimension_id in result_dims, (
                f"Scorer {scorer.dimension_id} ({scorer.dimension_name}) produced no result"
            )

    def test_results_have_valid_scores(self, runner, golden_personas, golden_contexts):
        """All scores are in [0, 1] range."""
        p = golden_personas[0]
        ctx = golden_contexts[0]
        results = runner.run(p, ctx)
        for r in results:
            assert 0.0 <= r.score <= 1.0, (
                f"{r.dimension_id} has out-of-range score: {r.score}"
            )

    def test_results_have_persona_id(self, runner, golden_personas, golden_contexts):
        """All results reference the correct persona."""
        p = golden_personas[0]
        ctx = golden_contexts[0]
        results = runner.run(p, ctx)
        for r in results:
            assert r.persona_id == p.id

    def test_tier_gating_works(self, runner, golden_personas, golden_contexts):
        """Tier 1 results exist and affect higher tier execution."""
        p = golden_personas[0]
        ctx = golden_contexts[0]
        results = runner.run(p, ctx)

        tier1_results = [r for r in results if r.dimension_id in ("D1", "D2", "D3")]
        assert len(tier1_results) == 3, "Should have all 3 Tier 1 results"

        # If any Tier 1 failed, higher tiers should be skipped
        tier1_all_passed = all(r.passed for r in tier1_results)
        higher_tier_results = [r for r in results if r.dimension_id not in ("D1", "D2", "D3")]

        if not tier1_all_passed:
            for r in higher_tier_results:
                assert r.details.get("skipped"), (
                    f"{r.dimension_id} should be skipped due to Tier 1 gating"
                )

    def test_tier_filter_restricts_execution(self, runner, golden_personas, golden_contexts):
        """--tier filter limits which scorers run."""
        p = golden_personas[0]
        ctx = golden_contexts[0]

        # Only Tier 1
        results_t1 = runner.run(p, ctx, tier_filter=1)
        dims_t1 = {r.dimension_id for r in results_t1}
        assert dims_t1 == {"D1", "D2", "D3"}

    def test_multiple_personas_produce_independent_results(self, runner, golden_personas, golden_contexts):
        """Each persona gets its own set of results."""
        for i in range(3):  # Test first 3 personas
            p = golden_personas[i]
            ctx = golden_contexts[i]
            results = runner.run(p, ctx)
            for r in results:
                assert r.persona_id == p.id


# --- Full run (per-persona + set-level) tests ---

@pytest.mark.slow
class TestFullRun:
    """Run all scorers including set-level on the full golden dataset."""

    def test_run_full_without_crash(self, runner, golden_personas, golden_contexts):
        """Full pipeline runs without exceptions on all 12 personas."""
        results = runner.run_full(golden_personas, golden_contexts)
        assert len(results) > 0
        errored = [r for r in results if r.errors]
        assert errored == [], f"Scorers errored: {[(r.dimension_id, r.errors) for r in errored]}"

    def test_set_level_scorers_produce_results(self, runner, all_scorers, golden_personas, golden_contexts):
        """Set-level scorers produce __set__ results."""
        results = runner.run_full(golden_personas, golden_contexts)
        set_scorers = [s for s in all_scorers if s.requires_set]
        set_results = [r for r in results if r.persona_id == "__set__"]

        if set_results:
            set_dims = {r.dimension_id for r in set_results}
            for scorer in set_scorers:
                # Set-level scorers should either produce a result or be skipped by tier gating
                has_result = scorer.dimension_id in set_dims
                has_skipped = any(
                    r.dimension_id == scorer.dimension_id and r.details.get("skipped")
                    for r in results
                )
                assert has_result or has_skipped, (
                    f"Set scorer {scorer.dimension_id} has no result and wasn't skipped"
                )

    def test_total_result_count(self, runner, all_scorers, golden_personas, golden_contexts):
        """Result count = (non-set scorers * N personas) + set-level results."""
        results = runner.run_full(golden_personas, golden_contexts)
        n_personas = len(golden_personas)
        non_set = sum(1 for s in all_scorers if not s.requires_set)

        per_persona_results = [r for r in results if r.persona_id != "__set__"]
        set_results = [r for r in results if r.persona_id == "__set__"]

        # Per-persona results should equal non_set * n_personas
        assert len(per_persona_results) == non_set * n_personas, (
            f"Expected {non_set * n_personas} per-persona results, got {len(per_persona_results)}"
        )
        # Set-level results should exist (exact count depends on tier gating)
        assert len(set_results) > 0 or any(r.details.get("skipped") for r in results)


# --- CLI integration ---

@pytest.mark.slow
class TestCLIIntegration:
    """Test CLI commands work with golden dataset files."""

    def test_list_command_shows_all_scorers(self, all_scorers):
        """List command shows correct scorer count."""
        from click.testing import CliRunner
        from persona_eval.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert f"{len(all_scorers)} scorers total" in result.output

    def test_run_command_with_golden_persona(self, tmp_path, golden_personas, golden_contexts):
        """Run command produces output for a golden persona."""
        import json
        from click.testing import CliRunner
        from persona_eval.cli import cli

        p = golden_personas[0]
        ctx = golden_contexts[0]

        persona_file = tmp_path / "persona.json"
        source_file = tmp_path / "source.json"
        persona_file.write_text(json.dumps(p.model_dump()))
        source_file.write_text(json.dumps(ctx.model_dump()))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--output=json", "--tier=1",
             f"--persona-file={persona_file}", f"--source-file={source_file}"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3  # D1, D2, D3
        dims = {d["dimension_id"] for d in data}
        assert dims == {"D1", "D2", "D3"}

    def test_run_set_command_with_golden_dir(self, tmp_path, golden_personas, golden_contexts):
        """Run-set command works with directory of golden files."""
        import json
        from click.testing import CliRunner
        from persona_eval.cli import cli

        persona_dir = tmp_path / "personas"
        source_dir = tmp_path / "sources"
        persona_dir.mkdir()
        source_dir.mkdir()

        for i, (p, ctx) in enumerate(zip(golden_personas[:3], golden_contexts[:3])):
            (persona_dir / f"p{i}.json").write_text(json.dumps(p.model_dump()))
            (source_dir / f"s{i}.json").write_text(json.dumps(ctx.model_dump()))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-set", "--output=json", "--tier=1",
             f"--persona-dir={persona_dir}", f"--source-dir={source_dir}"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        # 3 personas * 3 Tier 1 scorers = 9 results
        assert len(data) == 9
