import json
import pytest
from unittest.mock import patch
from click.testing import CliRunner
from persona_eval.cli import cli
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext
from persona_eval.scorer import BaseScorer


class AlwaysPassScorer(BaseScorer):
    dimension_id = "D0"
    dimension_name = "Always Pass"
    tier = 1

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(persona, passed=True, score=1.0)


class AlwaysFailScorer(BaseScorer):
    dimension_id = "D0F"
    dimension_name = "Always Fail"
    tier = 1

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(persona, passed=False, score=0.2)


class Tier2Scorer(BaseScorer):
    dimension_id = "D0T2"
    dimension_name = "Tier 2 Test"
    tier = 2

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(persona, passed=True, score=0.9)


class SetScorer(BaseScorer):
    dimension_id = "D0S"
    dimension_name = "Set Level"
    tier = 2
    requires_set = True

    def score(self, persona, source_context):
        return self._result(persona, passed=True, score=1.0)

    def score_set(self, personas, source_contexts):
        sentinel = Persona(id="__set__", name="__set__")
        return [self._result(sentinel, passed=True, score=0.85,
                            details={"n_personas": len(personas)})]


MOCK_SCORERS = [AlwaysPassScorer()]


def _mock_get_all(scorers=None):
    """Patch get_all_scorers to return controlled scorers."""
    return patch("persona_eval.scorers.all.get_all_scorers",
                 return_value=scorers or MOCK_SCORERS)


def _mock_all_scorers(scorers=None):
    """Patch ALL_SCORERS list for the list command."""
    return patch("persona_eval.scorers.all.ALL_SCORERS",
                 scorers or MOCK_SCORERS)


# --- run command ---

def test_cli_run_table(tmp_path):
    persona_file = tmp_path / "persona.json"
    source_file = tmp_path / "source.json"
    persona_file.write_text(json.dumps({"id": "p1", "name": "Alice"}))
    source_file.write_text(json.dumps({"id": "s1", "text": "hello world"}))

    with _mock_get_all():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--output=table",
             f"--persona-file={persona_file}", f"--source-file={source_file}"],
        )
    assert result.exit_code == 0
    assert "Always Pass" in result.output
    assert "PASS" in result.output


def test_cli_run_json(tmp_path):
    persona_file = tmp_path / "persona.json"
    source_file = tmp_path / "source.json"
    persona_file.write_text(json.dumps({"id": "p1", "name": "Alice"}))
    source_file.write_text(json.dumps({"id": "s1", "text": "hello world"}))

    with _mock_get_all():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--output=json",
             f"--persona-file={persona_file}", f"--source-file={source_file}"],
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["passed"] is True
    assert data[0]["dimension_id"] == "D0"


def test_cli_run_with_tier_filter(tmp_path):
    persona_file = tmp_path / "persona.json"
    source_file = tmp_path / "source.json"
    persona_file.write_text(json.dumps({"id": "p1", "name": "Alice"}))
    source_file.write_text(json.dumps({"id": "s1", "text": "hello world"}))

    scorers = [AlwaysPassScorer(), Tier2Scorer()]
    with _mock_get_all(scorers):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--tier=2", "--output=json",
             f"--persona-file={persona_file}", f"--source-file={source_file}"],
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["dimension_id"] == "D0T2"


# --- run-set command ---

def test_cli_run_set_table(tmp_path):
    persona_dir = tmp_path / "personas"
    source_dir = tmp_path / "sources"
    persona_dir.mkdir()
    source_dir.mkdir()

    (persona_dir / "p1.json").write_text(json.dumps({"id": "p1", "name": "Alice"}))
    (persona_dir / "p2.json").write_text(json.dumps({"id": "p2", "name": "Bob"}))
    (source_dir / "s1.json").write_text(json.dumps({"id": "s1", "text": "hello"}))
    (source_dir / "s2.json").write_text(json.dumps({"id": "s2", "text": "world"}))

    with _mock_get_all():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-set", "--output=table",
             f"--persona-dir={persona_dir}", f"--source-dir={source_dir}"],
        )
    assert result.exit_code == 0
    assert "Always Pass" in result.output
    # Should have results for both personas
    assert "p1" in result.output
    assert "p2" in result.output


def test_cli_run_set_json(tmp_path):
    persona_dir = tmp_path / "personas"
    source_dir = tmp_path / "sources"
    persona_dir.mkdir()
    source_dir.mkdir()

    (persona_dir / "p1.json").write_text(json.dumps({"id": "p1", "name": "Alice"}))
    (source_dir / "s1.json").write_text(json.dumps({"id": "s1", "text": "hello"}))

    with _mock_get_all():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-set", "--output=json",
             f"--persona-dir={persona_dir}", f"--source-dir={source_dir}"],
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["persona_id"] == "p1"


def test_cli_run_set_no_persona_files(tmp_path):
    persona_dir = tmp_path / "empty"
    source_dir = tmp_path / "sources"
    persona_dir.mkdir()
    source_dir.mkdir()

    with _mock_get_all():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-set",
             f"--persona-dir={persona_dir}", f"--source-dir={source_dir}"],
        )
    assert result.exit_code == 1
    assert "no json files" in result.output.lower()


def test_cli_run_set_pads_source_contexts(tmp_path):
    """When fewer source files than persona files, should pad with defaults."""
    persona_dir = tmp_path / "personas"
    source_dir = tmp_path / "sources"
    persona_dir.mkdir()
    source_dir.mkdir()

    (persona_dir / "p1.json").write_text(json.dumps({"id": "p1", "name": "Alice"}))
    (persona_dir / "p2.json").write_text(json.dumps({"id": "p2", "name": "Bob"}))
    # Only one source file for two personas
    (source_dir / "s1.json").write_text(json.dumps({"id": "s1", "text": "hello"}))

    with _mock_get_all():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-set", "--output=json",
             f"--persona-dir={persona_dir}", f"--source-dir={source_dir}"],
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 2  # Both personas scored


def test_cli_run_set_with_set_scorer(tmp_path):
    """Set-level scorers should produce __set__ results in run-set."""
    persona_dir = tmp_path / "personas"
    source_dir = tmp_path / "sources"
    persona_dir.mkdir()
    source_dir.mkdir()

    (persona_dir / "p1.json").write_text(json.dumps({"id": "p1", "name": "Alice"}))
    (persona_dir / "p2.json").write_text(json.dumps({"id": "p2", "name": "Bob"}))
    (source_dir / "s1.json").write_text(json.dumps({"id": "s1", "text": "hello"}))
    (source_dir / "s2.json").write_text(json.dumps({"id": "s2", "text": "world"}))

    scorers = [AlwaysPassScorer(), SetScorer()]
    with _mock_get_all(scorers):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run-set", "--output=json",
             f"--persona-dir={persona_dir}", f"--source-dir={source_dir}"],
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    per_persona = [d for d in data if d["dimension_id"] == "D0"]
    set_level = [d for d in data if d["dimension_id"] == "D0S"]
    assert len(per_persona) == 2
    assert len(set_level) == 1
    assert set_level[0]["persona_id"] == "__set__"


# --- list command ---

def test_cli_list():
    with _mock_all_scorers():
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "Always Pass" in result.output
    assert "1 scorers total" in result.output


def test_cli_list_shows_set_flag():
    scorers = [AlwaysPassScorer(), SetScorer()]
    with _mock_all_scorers(scorers):
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "2 scorers total" in result.output
