"""Tests for experiment 5.05: Rubric ablation harness."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from rubric_ablation import (
    FULL_DIMENSIONS,
    DIMENSION_DESCRIPTIONS,
    AblationResult,
    AblationScore,
    build_rubric_system_prompt,
    build_judge_prompt,
    analyze_ablation,
    format_analysis,
    _pearson_r,
    _kendall_tau,
    _parse_response,
)


# ── Rubric builder tests ─────────────────────────────────────────────

class TestBuildRubricSystemPrompt:
    def test_full_rubric_includes_all_dimensions(self):
        prompt = build_rubric_system_prompt(FULL_DIMENSIONS)
        for dim in FULL_DIMENSIONS:
            assert dim in prompt

    def test_ablated_rubric_excludes_dropped_dimension(self):
        for drop in FULL_DIMENSIONS:
            remaining = tuple(d for d in FULL_DIMENSIONS if d != drop)
            prompt = build_rubric_system_prompt(remaining)
            # The dropped dimension should not appear in the JSON format spec
            assert f'"{drop}": <1-5>' not in prompt
            for d in remaining:
                assert f'"{d}": <1-5>' in prompt

    def test_all_dimension_descriptions_exist(self):
        for dim in FULL_DIMENSIONS:
            assert dim in DIMENSION_DESCRIPTIONS


class TestBuildJudgePrompt:
    def test_includes_persona_json(self):
        persona = {"name": "Test Persona", "goals": ["goal1"]}
        prompt = build_judge_prompt(persona)
        assert "Test Persona" in prompt
        assert "goal1" in prompt


# ── Parse response tests ─────────────────────────────────────────────

class TestParseResponse:
    def test_valid_json(self):
        text = '{"grounded": 4, "distinctive": 3, "overall": 3.5, "rationale": "ok"}'
        result = _parse_response(text, ("grounded", "distinctive"))
        assert result["grounded"] == 4.0
        assert result["distinctive"] == 3.0

    def test_markdown_fences(self):
        text = '```json\n{"grounded": 5}\n```'
        result = _parse_response(text, ("grounded",))
        assert result["grounded"] == 5.0

    def test_invalid_json_returns_nan(self):
        result = _parse_response("not json at all", ("grounded",))
        assert math.isnan(result["grounded"])


# ── Statistics tests ──────────────────────────────────────────────────

class TestPearsonR:
    def test_perfect_correlation(self):
        r = _pearson_r([1, 2, 3, 4], [2, 4, 6, 8])
        assert abs(r - 1.0) < 0.001

    def test_negative_correlation(self):
        r = _pearson_r([1, 2, 3, 4], [8, 6, 4, 2])
        assert abs(r - (-1.0)) < 0.001

    def test_too_few_returns_nan(self):
        assert math.isnan(_pearson_r([1, 2], [3, 4]))


class TestKendallTau:
    def test_perfect_concordance(self):
        tau = _kendall_tau([1, 2, 3, 4], [10, 20, 30, 40])
        assert abs(tau - 1.0) < 0.001

    def test_perfect_discordance(self):
        tau = _kendall_tau([1, 2, 3, 4], [40, 30, 20, 10])
        assert abs(tau - (-1.0)) < 0.001

    def test_too_few_returns_nan(self):
        assert math.isnan(_kendall_tau([1, 2], [3, 4]))


# ── Analysis tests ────────────────────────────────────────────────────

def _make_result(pid: str, ctrl_scores: dict, ablated: dict) -> AblationResult:
    """Helper to build an AblationResult."""
    valid_scores = [v for v in ctrl_scores.values() if not math.isnan(v)]
    ctrl_overall = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")

    result = AblationResult(
        persona_id=pid,
        persona_dict={"name": pid},
        control_score=AblationScore(
            variant="full",
            dimensions_used=FULL_DIMENSIONS,
            persona_id=pid,
            scores=ctrl_scores,
            overall=ctrl_overall,
        ),
    )
    for drop_dim, scores in ablated.items():
        remaining = tuple(d for d in FULL_DIMENSIONS if d != drop_dim)
        valid = [v for v in scores.values() if not math.isnan(v)]
        overall = sum(valid) / len(valid) if valid else float("nan")
        result.ablated_scores[drop_dim] = AblationScore(
            variant=f"drop_{drop_dim}",
            dimensions_used=remaining,
            persona_id=pid,
            scores=scores,
            overall=overall,
        )
    return result


class TestAnalyzeAblation:
    def test_empty_results(self):
        analysis = analyze_ablation([])
        assert analysis.n_personas == 0

    def test_detects_inert_dimension(self):
        """If removing a dim doesn't change rankings, it should be flagged inert."""
        # 4 personas with scores that maintain rank order when any dim is dropped
        results = []
        for i, pid in enumerate(["p1", "p2", "p3", "p4"]):
            base = i + 1  # 1, 2, 3, 4
            ctrl = {d: float(base) for d in FULL_DIMENSIONS}
            ablated = {}
            for drop_dim in FULL_DIMENSIONS:
                remaining = {d: float(base) for d in FULL_DIMENSIONS if d != drop_dim}
                ablated[drop_dim] = remaining
            results.append(_make_result(pid, ctrl, ablated))

        analysis = analyze_ablation(results)
        # All dimensions should be inert since rank order is preserved
        assert len(analysis.inert_dimensions) > 0

    def test_format_analysis_runs(self):
        analysis = analyze_ablation([])
        text = format_analysis(analysis)
        assert "PAIRWISE" in text
        assert "FINDINGS" in text
