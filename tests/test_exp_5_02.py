"""Tests for Experiment 5.02: Cross-judge agreement.

Verifies:
1. Agreement matrix computation is correct
2. Disagreement hotspot detection works
3. Judge rubric prompt is well-formed
4. Harness gracefully handles missing API keys
5. Pairwise MAD computation is accurate
6. Edge cases (single judge, all errors, NaN handling)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from judge_harness import (
    DIMENSIONS,
    JUDGE_RUBRIC_PROMPT,
    AgreementCell,
    AgreementMatrix,
    DisagreementHotspot,
    JudgeConfig,
    JudgeResult,
    MultiJudgeHarness,
    compute_agreement_matrix,
    find_disagreement_hotspots,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_result(
    model: str,
    backend: str = "anthropic",
    scores: dict[str, float] | None = None,
    error: str | None = None,
) -> JudgeResult:
    """Helper to create a JudgeResult with given scores."""
    if scores is None:
        scores = {d: 0.5 for d in DIMENSIONS}
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return JudgeResult(
        judge_model=model,
        judge_backend=backend,
        overall=overall,
        dimensions=scores,
        rationale="test rationale",
        error=error,
    )


# Two judges with identical scores
def _identical_results() -> list[list[JudgeResult]]:
    scores = {"grounded": 0.8, "distinctive": 0.7, "coherent": 0.9,
              "actionable": 0.6, "voice_fidelity": 0.75}
    return [[
        _make_result("model-a", scores=dict(scores)),
        _make_result("model-b", scores=dict(scores)),
    ]]


# Two judges with divergent scores
def _divergent_results() -> list[list[JudgeResult]]:
    return [[
        _make_result("model-a", scores={
            "grounded": 0.9, "distinctive": 0.3, "coherent": 0.85,
            "actionable": 0.7, "voice_fidelity": 0.4,
        }),
        _make_result("model-b", scores={
            "grounded": 0.85, "distinctive": 0.8, "coherent": 0.9,
            "actionable": 0.65, "voice_fidelity": 0.9,
        }),
    ]]


# Three judges, mixed agreement
def _three_judge_results() -> list[list[JudgeResult]]:
    return [[
        _make_result("model-a", scores={
            "grounded": 0.9, "distinctive": 0.5, "coherent": 0.85,
            "actionable": 0.7, "voice_fidelity": 0.4,
        }),
        _make_result("model-b", scores={
            "grounded": 0.88, "distinctive": 0.75, "coherent": 0.83,
            "actionable": 0.72, "voice_fidelity": 0.8,
        }),
        _make_result("model-c", scores={
            "grounded": 0.92, "distinctive": 0.6, "coherent": 0.87,
            "actionable": 0.68, "voice_fidelity": 0.65,
        }),
    ]]


# ── Agreement matrix tests ───────────────────────────────────────────

class TestAgreementMatrix:
    def test_identical_judges_zero_std(self):
        matrix = compute_agreement_matrix(_identical_results())
        for cell in matrix.cells:
            assert cell.std_dev == 0.0, f"{cell.dimension} should have zero std"
            assert cell.spread == 0.0, f"{cell.dimension} should have zero spread"

    def test_identical_judges_correct_mean(self):
        matrix = compute_agreement_matrix(_identical_results())
        expected = {"grounded": 0.8, "distinctive": 0.7, "coherent": 0.9,
                    "actionable": 0.6, "voice_fidelity": 0.75}
        for cell in matrix.cells:
            assert abs(cell.mean_score - expected[cell.dimension]) < 1e-6

    def test_divergent_judges_nonzero_std(self):
        matrix = compute_agreement_matrix(_divergent_results())
        # distinctive and voice_fidelity have large divergence
        dim_cells = {c.dimension: c for c in matrix.cells}
        assert dim_cells["distinctive"].std_dev > 0.2
        assert dim_cells["voice_fidelity"].std_dev > 0.2
        # grounded and coherent should be closer
        assert dim_cells["grounded"].std_dev < 0.1
        assert dim_cells["coherent"].std_dev < 0.1

    def test_spread_equals_max_minus_min(self):
        matrix = compute_agreement_matrix(_divergent_results())
        for cell in matrix.cells:
            assert abs(cell.spread - (cell.max_score - cell.min_score)) < 1e-6

    def test_n_judges_count(self):
        matrix = compute_agreement_matrix(_three_judge_results())
        assert matrix.n_judges == 3

    def test_n_personas_count(self):
        # Two personas
        results = _identical_results() + _divergent_results()
        matrix = compute_agreement_matrix(results)
        assert matrix.n_personas == 2

    def test_pairwise_mad_identical(self):
        matrix = compute_agreement_matrix(_identical_results())
        for pair, mad in matrix.pairwise_mad.items():
            assert mad == 0.0, f"MAD for {pair} should be 0 for identical scores"

    def test_pairwise_mad_divergent(self):
        matrix = compute_agreement_matrix(_divergent_results())
        mad = list(matrix.pairwise_mad.values())[0]
        assert mad > 0.0

    def test_pairwise_keys_three_judges(self):
        matrix = compute_agreement_matrix(_three_judge_results())
        assert len(matrix.pairwise_mad) == 3  # 3 choose 2

    def test_empty_input(self):
        matrix = compute_agreement_matrix([])
        assert matrix.n_personas == 0
        assert matrix.cells == []


# ── Error handling tests ─────────────────────────────────────────────

class TestErrorHandling:
    def test_error_results_filtered(self):
        results = [[
            _make_result("model-a", scores={d: 0.8 for d in DIMENSIONS}),
            _make_result("model-b", error="API timeout"),
        ]]
        matrix = compute_agreement_matrix(results)
        # Only 1 valid judge, so std_dev should be 0
        for cell in matrix.cells:
            assert cell.std_dev == 0.0

    def test_all_errors_produces_empty_matrix(self):
        results = [[
            _make_result("model-a", error="error1"),
            _make_result("model-b", error="error2"),
        ]]
        matrix = compute_agreement_matrix(results)
        for cell in matrix.cells:
            assert math.isnan(cell.mean_score)


# ── Disagreement hotspot tests ───────────────────────────────────────

class TestDisagreementHotspots:
    def test_identical_all_high_trust(self):
        matrix = compute_agreement_matrix(_identical_results())
        hotspots = find_disagreement_hotspots(matrix)
        for h in hotspots:
            assert h.trust_level == "high"

    def test_divergent_flags_hotspots(self):
        matrix = compute_agreement_matrix(_divergent_results())
        hotspots = find_disagreement_hotspots(matrix)
        low_trust = [h for h in hotspots if h.trust_level == "low"]
        dims = {h.dimension for h in low_trust}
        assert "distinctive" in dims, "distinctive should be low-trust"
        assert "voice_fidelity" in dims, "voice_fidelity should be low-trust"

    def test_sorted_by_std_desc(self):
        matrix = compute_agreement_matrix(_divergent_results())
        hotspots = find_disagreement_hotspots(matrix)
        for i in range(len(hotspots) - 1):
            assert hotspots[i].std_dev >= hotspots[i + 1].std_dev

    def test_custom_thresholds(self):
        matrix = compute_agreement_matrix(_divergent_results())
        # Very tight threshold: everything becomes low trust
        hotspots = find_disagreement_hotspots(matrix, std_threshold=0.01, spread_threshold=0.02)
        low = [h for h in hotspots if h.trust_level == "low"]
        assert len(low) >= 3  # most dims should be flagged

    def test_hotspot_has_scores_by_judge(self):
        matrix = compute_agreement_matrix(_divergent_results())
        hotspots = find_disagreement_hotspots(matrix)
        for h in hotspots:
            assert len(h.scores_by_judge) == 2  # two judges


# ── Rubric prompt tests ──────────────────────────────────────────────

class TestRubricPrompt:
    def test_all_dimensions_in_prompt(self):
        for dim in DIMENSIONS:
            assert dim in JUDGE_RUBRIC_PROMPT, f"{dim} missing from rubric"

    def test_json_format_requested(self):
        assert "JSON" in JUDGE_RUBRIC_PROMPT

    def test_scoring_scale_mentioned(self):
        assert "0.0" in JUDGE_RUBRIC_PROMPT
        assert "1.0" in JUDGE_RUBRIC_PROMPT

    def test_overall_requested(self):
        assert "overall" in JUDGE_RUBRIC_PROMPT


# ── Harness config tests ─────────────────────────────────────────────

class TestHarnessConfig:
    def test_from_env_with_anthropic_only(self):
        harness = MultiJudgeHarness.from_env(
            anthropic_key="test-key",
            openai_key="",
            google_key="",
        )
        assert len(harness.judges) == 3  # opus + sonnet + haiku
        assert all(j.backend == "anthropic" for j in harness.judges)

    def test_from_env_all_keys(self):
        harness = MultiJudgeHarness.from_env(
            anthropic_key="test-key",
            openai_key="test-key",
            google_key="test-key",
        )
        assert len(harness.judges) == 5  # opus + sonnet + haiku + gpt + gemini

    def test_from_env_no_keys_raises(self):
        with pytest.raises(ValueError, match="No judge backends"):
            MultiJudgeHarness.from_env(
                anthropic_key="",
                openai_key="",
                google_key="",
            )

    def test_judge_names(self):
        harness = MultiJudgeHarness.from_env(
            anthropic_key="k", openai_key="k", google_key="k",
        )
        names = {j.name for j in harness.judges}
        assert names == {"opus", "sonnet", "haiku", "gpt-4o", "gemini-pro"}


# ── Multi-persona agreement tests ────────────────────────────────────

class TestMultiPersona:
    def test_two_personas_averaged(self):
        """Agreement matrix averages across multiple personas."""
        # Persona 1: judges agree
        p1 = [
            _make_result("model-a", scores={d: 0.8 for d in DIMENSIONS}),
            _make_result("model-b", scores={d: 0.8 for d in DIMENSIONS}),
        ]
        # Persona 2: judges disagree on distinctive
        p2 = [
            _make_result("model-a", scores={
                "grounded": 0.8, "distinctive": 0.3, "coherent": 0.8,
                "actionable": 0.8, "voice_fidelity": 0.8,
            }),
            _make_result("model-b", scores={
                "grounded": 0.8, "distinctive": 0.9, "coherent": 0.8,
                "actionable": 0.8, "voice_fidelity": 0.8,
            }),
        ]
        matrix = compute_agreement_matrix([p1, p2])
        dim_cells = {c.dimension: c for c in matrix.cells}

        # distinctive should have some std (mean of [0.55, 0.85] = stdev > 0)
        assert dim_cells["distinctive"].std_dev > 0
        # grounded should have zero std (both judges gave 0.8 in both personas)
        assert dim_cells["grounded"].std_dev == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
