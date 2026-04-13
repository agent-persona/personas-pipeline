"""Tests for Experiment 5.03: Self-preference bias.

Verifies:
1. Preference matrix construction from score results
2. Preference delta computation (diagonal vs off-diagonal)
3. Debiasing prior calculation
4. Edge cases (missing models, NaN scores, single judge)
5. Model key matching (opus/sonnet/haiku detection)
6. Per-dimension delta tracking
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from self_preference import (
    DIMENSIONS,
    JUDGE_RUBRIC_PROMPT,
    DebiasingPrior,
    PreferenceDelta,
    PreferenceMatrix,
    ScoreResult,
    build_preference_matrix,
    compute_debiasing_prior,
    compute_preference_deltas,
    _model_key,
)


# ── Helpers ──────────────────────────────────────────────────────────

SYNTH_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"]
JUDGE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"]


def _score(
    synth: str,
    judge: str,
    overall: float,
    dims: dict[str, float] | None = None,
    error: str | None = None,
) -> ScoreResult:
    if dims is None:
        dims = {d: overall for d in DIMENSIONS}
    return ScoreResult(
        synthesizer_model=synth,
        judge_model=judge,
        overall=overall,
        dimensions=dims,
        rationale="test",
        error=error,
    )


def _uniform_scores(value: float = 0.8) -> list[ScoreResult]:
    """All judges give the same score to all synthesizers."""
    scores = []
    for s in SYNTH_MODELS:
        for j in JUDGE_MODELS:
            scores.append(_score(s, j, value))
    return scores


def _biased_scores() -> list[ScoreResult]:
    """Judges inflate their own model's output by 0.10."""
    scores = []
    for s in SYNTH_MODELS:
        for j in JUDGE_MODELS:
            base = 0.75
            if _model_key(s) == _model_key(j):
                base = 0.85  # +0.10 diagonal bias
            scores.append(_score(s, j, base))
    return scores


def _asymmetric_bias_scores() -> list[ScoreResult]:
    """Only opus shows self-preference; others are neutral."""
    scores = []
    for s in SYNTH_MODELS:
        for j in JUDGE_MODELS:
            base = 0.80
            if _model_key(s) == "opus" and _model_key(j) == "opus":
                base = 0.92  # opus inflates own output
            scores.append(_score(s, j, base))
    return scores


# ── Model key tests ──────────────────────────────────────────────────

class TestModelKey:
    def test_opus(self):
        assert _model_key("claude-opus-4-6") == "opus"

    def test_sonnet(self):
        assert _model_key("claude-sonnet-4-6") == "sonnet"

    def test_haiku(self):
        assert _model_key("claude-haiku-4-5-20251001") == "haiku"

    def test_unknown(self):
        assert _model_key("gpt-4o") == "gpt-4o"


# ── Preference matrix tests ─────────────────────────────────────────

class TestPreferenceMatrix:
    def test_uniform_all_same(self):
        matrix = build_preference_matrix(_uniform_scores(0.8), SYNTH_MODELS, JUDGE_MODELS)
        for s in SYNTH_MODELS:
            for j in JUDGE_MODELS:
                assert abs(matrix.matrix[s][j] - 0.8) < 1e-6

    def test_biased_diagonal_higher(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        for s in SYNTH_MODELS:
            for j in JUDGE_MODELS:
                if _model_key(s) == _model_key(j):
                    assert matrix.matrix[s][j] == pytest.approx(0.85, abs=1e-6)
                else:
                    assert matrix.matrix[s][j] == pytest.approx(0.75, abs=1e-6)

    def test_model_lists_preserved(self):
        matrix = build_preference_matrix(_uniform_scores(), SYNTH_MODELS, JUDGE_MODELS)
        assert matrix.synthesizer_models == SYNTH_MODELS
        assert matrix.judge_models == JUDGE_MODELS

    def test_per_dim_populated(self):
        matrix = build_preference_matrix(_uniform_scores(0.7), SYNTH_MODELS, JUDGE_MODELS)
        for s in SYNTH_MODELS:
            for j in JUDGE_MODELS:
                for d in DIMENSIONS:
                    assert abs(matrix.per_dim[s][j][d] - 0.7) < 1e-6

    def test_error_scores_excluded(self):
        scores = _uniform_scores(0.8)
        # Add an error score — should be ignored
        scores.append(_score(SYNTH_MODELS[0], JUDGE_MODELS[0], 0.0, error="timeout"))
        matrix = build_preference_matrix(scores, SYNTH_MODELS, JUDGE_MODELS)
        # Original valid score should still be 0.8
        assert abs(matrix.matrix[SYNTH_MODELS[0]][JUDGE_MODELS[0]] - 0.8) < 1e-6


# ── Preference delta tests ──────────────────────────────────────────

class TestPreferenceDeltas:
    def test_uniform_zero_delta(self):
        matrix = build_preference_matrix(_uniform_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        for d in deltas:
            assert abs(d.delta) < 1e-6, f"{d.judge_model} should have zero delta"

    def test_biased_positive_delta(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        assert len(deltas) == 3
        for d in deltas:
            assert d.delta == pytest.approx(0.10, abs=1e-6), (
                f"{d.judge_model}: expected delta=+0.10, got {d.delta}"
            )

    def test_asymmetric_only_opus_biased(self):
        matrix = build_preference_matrix(_asymmetric_bias_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        delta_map = {_model_key(d.judge_model): d for d in deltas}

        assert delta_map["opus"].delta > 0.05
        assert abs(delta_map["sonnet"].delta) < 1e-6
        assert abs(delta_map["haiku"].delta) < 1e-6

    def test_diagonal_vs_off_diagonal(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        for d in deltas:
            assert d.diagonal_score == pytest.approx(0.85, abs=1e-6)
            assert d.off_diagonal_mean == pytest.approx(0.75, abs=1e-6)

    def test_per_dim_delta_populated(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        for d in deltas:
            for dim in DIMENSIONS:
                assert dim in d.per_dim_delta
                assert d.per_dim_delta[dim] == pytest.approx(0.10, abs=1e-6)


# ── Debiasing prior tests ───────────────────────────────────────────

class TestDebiasingPrior:
    def test_no_bias_not_detected(self):
        matrix = build_preference_matrix(_uniform_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        prior = compute_debiasing_prior(deltas)
        assert not prior.bias_detected
        assert abs(prior.overall_bias) < 1e-6

    def test_biased_detected(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        prior = compute_debiasing_prior(deltas)
        assert prior.bias_detected
        assert prior.overall_bias == pytest.approx(0.10, abs=1e-6)

    def test_judge_offsets_match_deltas(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        prior = compute_debiasing_prior(deltas)
        for d in deltas:
            assert prior.judge_offsets[d.judge_model] == pytest.approx(d.delta, abs=1e-6)

    def test_custom_threshold(self):
        matrix = build_preference_matrix(_biased_scores(), SYNTH_MODELS, JUDGE_MODELS)
        deltas = compute_preference_deltas(matrix)
        # Very high threshold — bias should NOT be detected
        prior = compute_debiasing_prior(deltas, bias_threshold=0.50)
        assert not prior.bias_detected

    def test_empty_deltas(self):
        prior = compute_debiasing_prior([])
        assert not prior.bias_detected
        assert prior.overall_bias == 0.0
        assert prior.judge_offsets == {}


# ── Multi-persona averaging tests ───────────────────────────────────

class TestMultiPersona:
    def test_two_personas_averaged(self):
        """Matrix should average across multiple personas per cell."""
        scores = []
        for s in SYNTH_MODELS:
            for j in JUDGE_MODELS:
                # Two "personas" with different scores
                scores.append(_score(s, j, 0.7))
                scores.append(_score(s, j, 0.9))
        matrix = build_preference_matrix(scores, SYNTH_MODELS, JUDGE_MODELS)
        for s in SYNTH_MODELS:
            for j in JUDGE_MODELS:
                assert matrix.matrix[s][j] == pytest.approx(0.8, abs=1e-6)


# ── Rubric prompt tests ─────────────────────────────────────────────

class TestRubricPrompt:
    def test_all_dimensions_present(self):
        for dim in DIMENSIONS:
            assert dim in JUDGE_RUBRIC_PROMPT

    def test_json_format(self):
        assert "JSON" in JUDGE_RUBRIC_PROMPT


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_model_no_deltas(self):
        """With only one model, no off-diagonal exists."""
        single = ["claude-opus-4-6"]
        scores = [_score(single[0], single[0], 0.9)]
        matrix = build_preference_matrix(scores, single, single)
        deltas = compute_preference_deltas(matrix)
        # Only one model: no off-diagonal to compare
        assert len(deltas) == 0 or all(
            math.isnan(d.delta) or d.delta == 0 for d in deltas
        )

    def test_nan_handling(self):
        scores = [
            _score(SYNTH_MODELS[0], JUDGE_MODELS[0], float("nan"), error="timeout"),
        ]
        matrix = build_preference_matrix(scores, SYNTH_MODELS, JUDGE_MODELS)
        # Should not crash
        deltas = compute_preference_deltas(matrix)
        prior = compute_debiasing_prior(deltas)
        assert isinstance(prior, DebiasingPrior)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
