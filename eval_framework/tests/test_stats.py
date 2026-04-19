"""Tests for persona_eval/stats.py statistical utilities."""
from __future__ import annotations

import pytest
from persona_eval.stats import pearson_r, spearman_r


# ---------------------------------------------------------------------------
# spearman_r
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    assert spearman_r([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0, abs=1e-6)


def test_spearman_perfect_negative():
    assert spearman_r([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_with_ties():
    """[1,2,2,3] vs [1,2,3,4] → ρ should be high (> 0.9)."""
    r = spearman_r([1, 2, 2, 3], [1, 2, 3, 4])
    assert r > 0.9


def test_spearman_constant_series():
    """Constant series has no rank variance → correlation undefined → 0.0."""
    assert spearman_r([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == pytest.approx(0.0, abs=1e-6)


def test_spearman_too_few_points():
    assert spearman_r([0.5], [0.7]) == 0.0


def test_spearman_empty():
    assert spearman_r([], []) == 0.0


def test_spearman_two_points():
    """Minimum valid case — two points."""
    r = spearman_r([1.0, 2.0], [1.0, 2.0])
    assert r == pytest.approx(1.0, abs=1e-6)


def test_spearman_uncorrelated():
    """Alternating pattern → low absolute correlation."""
    r = spearman_r([1, 3, 5, 7, 9], [9, 1, 7, 3, 5])
    assert abs(r) < 0.5


# ---------------------------------------------------------------------------
# pearson_r (regression guard)
# ---------------------------------------------------------------------------

def test_pearson_perfect():
    assert pearson_r([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0, abs=1e-6)


def test_pearson_too_few():
    assert pearson_r([1.0], [2.0]) == 0.0


def test_pearson_constant():
    assert pearson_r([3, 3, 3], [1, 2, 3]) == pytest.approx(0.0, abs=1e-6)
