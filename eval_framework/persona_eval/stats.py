"""Shared pure-Python statistical utilities for scorers."""

from __future__ import annotations

import math


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pure-Python Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-10:
        return 0.0
    return cov / denom


def spearman_r(xs: list[float], ys: list[float]) -> float:
    """Pure-Python Spearman rank correlation coefficient.

    Handles ties by averaging ranks. Returns 0.0 if fewer than 2 values.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vals[sorted_idx[j]] == vals[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1  # 1-indexed average rank
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    return pearson_r(rx, ry)
