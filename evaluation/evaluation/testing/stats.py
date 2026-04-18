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
