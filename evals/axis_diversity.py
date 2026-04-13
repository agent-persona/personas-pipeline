"""Experiment 6.10: Diversity along specific axes.

For each demographic/firmographic axis (age, income, geography, role),
computes spread and collapse-to-default rate across a persona set.

Metrics:
  - **Unique values**: how many distinct values exist on each axis.
  - **Collapse rate**: fraction of personas sharing the most common value
    (1.0 = all identical, 1/N = perfectly spread).
  - **Gini coefficient**: concentration measure (0 = perfectly equal
    distribution, 1 = maximally concentrated).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field


# ── Axis definitions ────────────────────────────────────────────────

AXES = {
    "age_range": lambda p: p.get("demographics", {}).get("age_range", "unknown"),
    "income_bracket": lambda p: p.get("demographics", {}).get("income_bracket") or "unknown",
    "education_level": lambda p: p.get("demographics", {}).get("education_level") or "unknown",
    "geography": lambda p: ", ".join(p.get("demographics", {}).get("location_signals", [])) or "unknown",
    "industry": lambda p: p.get("firmographics", {}).get("industry") or "unknown",
    "company_size": lambda p: p.get("firmographics", {}).get("company_size") or "unknown",
    "role": lambda p: ", ".join(p.get("firmographics", {}).get("role_titles", [])) or "unknown",
}


# ── Gini coefficient ────────────────────────────────────────────────

def gini_coefficient(values: list[str]) -> float:
    """Compute Gini coefficient for categorical distribution.

    0.0 = perfectly equal (every value appears once).
    1.0 = maximally concentrated (all same value).

    Uses the formula for categorical Gini (1 - sum of p_i^2), inverted
    so that 1.0 = concentrated and 0.0 = diverse.
    """
    if not values:
        return 0.0
    n = len(values)
    counts = Counter(values)
    sum_sq = sum((c / n) ** 2 for c in counts.values())
    # Herfindahl index = sum_sq; Gini-like = 1 - (1 - sum_sq) normalized
    # Simple: collapse_metric = max_share
    # More standard: 1 - (1/HHI_normalized)
    # Let's use: 1 - (number_of_unique / n) as a simple diversity score
    # Actually, let's use the proper concentration: max_freq / n
    # But the experiment asks for Gini, so let's compute it properly.

    # For categorical data, use Simpson's diversity index complement:
    # Concentration = sum(p_i^2). Ranges from 1/k (uniform) to 1.0 (all same).
    # Normalize to 0-1 where 0 = max diversity, 1 = no diversity:
    k = len(counts)
    if k <= 1:
        return 1.0  # only one category = fully collapsed
    min_concentration = 1.0 / k  # best case: uniform
    # concentration ranges from 1/k to 1.0
    # normalize: (actual - best) / (worst - best)
    return (sum_sq - min_concentration) / (1.0 - min_concentration)


# ── Data types ──────────────────────────────────────────────────────

@dataclass
class AxisReport:
    """Diversity analysis for one axis."""
    axis: str
    values: list[str]
    unique_count: int
    total: int
    most_common: str
    most_common_count: int
    collapse_rate: float  # most_common_count / total
    gini: float
    value_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class DiversityReport:
    """Full diversity analysis across all axes."""
    n_personas: int
    axes: list[AxisReport]
    collapsed_axes: list[str]  # axes with collapse_rate > 0.6
    diverse_axes: list[str]    # axes with collapse_rate < 0.4
    avg_gini: float
    avg_collapse_rate: float


# ── Analysis ────────────────────────────────────────────────────────

def analyze_axis(axis_name: str, values: list[str]) -> AxisReport:
    """Analyze diversity on a single axis."""
    counts = Counter(values)
    most_common_val, most_common_count = counts.most_common(1)[0] if counts else ("", 0)
    total = len(values)

    return AxisReport(
        axis=axis_name,
        values=values,
        unique_count=len(counts),
        total=total,
        most_common=most_common_val,
        most_common_count=most_common_count,
        collapse_rate=most_common_count / total if total > 0 else 0.0,
        gini=gini_coefficient(values),
        value_counts=dict(counts),
    )


def analyze_diversity(personas: list[dict]) -> DiversityReport:
    """Analyze diversity across all axes for a persona set."""
    axes: list[AxisReport] = []

    for axis_name, extractor in AXES.items():
        values = [extractor(p) for p in personas]
        report = analyze_axis(axis_name, values)
        axes.append(report)

    collapsed = [a.axis for a in axes if a.collapse_rate > 0.6]
    diverse = [a.axis for a in axes if a.collapse_rate < 0.4]

    ginis = [a.gini for a in axes]
    collapses = [a.collapse_rate for a in axes]

    return DiversityReport(
        n_personas=len(personas),
        axes=axes,
        collapsed_axes=collapsed,
        diverse_axes=diverse,
        avg_gini=statistics.mean(ginis) if ginis else 0.0,
        avg_collapse_rate=statistics.mean(collapses) if collapses else 0.0,
    )
