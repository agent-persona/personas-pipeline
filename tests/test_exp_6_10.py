"""Tests for Experiment 6.10: Diversity along specific axes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from axis_diversity import (
    AXES,
    analyze_axis,
    analyze_diversity,
    gini_coefficient,
)


class TestGini:
    def test_all_same(self):
        assert gini_coefficient(["a", "a", "a"]) == 1.0

    def test_all_different(self):
        assert gini_coefficient(["a", "b", "c"]) == pytest.approx(0.0, abs=0.01)

    def test_partial(self):
        g = gini_coefficient(["a", "a", "b"])
        assert 0.0 < g < 1.0

    def test_empty(self):
        assert gini_coefficient([]) == 0.0

    def test_single(self):
        assert gini_coefficient(["a"]) == 1.0


class TestAnalyzeAxis:
    def test_collapsed(self):
        r = analyze_axis("age", ["25-34", "25-34", "25-34"])
        assert r.collapse_rate == 1.0
        assert r.unique_count == 1
        assert r.most_common == "25-34"

    def test_diverse(self):
        r = analyze_axis("age", ["18-24", "25-34", "35-44"])
        assert r.unique_count == 3
        assert r.collapse_rate == pytest.approx(1 / 3, abs=0.01)

    def test_value_counts(self):
        r = analyze_axis("role", ["engineer", "designer", "engineer"])
        assert r.value_counts["engineer"] == 2
        assert r.value_counts["designer"] == 1


class TestAnalyzeDiversity:
    def test_two_distinct_personas(self):
        personas = [
            {
                "demographics": {"age_range": "25-34", "location_signals": ["SF"], "income_bracket": "high"},
                "firmographics": {"industry": "fintech", "company_size": "50-200", "role_titles": ["Engineer"]},
            },
            {
                "demographics": {"age_range": "30-40", "location_signals": ["NYC"], "income_bracket": "medium"},
                "firmographics": {"industry": "design", "company_size": "1-10", "role_titles": ["Designer"]},
            },
        ]
        report = analyze_diversity(personas)
        assert report.n_personas == 2
        assert len(report.axes) == len(AXES)
        assert report.avg_collapse_rate <= 1.0

    def test_identical_personas_high_collapse(self):
        p = {
            "demographics": {"age_range": "25-34", "location_signals": ["SF"]},
            "firmographics": {"industry": "tech", "role_titles": ["Dev"]},
        }
        report = analyze_diversity([p, p, p])
        # All axes should be collapsed (all same values)
        assert report.avg_collapse_rate == 1.0
        assert len(report.collapsed_axes) == len(AXES)

    def test_axes_defined(self):
        assert "age_range" in AXES
        assert "industry" in AXES
        assert "role" in AXES
        assert "geography" in AXES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
