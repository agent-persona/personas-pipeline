"""Tests for Experiment 5.22: Eval contamination check.

Verifies:
1. Probe data structures
2. Public and novel probe banks
3. Contamination report computation
4. Hit rate and memorization thresholds
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from contamination import (
    ContaminationReport,
    NOVEL_PERSONA_PROBES,
    PUBLIC_PERSONA_PROBES,
    ProbeResult,
)


class TestProbeBank:
    def test_public_probes_have_required_fields(self):
        for p in PUBLIC_PERSONA_PROBES:
            assert "source" in p
            assert "fragment" in p
            assert "completion_seed" in p
            assert "expected_keywords" in p
            assert len(p["expected_keywords"]) >= 2

    def test_novel_probes_have_required_fields(self):
        for p in NOVEL_PERSONA_PROBES:
            assert "source" in p
            assert p["source"] == "novel"
            assert "completion_seed" in p

    def test_public_sources(self):
        sources = {p["source"] for p in PUBLIC_PERSONA_PROBES}
        assert "persona-hub" in sources
        assert "TinyTroupe" in sources

    def test_at_least_3_public(self):
        assert len(PUBLIC_PERSONA_PROBES) >= 3

    def test_at_least_2_novel(self):
        assert len(NOVEL_PERSONA_PROBES) >= 2


class TestProbeResult:
    def test_memorized_above_threshold(self):
        pr = ProbeResult(
            source="test", probe_type="completion", fragment="x",
            model_response="y", keywords_matched=3, keywords_total=4,
            hit_rate=0.75, is_memorized=True,
        )
        assert pr.is_memorized

    def test_clean_below_threshold(self):
        pr = ProbeResult(
            source="test", probe_type="completion", fragment="x",
            model_response="y", keywords_matched=1, keywords_total=4,
            hit_rate=0.25, is_memorized=False,
        )
        assert not pr.is_memorized


class TestContaminationReport:
    def test_contaminated_when_delta_high(self):
        r = ContaminationReport(
            model="test",
            public_hit_rate=0.6,
            novel_hit_rate=0.2,
            contamination_delta=0.4,
            is_contaminated=True,
        )
        assert r.is_contaminated

    def test_clean_when_delta_low(self):
        r = ContaminationReport(
            model="test",
            public_hit_rate=0.3,
            novel_hit_rate=0.25,
            contamination_delta=0.05,
            is_contaminated=False,
        )
        assert not r.is_contaminated

    def test_defaults(self):
        r = ContaminationReport(model="test")
        assert r.public_hit_rate == 0.0
        assert r.novel_hit_rate == 0.0
        assert not r.is_contaminated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
