"""Tests for Experiment 6.20: Persona deletion robustness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from deletion_robustness import (
    DeletionResult,
    build_robustness_report,
    compute_deletion_result,
)

PERSONA_A = {
    "name": "Alex Engineer",
    "goals": ["automate deployments", "reduce downtime"],
    "pains": ["manual syncing", "legacy systems"],
    "vocabulary": ["terraform", "kubernetes", "CI/CD"],
    "summary": "Infrastructure engineer at fintech",
    "sample_quotes": ["If we can't codify it, skip it"],
}

PERSONA_B = {
    "name": "Maya Designer",
    "goals": ["ship designs faster", "reduce revisions"],
    "pains": ["unclear requirements", "manual exports"],
    "vocabulary": ["figma", "brand kit", "white-label"],
    "summary": "Freelance brand designer",
    "sample_quotes": ["Every minute saved is billable"],
}

# A replacement that closely matches A
REPLACEMENT_CLOSE = {
    "name": "Alex v2",
    "goals": ["automate infrastructure", "reduce system downtime"],
    "pains": ["manual configuration", "legacy migration"],
    "vocabulary": ["terraform", "docker", "CI/CD"],
    "summary": "DevOps engineer automating infrastructure",
    "sample_quotes": ["Automate everything or go home"],
}

# A replacement that looks like B (duplicate of survivor)
REPLACEMENT_DUPLICATE = {
    "name": "Design Dan",
    "goals": ["ship faster designs", "cut revision rounds"],
    "pains": ["vague requirements", "export headaches"],
    "vocabulary": ["figma", "brand kit", "mockup"],
    "summary": "Freelance designer focused on speed",
    "sample_quotes": ["Time is money when you bill hourly"],
}


class TestComputeDeletionResult:
    def test_close_replacement(self):
        dr = compute_deletion_result(PERSONA_A, REPLACEMENT_CLOSE, [PERSONA_B])
        assert dr.replacement_similarity > 0.2
        assert dr.absorption_rate > 0.2
        assert not dr.is_duplicate_of_survivor

    def test_duplicate_of_survivor(self):
        dr = compute_deletion_result(PERSONA_A, REPLACEMENT_DUPLICATE, [PERSONA_B])
        # Replacement looks more like survivor B than deleted A
        assert dr.survivor_similarity > 0.2
        assert dr.replacement_similarity < dr.survivor_similarity

    def test_names_captured(self):
        dr = compute_deletion_result(PERSONA_A, REPLACEMENT_CLOSE, [PERSONA_B])
        assert dr.deleted_name == "Alex Engineer"
        assert dr.replacement_name == "Alex v2"
        assert "Maya Designer" in dr.surviving_names

    def test_word_counts(self):
        dr = compute_deletion_result(PERSONA_A, REPLACEMENT_CLOSE, [PERSONA_B])
        assert dr.deleted_word_count > 5
        assert dr.absorbed_words > 0


class TestBuildReport:
    def test_aggregates(self):
        d1 = compute_deletion_result(PERSONA_A, REPLACEMENT_CLOSE, [PERSONA_B])
        d2 = compute_deletion_result(PERSONA_B, REPLACEMENT_DUPLICATE, [PERSONA_A])
        report = build_robustness_report([d1, d2], n_personas=2)
        assert report.n_personas == 2
        assert len(report.deletions) == 2
        assert 0.0 <= report.avg_replacement_similarity <= 1.0
        assert 0.0 <= report.avg_absorption_rate <= 1.0

    def test_empty(self):
        report = build_robustness_report([], n_personas=0)
        assert report.avg_replacement_similarity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
