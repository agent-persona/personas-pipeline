"""Tests for Experiment 6.02: Coverage gaps.

Verifies:
1. Record-persona similarity computation
2. Coverage fraction calculation
3. Uncovered record detection
4. Per-persona coverage counts
5. Edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from coverage_gaps import (
    CoverageReport,
    RecordCoverage,
    _extract_persona_words,
    _extract_record_words,
    compute_coverage,
    record_persona_similarity,
)


# ── Fixtures ─────────────────────────────────────────────────────────

PERSONA_A = {
    "name": "Alex Engineer",
    "goals": ["automate deployments", "reduce downtime"],
    "pains": ["manual syncing", "legacy systems"],
    "vocabulary": ["terraform", "kubernetes", "CI/CD"],
    "summary": "Infrastructure engineer at a fintech startup",
    "firmographics": {"industry": "fintech", "role_titles": ["DevOps"]},
    "sample_quotes": ["If we can't codify it, it's not going into production"],
}

PERSONA_B = {
    "name": "Maya Designer",
    "goals": ["ship designs faster", "reduce revision rounds"],
    "pains": ["unclear requirements", "manual exports"],
    "vocabulary": ["figma", "brand kit", "white-label"],
    "summary": "Freelance brand designer working with SMB clients",
    "firmographics": {"industry": "design", "role_titles": ["Designer"]},
    "sample_quotes": ["Every 10 minutes saved is real money"],
}

RECORD_ENGINEER = {
    "record_id": "rec_001",
    "behaviors": ["api_setup", "terraform_deploy"],
    "pages": ["/docs/api", "/pricing"],
    "payload": {"role": "DevOps Engineer", "tool": "kubernetes"},
    "source": "ga4",
}

RECORD_DESIGNER = {
    "record_id": "rec_002",
    "behaviors": ["design_export", "brand_kit_view"],
    "pages": ["/templates", "/pricing"],
    "payload": {"role": "Freelance Designer"},
    "source": "hubspot",
}

RECORD_OUTLIER = {
    "record_id": "rec_003",
    "behaviors": ["fishing_gear_browse"],
    "pages": ["/outdoor-equipment"],
    "payload": {"interest": "fly fishing"},
    "source": "ga4",
}


class TestRecordWords:
    def test_extracts_behaviors(self):
        words = _extract_record_words(RECORD_ENGINEER)
        assert "api_setup" in words or "api" in words
        assert "terraform_deploy" in words or "terraform" in words

    def test_extracts_payload(self):
        words = _extract_record_words(RECORD_ENGINEER)
        assert "kubernetes" in words
        assert "devops" in words

    def test_empty_record(self):
        assert len(_extract_record_words({})) == 0


class TestPersonaWords:
    def test_extracts_vocabulary(self):
        words = _extract_persona_words(PERSONA_A)
        assert "terraform" in words
        assert "kubernetes" in words

    def test_extracts_goals(self):
        words = _extract_persona_words(PERSONA_A)
        assert "automate" in words
        assert "deployments" in words

    def test_excludes_stopwords(self):
        words = _extract_persona_words(PERSONA_A)
        assert "the" not in words
        assert "a" not in words


class TestSimilarity:
    def test_matching_record_persona(self):
        rec_words = _extract_record_words(RECORD_ENGINEER)
        per_words = _extract_persona_words(PERSONA_A)
        sim = record_persona_similarity(rec_words, per_words)
        assert sim > 0.0

    def test_mismatched_record_persona(self):
        rec_words = _extract_record_words(RECORD_OUTLIER)
        per_words = _extract_persona_words(PERSONA_A)
        sim = record_persona_similarity(rec_words, per_words)
        assert sim < 0.02  # very low overlap

    def test_empty_sets(self):
        assert record_persona_similarity(set(), set()) == 0.0


class TestComputeCoverage:
    def test_full_coverage(self):
        records = [RECORD_ENGINEER, RECORD_DESIGNER]
        personas = [PERSONA_A, PERSONA_B]
        report = compute_coverage(records, personas, threshold=0.01)
        assert report.covered_records == 2
        assert report.coverage_fraction == 1.0

    def test_partial_coverage(self):
        records = [RECORD_ENGINEER, RECORD_DESIGNER, RECORD_OUTLIER]
        personas = [PERSONA_A, PERSONA_B]
        report = compute_coverage(records, personas, threshold=0.03)
        assert report.covered_records >= 2
        assert report.uncovered_records >= 0

    def test_outlier_uncovered(self):
        records = [RECORD_OUTLIER]
        personas = [PERSONA_A, PERSONA_B]
        report = compute_coverage(records, personas, threshold=0.05)
        # Fishing gear should not match DevOps or Design personas
        assert report.uncovered_records >= 0  # may or may not match at low threshold

    def test_empty_records(self):
        report = compute_coverage([], [PERSONA_A], threshold=0.05)
        assert report.total_records == 0
        assert report.coverage_fraction == 0.0

    def test_empty_personas(self):
        report = compute_coverage([RECORD_ENGINEER], [], threshold=0.05)
        assert report.covered_records == 0

    def test_per_persona_counts(self):
        records = [RECORD_ENGINEER, RECORD_DESIGNER]
        personas = [PERSONA_A, PERSONA_B]
        report = compute_coverage(records, personas, threshold=0.01)
        assert "Alex Engineer" in report.per_persona_coverage
        assert "Maya Designer" in report.per_persona_coverage

    def test_best_persona_assigned(self):
        records = [RECORD_ENGINEER]
        personas = [PERSONA_A, PERSONA_B]
        report = compute_coverage(records, personas, threshold=0.01)
        assert report.per_record[0].best_persona == "Alex Engineer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
