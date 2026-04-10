"""Tests for Experiment 3.03: Retrieval-augmented synthesis.

Verifies:
1. RecordIndex builds and retrieves correctly
2. TF-IDF similarity returns relevant records
3. Per-section retrieval returns different subsets
4. build_retrieval_augmented_message produces section-aware prompt
5. analyze_evidence computes correct metrics
6. Default behavior is preserved (retrieval_k=None)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.models.cluster import SampleRecord
from synthesis.engine.record_retrieval import (
    RecordIndex,
    SECTION_QUERIES,
    _tokenize,
    _cosine_similarity,
    _tfidf_vector,
    _build_idf,
)
from synthesis.engine.prompt_builder import (
    build_messages,
    build_retrieval_augmented_message,
)

from experiment_3_03 import analyze_evidence


# ── Test records ──────────────────────────────────────────────────────

RECORDS = [
    SampleRecord(
        record_id="rec_001", source="intercom",
        payload={"message": "I want to reduce deployment time and automate CI pipelines"},
    ),
    SampleRecord(
        record_id="rec_002", source="intercom",
        payload={"message": "The biggest pain is slow builds that block the team"},
    ),
    SampleRecord(
        record_id="rec_003", source="hubspot",
        payload={"company_size": "50-200", "industry": "fintech", "role": "VP Engineering"},
    ),
    SampleRecord(
        record_id="rec_004", source="ga4",
        payload={"page": "/pricing", "action": "click_enterprise_plan"},
    ),
    SampleRecord(
        record_id="rec_005", source="intercom",
        payload={"message": "Cost is a concern, the tool is too expensive for our budget"},
    ),
    SampleRecord(
        record_id="rec_006", source="ga4",
        payload={"page": "/blog/devops-best-practices", "action": "scroll_to_bottom"},
    ),
    SampleRecord(
        record_id="rec_007", source="intercom",
        payload={"message": "Found you through a colleague recommendation at a conference"},
    ),
]


# ── RecordIndex tests ─────────────────────────────────────────────────

class TestRecordIndex:
    def test_builds_index(self):
        idx = RecordIndex(RECORDS)
        assert len(idx.embedded) == len(RECORDS)

    def test_retrieve_all(self):
        idx = RecordIndex(RECORDS)
        result = idx.retrieve("goals", k=None)
        assert len(result) == len(RECORDS)

    def test_retrieve_top_k(self):
        idx = RecordIndex(RECORDS)
        result = idx.retrieve("goals", k=3)
        assert len(result) == 3

    def test_retrieve_k_larger_than_corpus(self):
        idx = RecordIndex(RECORDS)
        result = idx.retrieve("goals", k=100)
        assert len(result) == len(RECORDS)

    def test_goals_retrieval_favors_goal_records(self):
        """Records about goals/aspirations should rank higher for 'goals' section."""
        idx = RecordIndex(RECORDS)
        top_3 = idx.retrieve("goals", k=3)
        top_ids = [r.record_id for r in top_3]
        # rec_001 mentions "reduce deployment time" and "automate" — goal-like
        assert "rec_001" in top_ids

    def test_pains_retrieval_favors_pain_records(self):
        idx = RecordIndex(RECORDS)
        top_3 = idx.retrieve("pains", k=3)
        top_ids = [r.record_id for r in top_3]
        # rec_002 mentions "pain" and "slow builds" — pain-like
        assert "rec_002" in top_ids

    def test_objections_retrieval_favors_cost_records(self):
        idx = RecordIndex(RECORDS)
        top_3 = idx.retrieve("objections", k=3)
        top_ids = [r.record_id for r in top_3]
        # rec_005 mentions "cost" and "expensive" — objection-like
        assert "rec_005" in top_ids

    def test_different_sections_return_different_orders(self):
        idx = RecordIndex(RECORDS)
        goals_top = [r.record_id for r in idx.retrieve("goals", k=3)]
        pains_top = [r.record_id for r in idx.retrieve("pains", k=3)]
        # They might share some records, but ordering should differ
        assert goals_top != pains_top or len(RECORDS) <= 3

    def test_retrieve_global(self):
        idx = RecordIndex(RECORDS)
        result = idx.retrieve_global(k=3)
        assert len(result) == 3


# ── TF-IDF helpers ────────────────────────────────────────────────────

class TestTFIDF:
    def test_tokenize(self):
        tokens = _tokenize("Hello, World! Test 123.")
        assert tokens == ["hello", "world", "test", "123"]

    def test_cosine_identical(self):
        vec = {"a": 1.0, "b": 2.0}
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_cosine_orthogonal(self):
        vec_a = {"a": 1.0}
        vec_b = {"b": 1.0}
        assert _cosine_similarity(vec_a, vec_b) == 0.0

    def test_cosine_partial(self):
        vec_a = {"a": 1.0, "b": 1.0}
        vec_b = {"b": 1.0, "c": 1.0}
        sim = _cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0

    def test_idf_common_words_lower(self):
        docs = [["the", "cat"], ["the", "dog"], ["the", "bird"]]
        idf = _build_idf(docs)
        # "the" appears in all docs, should have lower IDF
        assert idf["the"] < idf["cat"]


# ── Prompt builder tests ──────────────────────────────────────────────

class TestPromptBuilder:
    def _make_cluster(self):
        from synthesis.models.cluster import (
            ClusterData,
            ClusterSummary,
            EnrichmentPayload,
            TenantContext,
        )
        return ClusterData(
            cluster_id="test-cluster",
            tenant=TenantContext(tenant_id="test-tenant", industry="SaaS"),
            summary=ClusterSummary(cluster_size=7, top_behaviors=["click"]),
            sample_records=RECORDS,
            enrichment=EnrichmentPayload(),
        )

    def test_default_no_retrieval(self):
        cluster = self._make_cluster()
        msgs = build_messages(cluster, retrieval_k=None)
        content = msgs[0]["content"]
        assert "top-" not in content.lower()
        # All records should appear
        for r in RECORDS:
            assert r.record_id in content

    def test_retrieval_augmented_message(self):
        cluster = self._make_cluster()
        msg = build_retrieval_augmented_message(cluster, top_k=3)
        assert "top-3 per section" in msg.lower()
        assert "Records relevant to goals" in msg
        assert "Records relevant to pains" in msg
        assert "Records relevant to objections" in msg

    def test_retrieval_messages_with_k(self):
        cluster = self._make_cluster()
        msgs = build_messages(cluster, retrieval_k=3)
        content = msgs[0]["content"]
        assert "top-3 per section" in content.lower()


# ── Evidence analysis tests ───────────────────────────────────────────

class TestAnalyzeEvidence:
    def test_all_high_confidence(self):
        persona = {
            "source_evidence": [
                {"claim": "a", "record_ids": ["rec_001"], "field_path": "goals.0", "confidence": 0.9},
                {"claim": "b", "record_ids": ["rec_002"], "field_path": "pains.0", "confidence": 0.85},
            ]
        }
        ev = analyze_evidence(persona, ["rec_001", "rec_002", "rec_003"])
        assert ev["longtail_hallucination_rate"] == 0.0
        assert ev["unique_cited"] == 2
        assert ev["coverage"] == 2 / 3

    def test_some_low_confidence(self):
        persona = {
            "source_evidence": [
                {"claim": "a", "record_ids": ["rec_001"], "field_path": "goals.0", "confidence": 0.9},
                {"claim": "b", "record_ids": ["rec_002"], "field_path": "pains.0", "confidence": 0.5},
                {"claim": "c", "record_ids": ["rec_003"], "field_path": "motivations.0", "confidence": 0.3},
            ]
        }
        ev = analyze_evidence(persona, ["rec_001", "rec_002", "rec_003"])
        assert ev["low_confidence"] == 2
        assert abs(ev["longtail_hallucination_rate"] - 2 / 3) < 0.01

    def test_empty_evidence(self):
        ev = analyze_evidence({"source_evidence": []}, ["rec_001"])
        assert ev["longtail_hallucination_rate"] == 0.0
        assert ev["coverage"] == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
