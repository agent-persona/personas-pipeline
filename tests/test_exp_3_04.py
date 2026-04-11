"""Tests for Experiment 3.04: Evidence-first generation.

Verifies:
1. Evidence package data structures
2. Evidence brief builder
3. Demographic hallucination rate metric
4. Evidence selection tool schema
5. Conditioned synthesis prompt content
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.engine.evidence_first import (
    CONDITIONED_SYSTEM_PROMPT,
    EVIDENCE_SELECTION_TOOL,
    EVIDENCE_SELECTOR_PROMPT,
    EvidencePackage,
    SelectedRecord,
    _build_evidence_brief,
)
from synthesis.models.cluster import ClusterData

from experiment_3_04 import compute_demo_hallucination_rate


# ── Fixtures ─────────────────────────────────────────────────────────

def _cluster() -> ClusterData:
    return ClusterData.model_validate({
        "cluster_id": "clust_test",
        "tenant": {
            "tenant_id": "t1",
            "industry": "B2B SaaS",
            "product_description": "Project management tool",
        },
        "summary": {
            "cluster_size": 5,
            "top_behaviors": ["api_setup", "dashboard_view"],
            "top_pages": ["/docs", "/pricing"],
        },
        "sample_records": [
            {"record_id": "rec_001", "source": "ga4", "payload": {}},
            {"record_id": "rec_002", "source": "hubspot", "payload": {}},
        ],
        "enrichment": {},
    })


def _evidence_package() -> EvidencePackage:
    return EvidencePackage(
        selected_records=[
            SelectedRecord(
                record_id="rec_001",
                source="ga4",
                extracted_signal="User visited /docs 15 times",
                relevance="High docs engagement suggests technical role",
                category="behavior",
            ),
            SelectedRecord(
                record_id="rec_002",
                source="hubspot",
                extracted_signal="Requested API pricing info",
                relevance="Active buyer intent signal",
                category="goal",
            ),
        ],
        cluster_theme="Technical users evaluating API-first tools",
        record_ids=["rec_001", "rec_002"],
        selection_cost_usd=0.005,
    )


# ── Evidence package tests ───────────────────────────────────────────

class TestEvidencePackage:
    def test_record_ids_populated(self):
        pkg = _evidence_package()
        assert len(pkg.record_ids) == 2
        assert "rec_001" in pkg.record_ids

    def test_selected_records_categories(self):
        pkg = _evidence_package()
        categories = {r.category for r in pkg.selected_records}
        assert "behavior" in categories
        assert "goal" in categories


# ── Evidence brief tests ─────────────────────────────────────────────

class TestEvidenceBrief:
    def test_contains_cluster_theme(self):
        brief = _build_evidence_brief(_evidence_package(), _cluster())
        assert "Technical users evaluating API-first tools" in brief

    def test_contains_record_ids(self):
        brief = _build_evidence_brief(_evidence_package(), _cluster())
        assert "rec_001" in brief
        assert "rec_002" in brief

    def test_contains_extracted_signals(self):
        brief = _build_evidence_brief(_evidence_package(), _cluster())
        assert "visited /docs 15 times" in brief
        assert "API pricing info" in brief

    def test_contains_tenant_context(self):
        brief = _build_evidence_brief(_evidence_package(), _cluster())
        assert "B2B SaaS" in brief

    def test_contains_only_instruction(self):
        brief = _build_evidence_brief(_evidence_package(), _cluster())
        assert "ONLY" in brief


# ── Demographic hallucination tests ──────────────────────────────────

class TestDemoHallucination:
    def test_no_demographics_zero(self):
        persona = {"demographics": {}}
        assert compute_demo_hallucination_rate(persona) == 0.0

    def test_all_generic_zero(self):
        persona = {
            "demographics": {
                "age_range": "unknown",
                "gender_distribution": "mixed",
            },
            "source_evidence": [],
        }
        assert compute_demo_hallucination_rate(persona) == 0.0

    def test_specific_with_evidence_zero(self):
        persona = {
            "demographics": {
                "age_range": "25-34",
            },
            "source_evidence": [
                {"field_path": "demographics.age_range", "claim": "age range 25-34",
                 "record_ids": ["r1"], "confidence": 0.9},
            ],
        }
        rate = compute_demo_hallucination_rate(persona)
        assert rate == 0.0

    def test_specific_without_evidence_flagged(self):
        persona = {
            "demographics": {
                "age_range": "25-34",
                "education_level": "masters",
            },
            "source_evidence": [],  # no evidence at all
        }
        rate = compute_demo_hallucination_rate(persona)
        assert rate == 1.0  # 2 specific claims, 0 evidenced

    def test_mixed(self):
        persona = {
            "demographics": {
                "age_range": "25-34",        # specific, unevidenced
                "gender_distribution": "mixed",  # generic, ignored
                "education_level": "masters",  # specific, evidenced
            },
            "source_evidence": [
                {"field_path": "demographics.education_level",
                 "claim": "education level masters",
                 "record_ids": ["r1"], "confidence": 0.8},
            ],
        }
        rate = compute_demo_hallucination_rate(persona)
        assert rate == pytest.approx(0.5)  # 1 of 2 specific claims unevidenced

    def test_empty_persona(self):
        assert compute_demo_hallucination_rate({}) == 0.0


# ── Prompt/tool schema tests ────────────────────────────────────────

class TestPrompts:
    def test_selector_prompt_mentions_records(self):
        assert "record" in EVIDENCE_SELECTOR_PROMPT.lower()

    def test_selector_prompt_mentions_verbatim(self):
        assert "verbatim" in EVIDENCE_SELECTOR_PROMPT.lower()

    def test_conditioned_prompt_mentions_exclusively(self):
        assert "EXCLUSIVELY" in CONDITIONED_SYSTEM_PROMPT

    def test_conditioned_prompt_warns_hallucination(self):
        assert "hallucinate" in CONDITIONED_SYSTEM_PROMPT.lower() or "fabricate" in CONDITIONED_SYSTEM_PROMPT.lower()

    def test_selection_tool_schema_valid(self):
        schema = EVIDENCE_SELECTION_TOOL["input_schema"]
        assert "selected_records" in schema["properties"]
        assert "cluster_theme" in schema["properties"]
        items = schema["properties"]["selected_records"]["items"]
        assert "record_id" in items["properties"]
        assert "extracted_signal" in items["properties"]
        assert "category" in items["properties"]

    def test_selection_tool_categories(self):
        cats = EVIDENCE_SELECTION_TOOL["input_schema"]["properties"]["selected_records"]["items"]["properties"]["category"]["enum"]
        assert "goal" in cats
        assert "pain" in cats
        assert "demographic" in cats
        assert "vocabulary" in cats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
