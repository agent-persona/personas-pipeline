"""Tests for Experiment 2.11: Multi-agent debate.

Verifies:
1. Verdict application (keep/revise/drop)
2. Debate data structures
3. Cluster summary builder
4. Evidence stats computation
5. Edge cases (no challenges, all dropped, empty evidence)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.engine.debate import (
    Challenge,
    DebateRound,
    Verdict,
    _apply_verdicts,
    _build_cluster_summary,
)

from experiment_2_11 import _evidence_stats


# ── Fixtures ─────────────────────────────────────────────────────────

def _persona_with_evidence(n_evidence: int = 3, confidence: float = 0.9) -> dict:
    return {
        "name": "Test Persona",
        "summary": "A test",
        "goals": ["g1", "g2"],
        "pains": ["p1", "p2"],
        "source_evidence": [
            {
                "claim": f"claim {i}",
                "record_ids": ["rec_001"],
                "field_path": f"goals.{i}" if i < 2 else f"pains.{i-2}",
                "confidence": confidence,
            }
            for i in range(n_evidence)
        ],
    }


# ── Verdict application tests ────────────────────────────────────────

class TestApplyVerdicts:
    def test_all_kept(self):
        persona = _persona_with_evidence(3)
        verdicts = [
            Verdict(field_path="goals.0", verdict="keep", reasoning="ok"),
            Verdict(field_path="goals.1", verdict="keep", reasoning="ok"),
            Verdict(field_path="pains.0", verdict="keep", reasoning="ok"),
        ]
        result, kept, revised, dropped = _apply_verdicts(persona, verdicts)
        assert kept == 3
        assert revised == 0
        assert dropped == 0
        assert len(result["source_evidence"]) == 3

    def test_one_dropped(self):
        persona = _persona_with_evidence(3)
        verdicts = [
            Verdict(field_path="goals.0", verdict="drop", reasoning="hallucinated"),
        ]
        result, kept, revised, dropped = _apply_verdicts(persona, verdicts)
        assert dropped == 1
        assert len(result["source_evidence"]) == 2  # one removed

    def test_one_revised(self):
        persona = _persona_with_evidence(3, confidence=0.9)
        verdicts = [
            Verdict(field_path="goals.0", verdict="revise", reasoning="needs qualification"),
        ]
        result, kept, revised, dropped = _apply_verdicts(persona, verdicts)
        assert revised == 1
        # Confidence should be lowered
        revised_ev = [e for e in result["source_evidence"] if e["field_path"] == "goals.0"]
        assert len(revised_ev) == 1
        assert revised_ev[0]["confidence"] == pytest.approx(0.7, abs=0.01)

    def test_mixed_verdicts(self):
        persona = _persona_with_evidence(4)
        verdicts = [
            Verdict(field_path="goals.0", verdict="keep", reasoning="ok"),
            Verdict(field_path="goals.1", verdict="revise", reasoning="weak"),
            Verdict(field_path="pains.0", verdict="drop", reasoning="fabricated"),
            Verdict(field_path="pains.1", verdict="keep", reasoning="ok"),
        ]
        result, kept, revised, dropped = _apply_verdicts(persona, verdicts)
        assert kept == 2
        assert revised == 1
        assert dropped == 1
        assert len(result["source_evidence"]) == 3  # 4 - 1 dropped

    def test_no_verdicts(self):
        persona = _persona_with_evidence(3)
        result, kept, revised, dropped = _apply_verdicts(persona, [])
        # No verdicts = all evidence kept by default
        assert kept == 3
        assert revised == 0
        assert dropped == 0

    def test_confidence_floor(self):
        """Confidence should not go below 0.3 after revision."""
        persona = _persona_with_evidence(1, confidence=0.4)
        verdicts = [
            Verdict(field_path="goals.0", verdict="revise", reasoning="lower"),
        ]
        result, _, revised, _ = _apply_verdicts(persona, verdicts)
        assert revised == 1
        assert result["source_evidence"][0]["confidence"] == pytest.approx(0.3, abs=0.01)

    def test_original_not_mutated(self):
        persona = _persona_with_evidence(2)
        original_len = len(persona["source_evidence"])
        verdicts = [Verdict(field_path="goals.0", verdict="drop", reasoning="bad")]
        result, _, _, _ = _apply_verdicts(persona, verdicts)
        assert len(persona["source_evidence"]) == original_len  # original unchanged
        assert len(result["source_evidence"]) == original_len - 1


# ── Evidence stats tests ─────────────────────────────────────────────

class TestEvidenceStats:
    def test_normal(self):
        persona = _persona_with_evidence(3, confidence=0.8)
        count, avg, low = _evidence_stats(persona)
        assert count == 3
        assert avg == pytest.approx(0.8)
        assert low == 0

    def test_low_confidence(self):
        persona = {
            "source_evidence": [
                {"confidence": 0.5},
                {"confidence": 0.9},
                {"confidence": 0.3},
            ],
        }
        count, avg, low = _evidence_stats(persona)
        assert count == 3
        assert low == 2  # 0.5 and 0.3 are below 0.7

    def test_empty(self):
        count, avg, low = _evidence_stats({})
        assert count == 0
        assert avg == 0.0


# ── Data structure tests ─────────────────────────────────────────────

class TestDataStructures:
    def test_challenge_fields(self):
        c = Challenge(
            field_path="goals.0",
            claim="reduce costs",
            challenge="no data supports this",
            severity="high",
        )
        assert c.severity == "high"

    def test_verdict_fields(self):
        v = Verdict(
            field_path="goals.0",
            verdict="drop",
            reasoning="hallucinated",
        )
        assert v.verdict == "drop"

    def test_debate_round_defaults(self):
        dr = DebateRound(challenges=[], verdicts=[])
        assert dr.kept == 0
        assert dr.revised == 0
        assert dr.dropped == 0
        assert dr.adversary_cost_usd == 0.0


# ── Cluster summary tests ────────────────────────────────────────────

class TestClusterSummary:
    def test_builds_summary(self):
        from synthesis.models.cluster import ClusterData

        cluster = ClusterData.model_validate({
            "cluster_id": "clust_123",
            "tenant": {
                "tenant_id": "t1",
                "industry": "tech",
                "product_description": "product",
            },
            "summary": {
                "cluster_size": 10,
                "top_behaviors": ["api_setup", "dashboard_view"],
                "top_pages": ["/docs", "/pricing"],
            },
            "sample_records": [
                {"record_id": "rec_001", "source": "ga4", "payload": {}},
            ],
            "enrichment": {},
        })

        summary = _build_cluster_summary(cluster)
        assert "clust_123" in summary
        assert "api_setup" in summary
        assert "rec_001" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
