"""Tests for Experiment 3.20: Confidence-weighted corroboration.

Verifies:
1. SourceEvidence.corroboration_depth computed correctly
2. Corroboration check flags over-confident claims
3. Calibration score computation
4. Groundedness enforcement mode
5. Edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from synthesis.models.evidence import SourceEvidence
from synthesis.engine.groundedness import (
    CorroborationReport,
    check_corroboration,
    check_groundedness,
    HIGH_CONFIDENCE_THRESHOLD,
    MIN_CORROBORATION_FOR_HIGH_CONFIDENCE,
)
from synthesis.models.persona import PersonaV1
from synthesis.models.cluster import ClusterData


# ── Fixtures ─────────────────────────────────────────────────────────

def _evidence(
    confidence: float = 0.9,
    record_ids: list[str] | None = None,
    field_path: str = "goals.0",
) -> SourceEvidence:
    return SourceEvidence(
        claim="test claim",
        record_ids=record_ids or ["rec_001"],
        field_path=field_path,
        confidence=confidence,
    )


def _minimal_persona(evidence: list[SourceEvidence]) -> PersonaV1:
    return PersonaV1(
        name="Test",
        summary="A test persona",
        demographics={"age_range": "25-34", "gender_distribution": "mixed",
                       "location_signals": ["US"]},
        firmographics={},
        goals=["g1", "g2"],
        pains=["p1", "p2"],
        motivations=["m1", "m2"],
        objections=["o1"],
        channels=["email"],
        vocabulary=["agile", "sprint", "kanban"],
        decision_triggers=["free trial"],
        sample_quotes=["I need speed", "Time is money"],
        journey_stages=[
            {"stage": "awareness", "mindset": "curious", "key_actions": ["search"],
             "content_preferences": ["blogs"]},
            {"stage": "decision", "mindset": "evaluating", "key_actions": ["demo"],
             "content_preferences": ["case studies"]},
        ],
        source_evidence=evidence,
    )


def _cluster() -> ClusterData:
    return ClusterData.model_validate({
        "cluster_id": "test",
        "tenant": {"tenant_id": "t", "industry": "tech", "product_description": "p"},
        "summary": {"cluster_size": 2, "top_behaviors": [], "top_pages": []},
        "sample_records": [
            {"record_id": "rec_001", "source": "ga4", "payload": {}},
            {"record_id": "rec_002", "source": "hubspot", "payload": {}},
            {"record_id": "rec_003", "source": "intercom", "payload": {}},
        ],
        "enrichment": {},
    })


# ── Corroboration depth tests ────────────────────────────────────────

class TestCorroborationDepth:
    def test_single_record(self):
        ev = _evidence(record_ids=["rec_001"])
        assert ev.corroboration_depth == 1

    def test_multiple_records(self):
        ev = _evidence(record_ids=["rec_001", "rec_002", "rec_003"])
        assert ev.corroboration_depth == 3

    def test_available_on_instance(self):
        ev = _evidence(record_ids=["rec_001", "rec_002"])
        d = ev.model_dump()
        assert d["corroboration_depth"] == 2


# ── Corroboration check tests ────────────────────────────────────────

class TestCheckCorroboration:
    def test_over_confident_flagged(self):
        """High confidence + 1 record = over-confident."""
        evidence = [
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="goals.0"),
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="goals.1"),
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="pains.0"),
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="pains.1"),
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="motivations.0"),
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="motivations.1"),
            _evidence(confidence=0.9, record_ids=["rec_001"], field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report = check_corroboration(persona)
        assert report.over_confident_count == 7
        assert len(report.violations) == 7

    def test_well_corroborated_clean(self):
        """High confidence + multiple records = well-corroborated."""
        evidence = [
            _evidence(confidence=0.9, record_ids=["rec_001", "rec_002"], field_path="goals.0"),
            _evidence(confidence=0.85, record_ids=["rec_001", "rec_003"], field_path="goals.1"),
            _evidence(confidence=0.9, record_ids=["rec_002", "rec_003"], field_path="pains.0"),
            _evidence(confidence=0.8, record_ids=["rec_001", "rec_002"], field_path="pains.1"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="motivations.0"),
            _evidence(confidence=0.6, record_ids=["rec_002"], field_path="motivations.1"),
            _evidence(confidence=0.5, record_ids=["rec_003"], field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report = check_corroboration(persona)
        assert report.over_confident_count == 0
        assert len(report.violations) == 0

    def test_under_confident_detected(self):
        """Low confidence + many records = under-confident."""
        evidence = [
            _evidence(confidence=0.3, record_ids=["rec_001", "rec_002", "rec_003"],
                      field_path="goals.0"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="goals.1"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="pains.0"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="pains.1"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="motivations.0"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="motivations.1"),
            _evidence(confidence=0.7, record_ids=["rec_001"], field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report = check_corroboration(persona)
        assert report.under_confident_count == 1

    def test_calibration_perfect(self):
        """When confidence exactly matches normalized corroboration."""
        evidence = [
            _evidence(confidence=0.2, record_ids=["rec_001"], field_path="goals.0"),
            _evidence(confidence=0.4, record_ids=["rec_001", "rec_002"], field_path="goals.1"),
            _evidence(confidence=0.2, record_ids=["rec_001"], field_path="pains.0"),
            _evidence(confidence=0.4, record_ids=["rec_001", "rec_002"], field_path="pains.1"),
            _evidence(confidence=0.2, record_ids=["rec_001"], field_path="motivations.0"),
            _evidence(confidence=0.4, record_ids=["rec_001", "rec_002"], field_path="motivations.1"),
            _evidence(confidence=0.2, record_ids=["rec_001"], field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report = check_corroboration(persona)
        assert report.calibration_score == pytest.approx(1.0, abs=0.01)

    def test_empty_evidence(self):
        report = check_corroboration(
            _minimal_persona([
                _evidence(field_path="goals.0"),
                _evidence(field_path="goals.1"),
                _evidence(field_path="pains.0"),
                _evidence(field_path="pains.1"),
                _evidence(field_path="motivations.0"),
                _evidence(field_path="motivations.1"),
                _evidence(field_path="objections.0"),
            ])
        )
        assert report.total_evidence == 7


# ── Groundedness enforcement tests ───────────────────────────────────

class TestGroundednessEnforcement:
    def test_no_enforcement_passes(self):
        """Without enforcement, over-confident claims don't affect score."""
        evidence = [
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="goals.0"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="goals.1"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="pains.0"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="pains.1"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="motivations.0"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="motivations.1"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report = check_groundedness(persona, _cluster(), enforce_corroboration=False)
        assert report.passed
        assert report.corroboration is not None
        assert report.corroboration.over_confident_count > 0

    def test_enforcement_penalizes(self):
        """With enforcement, over-confident claims lower the score."""
        evidence = [
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="goals.0"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="goals.1"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="pains.0"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="pains.1"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="motivations.0"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="motivations.1"),
            _evidence(confidence=0.95, record_ids=["rec_001"], field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report_ctrl = check_groundedness(persona, _cluster(), enforce_corroboration=False)
        report_enf = check_groundedness(persona, _cluster(), enforce_corroboration=True)
        assert report_enf.score < report_ctrl.score
        assert len(report_enf.violations) > len(report_ctrl.violations)

    def test_corroboration_report_attached(self):
        evidence = [
            _evidence(field_path="goals.0"),
            _evidence(field_path="goals.1"),
            _evidence(field_path="pains.0"),
            _evidence(field_path="pains.1"),
            _evidence(field_path="motivations.0"),
            _evidence(field_path="motivations.1"),
            _evidence(field_path="objections.0"),
        ]
        persona = _minimal_persona(evidence)
        report = check_groundedness(persona, _cluster())
        assert report.corroboration is not None
        assert isinstance(report.corroboration, CorroborationReport)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
