"""Unit tests for the deterministic source_evidence backfill.

Covers the failure mode from the Gauntlet-cohort and acme_corp runs: the
LLM produces an otherwise-valid tool_input but omits (or under-produces)
source_evidence, tripping Pydantic's min_length=3 or groundedness's
per-field-item requirement. The backfill synthesizes missing entries from
cluster records so the pipeline doesn't burn attempts on a field the model
dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))

from synthesis.engine.synthesizer import (  # noqa: E402
    _backfill_source_evidence,
    _schema_has_source_evidence,
)
from synthesis.models.cluster import (  # noqa: E402
    ClusterData,
    ClusterSummary,
    SampleRecord,
    TenantContext,
)
from synthesis.models.persona import (  # noqa: E402
    PersonaV1,
    PersonaV1VoiceFirst,
    PersonaV2,
    PublicPersonPersonaV1,
)


def _cluster(num_records: int = 4) -> ClusterData:
    return ClusterData(
        cluster_id="clust_test",
        tenant=TenantContext(tenant_id="t"),
        summary=ClusterSummary(cluster_size=num_records),
        sample_records=[
            SampleRecord(
                record_id=f"rec_{i:03d}",
                source="slack",
                timestamp="2026-04-18T12:00:00Z",
                payload={"body": f"sample message {i}"},
            )
            for i in range(num_records)
        ],
    )


def _persona_payload_no_evidence() -> dict:
    """An LLM output that has everything a persona needs EXCEPT source_evidence."""
    return {
        "schema_version": "1.0",
        "name": "Test Persona",
        "summary": "A test persona for backfill behavior",
        "demographics": {
            "age_range": "30-40",
            "gender_distribution": "mixed",
            "location_signals": ["US"],
        },
        "firmographics": {
            "company_size": "50-200",
            "industry": "SaaS",
            "role_level": "mid",
            "department_signals": ["engineering"],
        },
        "goals": ["Ship faster", "Reduce bugs"],
        "pains": ["Slow CI", "Flaky tests"],
        "motivations": ["Recognition", "Autonomy"],
        "objections": ["Budget"],
        "not_this": ["Jargon for its own sake", "Process theater"],
        "channels": ["slack"],
        "vocabulary": ["ship", "deploy", "pr"],
        "decision_triggers": ["outage"],
        "sample_quotes": ["we ship it now", "deploy early and often"],
        "journey_stages": [
            {"stage": "awareness", "mindset": "curious", "key_actions": ["read docs"], "content_preferences": ["blog"]},
            {"stage": "consideration", "mindset": "comparing", "key_actions": ["demo"], "content_preferences": ["case studies"]},
        ],
        "communication_style": {
            "tone": "direct",
            "formality": "casual",
            "vocabulary_level": "advanced",
            "preferred_channels": ["slack"],
        },
        "emotional_profile": {
            "baseline_mood": "focused",
            "stress_triggers": ["deadlines"],
            "coping_mechanisms": ["music"],
        },
        "moral_framework": {
            "core_values": ["craft", "autonomy"],
            "ethical_stance": "pragmatic",
            "moral_foundations": {"fairness": 0.8, "liberty": 0.7},
        },
    }


# --- Schema detection ------------------------------------------------------

def test_schema_has_source_evidence_covers_v1_and_v2():
    assert _schema_has_source_evidence(PersonaV1)
    assert _schema_has_source_evidence(PersonaV2)
    assert _schema_has_source_evidence(PersonaV1VoiceFirst)


def test_schema_has_source_evidence_excludes_public_person():
    """PublicPersonPersonaV1 has its own post-validate repair path and should
    not be double-processed by the pre-validate backfill."""
    assert not _schema_has_source_evidence(PublicPersonPersonaV1)


# --- Backfill creates entries when LLM omits source_evidence entirely -----

def test_backfill_fills_missing_source_evidence():
    cluster = _cluster(num_records=4)
    payload = _persona_payload_no_evidence()
    assert "source_evidence" not in payload

    result = _backfill_source_evidence(payload, cluster)
    evidence = result["source_evidence"]

    # Must satisfy PersonaV1.source_evidence min_length=3
    assert len(evidence) >= 3
    # Every entry must reference a real cluster record_id
    valid_ids = set(cluster.all_record_ids)
    for ev in evidence:
        assert set(ev["record_ids"]).issubset(valid_ids)
    # Each required field's item gets a field_path entry
    paths = {ev["field_path"] for ev in evidence}
    assert "goals.0" in paths
    assert "goals.1" in paths
    assert "pains.0" in paths
    assert "motivations.0" in paths
    assert "objections.0" in paths


def test_backfill_validates_against_personav1():
    """After backfill, PersonaV1.model_validate must succeed — proves the
    synthetic entries satisfy every SourceEvidence field constraint."""
    cluster = _cluster(num_records=4)
    payload = _persona_payload_no_evidence()

    result = _backfill_source_evidence(payload, cluster)
    # This is the real test: round-trip through Pydantic
    persona = PersonaV1.model_validate(result)
    assert len(persona.source_evidence) >= 3


# --- Backfill preserves existing LLM-produced evidence ---------------------

def test_backfill_preserves_llm_entries():
    cluster = _cluster(num_records=4)
    payload = _persona_payload_no_evidence()
    payload["source_evidence"] = [
        {
            "claim": "Ship faster — inferred from high PR throughput",
            "record_ids": ["rec_002"],
            "field_path": "goals.0",
            "confidence": 0.9,
            "status": "used",
        },
    ]

    result = _backfill_source_evidence(payload, cluster)

    # The LLM's entry survives untouched
    goals0 = [e for e in result["source_evidence"] if e["field_path"] == "goals.0"]
    assert len(goals0) == 1
    assert goals0[0]["confidence"] == 0.9
    assert goals0[0]["claim"].startswith("Ship faster")

    # And other required-field gaps are filled
    paths = {e["field_path"] for e in result["source_evidence"]}
    assert "goals.1" in paths
    assert "pains.0" in paths


# --- Edge cases ------------------------------------------------------------

def test_backfill_tops_up_to_min_length_when_claims_are_sparse():
    """A persona with only one required-field item still needs min_length=3
    evidence. The backfill tops up with summary-rooted entries so the
    remaining slots get valid grounding."""
    cluster = _cluster(num_records=4)
    payload = _persona_payload_no_evidence()
    # Strip all required-field items except one goal
    payload["goals"] = ["Only goal"]
    payload["pains"] = []
    payload["motivations"] = []
    payload["objections"] = []

    result = _backfill_source_evidence(payload, cluster)
    assert len(result["source_evidence"]) >= 3
    # At least one entry covers goals.0
    paths = [e["field_path"] for e in result["source_evidence"]]
    assert "goals.0" in paths


def test_backfill_noop_when_cluster_has_no_records():
    """If the cluster somehow has zero records, backfill shouldn't invent
    record_ids — the pipeline should fail loudly at validation instead."""
    # ClusterData requires sample_records min_length=1, so construct via
    # a model_validate that bypasses to simulate an empty edge case.
    cluster = _cluster(num_records=1)
    # Simulate an empty all_record_ids by monkey-patching
    object.__setattr__(cluster, "sample_records", [])
    payload = _persona_payload_no_evidence()

    result = _backfill_source_evidence(payload, cluster)
    # No record IDs available → no synthetic entries can be safely made
    assert "source_evidence" not in result or result["source_evidence"] == []


def test_backfill_is_idempotent():
    """Calling backfill twice produces the same result as calling once."""
    cluster = _cluster(num_records=4)
    payload = _persona_payload_no_evidence()

    once = _backfill_source_evidence(dict(payload), cluster)
    twice = _backfill_source_evidence(once, cluster)

    assert once["source_evidence"] == twice["source_evidence"]
