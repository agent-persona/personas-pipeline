"""Tests for semantic_groundedness_proxy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "evaluation"))
from evaluation.metrics import semantic_groundedness_proxy

CLUSTER = {
    "sample_records": [
        {
            "record_id": "r001",
            "source": "product",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"behavior": "webhook_config", "page": "/settings/webhooks"},
        },
        {
            "record_id": "r002",
            "source": "product",
            "timestamp": "2026-01-02T00:00:00Z",
            "payload": {"behavior": "graphql_query", "page": "/api/graphql"},
        },
        {
            "record_id": "r003",
            "source": "email",
            "timestamp": "2026-01-03T00:00:00Z",
            "payload": {"subject": "onboarding follow-up", "opened": True},
        },
    ]
}

def _persona(goals, pains, evidence):
    return {
        "goals": goals,
        "pains": pains,
        "motivations": [],
        "objections": [],
        "source_evidence": evidence,
    }

def test_matching_evidence_scores_high():
    """A claim about webhooks citing the webhook record should score well."""
    persona = _persona(
        goals=["Configure webhook integrations reliably"],
        pains=[],
        evidence=[{"field_path": "goals.0", "record_ids": ["r001"]}],
    )
    result = semantic_groundedness_proxy(persona, CLUSTER)
    assert result["semantic_score"] > 0.1
    assert result["weak_count"] == 0

def test_unrelated_evidence_scores_low():
    """A claim about email marketing citing the webhook record should score low."""
    persona = _persona(
        goals=["Drive email marketing campaigns"],
        pains=[],
        evidence=[{"field_path": "goals.0", "record_ids": ["r001"]}],
    )
    result = semantic_groundedness_proxy(persona, CLUSTER)
    assert result["weak_count"] > 0

def test_no_evidence_returns_zero():
    """Persona with no source_evidence returns zeros."""
    persona = _persona(goals=["Some goal"], pains=[], evidence=[])
    result = semantic_groundedness_proxy(persona, CLUSTER)
    assert result["claim_count"] == 0
    assert result["semantic_score"] == 0.0

def test_multiple_records_union():
    """Evidence citing multiple records should use union of their tokens."""
    persona = _persona(
        goals=["webhook graphql integration"],
        pains=[],
        evidence=[{"field_path": "goals.0", "record_ids": ["r001", "r002"]}],
    )
    result = semantic_groundedness_proxy(persona, CLUSTER)
    assert result["semantic_score"] > 0.2  # both records contribute matching tokens

def test_coverage_metric():
    """Coverage = claims with evidence / total claims."""
    persona = _persona(
        goals=["goal with evidence", "goal without evidence"],
        pains=[],
        evidence=[{"field_path": "goals.0", "record_ids": ["r001"]}],
    )
    result = semantic_groundedness_proxy(persona, CLUSTER)
    assert result["coverage"] == 0.5  # 1 of 2 goals has evidence
