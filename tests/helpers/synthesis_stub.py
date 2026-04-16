"""Deterministic synthesis stub for testing downstream persona quality without LLM calls."""

from __future__ import annotations


def stub_persona_from_cluster(cluster: dict) -> dict:
    """Generate a deterministic persona-like dict from cluster data.

    This is NOT a real persona — it's a structural stub that mirrors PersonaV1
    shape for testing handoff compatibility. No LLM call is made.
    """
    summary = cluster["summary"]
    tenant = cluster["tenant"]
    sample = cluster.get("sample_records", [])

    # Extract behaviors and pages for goals/pains
    top_b = summary.get("top_behaviors", [])
    top_p = summary.get("top_pages", [])

    return {
        "name": f"Persona from {cluster['cluster_id']}",
        "summary": f"A user persona based on {summary['cluster_size']} users.",
        "demographics": {
            "age_range": "25-45",
            "gender_distribution": "mixed",
            "location_signals": [],
            "education_level": None,
            "income_bracket": None,
        },
        "firmographics": {
            "company_size": None,
            "industry": tenant.get("industry"),
            "role_titles": [],
            "tech_stack_signals": [],
        },
        "goals": [{"goal": b, "source_evidence": []} for b in top_b[:3]],
        "pains": [{"pain": b, "source_evidence": []} for b in top_b[3:5]],
        "motivations": [{"motivation": "efficiency"}],
        "objections": [{"objection": "pricing"}],
        "channels": [],
        "vocabulary": top_b[:5],
        "decision_triggers": [],
        "sample_quotes": [sr.get("payload", {}).get("message", "N/A") for sr in sample[:2]],
        "journey_stages": [],
        "source_evidence": [
            {
                "claim": f"Uses {top_b[0]}" if top_b else "Unknown",
                "record_ids": [sr["record_id"] for sr in sample[:2]],
                "field_path": "goals[0]",
                "confidence": 0.9,
            }
        ] if sample else [],
        "_groundedness_proxy": len([sr for sr in sample if sr.get("payload")]) / max(len(sample), 1),
        "_cluster_id": cluster["cluster_id"],
    }


def stub_groundedness_score(persona: dict, cluster: dict) -> float:
    """Proxy groundedness: fraction of sample records with non-empty payloads."""
    sample = cluster.get("sample_records", [])
    if not sample:
        return 0.0
    with_payload = sum(1 for sr in sample if sr.get("payload"))
    return with_payload / len(sample)
