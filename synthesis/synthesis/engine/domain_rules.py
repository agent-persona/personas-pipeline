"""
Domain-specific grounding rules.
Identifies claims in persona fields that touch sensitive categories
(financial, medical, legal, security) and flags them for N+1 source enforcement.
"""
from __future__ import annotations

SENSITIVE_KEYWORDS: dict[str, list[str]] = {
    "financial": ["budget", "cost", "price", "revenue", "spend", "roi", "invest", "money", "pay", "salary"],
    "medical": ["health", "medical", "diagnosis", "treatment", "patient", "clinical", "symptom"],
    "security": ["password", "auth", "credential", "vulnerability", "breach", "exploit", "attack"],
    "legal": ["compliance", "regulation", "gdpr", "hipaa", "legal", "liability", "audit"],
}

# Minimum N+1 sources required for sensitive claims (baseline=1, so sensitive=2)
N_PLUS_1_THRESHOLD = 2

# Fields to inspect for claim text (covers all text-bearing persona fields)
CLAIM_FIELDS = [
    "goals",
    "pains",
    "motivations",
    "objections",
    "decision_triggers",
    "sample_quotes",
    "vocabulary",
]


def classify_claim(text: str) -> list[str]:
    """Return list of sensitive categories found in text (case-insensitive keyword match)."""
    lower = text.lower()
    found = []
    for category, keywords in SENSITIVE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.append(category)
    return found


def _sources_for_path(field_path: str, source_evidence: list[dict]) -> int:
    """Count the number of unique source record_ids backing a given field_path."""
    ids: set[str] = set()
    for ev in source_evidence:
        if ev.get("field_path") == field_path:
            ids.update(ev.get("record_ids", []))
    return len(ids)


def check_domain_rules(persona: dict, cluster: dict) -> dict:
    """
    Inspect all claim-bearing fields in a persona dict and flag entries that
    match sensitive keyword categories. For flagged claims, check whether the
    N+1 source requirement (>=2 unique record_ids) is satisfied.

    Returns:
    {
      "total_claims_checked": int,
      "sensitive_claims_found": int,
      "false_positive_rate": float,  # claims flagged / total claims
      "category_breakdown": {category: count},
      "flagged_claims": [
          {
              "field": str,
              "text": str,
              "categories": list[str],
              "sources_available": int,
              "n_plus_1_satisfied": bool,
          }
      ]
    }
    """
    source_evidence: list[dict] = persona.get("source_evidence", [])
    category_breakdown: dict[str, int] = {cat: 0 for cat in SENSITIVE_KEYWORDS}
    flagged_claims: list[dict] = []
    total_claims_checked = 0

    for field_name in CLAIM_FIELDS:
        items = persona.get(field_name, [])
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, str):
                continue
            total_claims_checked += 1
            categories = classify_claim(item)
            if not categories:
                continue

            field_path = f"{field_name}.{idx}"
            sources_available = _sources_for_path(field_path, source_evidence)
            n_plus_1_satisfied = sources_available >= N_PLUS_1_THRESHOLD

            for cat in categories:
                category_breakdown[cat] += 1

            flagged_claims.append(
                {
                    "field": field_path,
                    "text": item,
                    "categories": categories,
                    "sources_available": sources_available,
                    "n_plus_1_satisfied": n_plus_1_satisfied,
                }
            )

    sensitive_claims_found = len(flagged_claims)
    false_positive_rate = (
        sensitive_claims_found / total_claims_checked if total_claims_checked > 0 else 0.0
    )

    return {
        "total_claims_checked": total_claims_checked,
        "sensitive_claims_found": sensitive_claims_found,
        "false_positive_rate": round(false_positive_rate, 4),
        "category_breakdown": {k: v for k, v in category_breakdown.items() if v > 0},
        "flagged_claims": flagged_claims,
    }
