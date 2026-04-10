"""
B3: Re-run exp-3.22 with semantic groundedness proxy.

Compares domain-rule-flagged claims vs. non-flagged claims using semantic
overlap between claim tokens and backing record tokens.

Key question: Does the keyword classifier correlate with actual semantic weakness?
If yes, the N+1 rule is catching genuinely weak grounding. If no, it's just
triggering on keywords regardless of evidence quality.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE_3_22 = Path(
    "/Users/ivanma/Desktop/gauntlet/Capstone/personas-pipeline"
    "/.worktrees/exp-3.22-domain-specific-grounding-rules/output"
)
OUT_DIR = Path(__file__).parent.parent / "output" / "experiments" / "exp-b3-rerun-3.22-semantic"

# ---------------------------------------------------------------------------
# Semantic proxy (inline per spec)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "for", "to", "with", "that",
    "their", "our", "is", "are", "was", "were", "be", "been", "have", "has",
    "this", "it", "its", "at", "by", "from", "on", "as", "not", "but", "so",
    "they", "we", "you", "he", "she", "also", "which", "what", "when", "how",
    "can", "will", "would", "could", "should", "may", "might", "do", "did",
    "use", "used", "using", "get", "got", "need", "needs", "work", "works",
}


def _claim_tokens(text: str) -> set[str]:
    return {
        w.lower().strip(".,;:'\"()")
        for w in text.split()
        if len(w) > 3 and w.lower().strip(".,;:'\"()") not in _STOPWORDS
    }


def _payload_tokens(text: str) -> set[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", text)
    return {p.lower() for p in parts if len(p) > 3 and p.lower() not in _STOPWORDS}


def semantic_groundedness_proxy(persona: dict, cluster: dict) -> dict:
    records_by_id = {r["record_id"]: r for r in cluster.get("sample_records", [])}
    evidence_map = {e["field_path"]: e["record_ids"] for e in persona.get("source_evidence", [])}
    FIELDS = ["goals", "pains", "motivations", "objections"]
    overlaps, weak, covered = [], [], 0
    for field in FIELDS:
        for i, item in enumerate(persona.get(field, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            field_path = f"{field}.{i}"
            record_ids = evidence_map.get(field_path, [])
            if not record_ids:
                continue
            covered += 1
            claim_tokens = _claim_tokens(text)
            record_tokens: set[str] = set()
            for rid in record_ids:
                rec = records_by_id.get(rid, {})
                for v in rec.get("payload", {}).values():
                    record_tokens |= _payload_tokens(str(v))
            overlap = len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
            overlaps.append(overlap)
            if overlap < 0.1:
                weak.append(
                    {"field_path": field_path, "claim": text[:100], "overlap": round(overlap, 4)}
                )
    total_claims = sum(len(persona.get(f, [])) for f in FIELDS)
    score = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return {
        "semantic_score": round(score, 4),
        "weak_pairs": weak,
        "claim_count": len(overlaps),
        "weak_count": len(weak),
        "coverage": round(covered / max(total_claims, 1), 4),
    }


# ---------------------------------------------------------------------------
# Domain-rules classifier (inline, mirrors exp-3.22 domain_rules.py)
# ---------------------------------------------------------------------------
SENSITIVE_KEYWORDS: dict[str, list[str]] = {
    "financial": ["budget", "cost", "price", "revenue", "spend", "roi", "invest", "money", "pay", "salary"],
    "medical": ["health", "medical", "diagnosis", "treatment", "patient", "clinical", "symptom"],
    "security": ["password", "auth", "credential", "vulnerability", "breach", "exploit", "attack"],
    "legal": ["compliance", "regulation", "gdpr", "hipaa", "legal", "liability", "audit"],
}

N_PLUS_1_THRESHOLD = 2

# Only the four core content fields — decision_triggers and sample_quotes are
# excluded from semantic analysis because they have no source_evidence entries,
# which is the structural gap noted in exp-3.22.
SEMANTIC_FIELDS = ["goals", "pains", "motivations", "objections"]

# All fields the domain classifier inspects (same as exp-3.22)
CLASSIFIER_FIELDS = [
    "goals", "pains", "motivations", "objections",
    "decision_triggers", "sample_quotes", "vocabulary",
]


def classify_claim(text: str) -> list[str]:
    lower = text.lower()
    return [cat for cat, kws in SENSITIVE_KEYWORDS.items() if any(kw in lower for kw in kws)]


def _sources_for_path(field_path: str, source_evidence: list[dict]) -> int:
    ids: set[str] = set()
    for ev in source_evidence:
        if ev.get("field_path") == field_path:
            ids.update(ev.get("record_ids", []))
    return len(ids)


def check_domain_rules(persona: dict) -> dict:
    source_evidence = persona.get("source_evidence", [])
    category_breakdown: dict[str, int] = {cat: 0 for cat in SENSITIVE_KEYWORDS}
    flagged_claims: list[dict] = []
    total_claims_checked = 0

    for field_name in CLASSIFIER_FIELDS:
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
    flagging_rate = sensitive_claims_found / total_claims_checked if total_claims_checked > 0 else 0.0
    return {
        "total_claims_checked": total_claims_checked,
        "sensitive_claims_found": sensitive_claims_found,
        "flagging_rate": round(flagging_rate, 4),
        "category_breakdown": {k: v for k, v in category_breakdown.items() if v > 0},
        "flagged_claims": flagged_claims,
    }


# ---------------------------------------------------------------------------
# Per-claim semantic overlap (for individual flagged/non-flagged comparison)
# ---------------------------------------------------------------------------
def compute_per_claim_overlap(
    persona: dict, cluster: dict
) -> dict[str, float]:
    """
    Returns a mapping of field_path -> semantic overlap score for every claim
    in SEMANTIC_FIELDS that has at least one backing record.
    """
    records_by_id = {r["record_id"]: r for r in cluster.get("sample_records", [])}
    evidence_map = {e["field_path"]: e["record_ids"] for e in persona.get("source_evidence", [])}
    result: dict[str, float] = {}

    for field in SEMANTIC_FIELDS:
        for i, item in enumerate(persona.get(field, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            field_path = f"{field}.{i}"
            record_ids = evidence_map.get(field_path, [])
            if not record_ids:
                continue
            claim_tokens = _claim_tokens(text)
            record_tokens: set[str] = set()
            for rid in record_ids:
                rec = records_by_id.get(rid, {})
                for v in rec.get("payload", {}).values():
                    record_tokens |= _payload_tokens(str(v))
            overlap = len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
            result[field_path] = round(overlap, 4)

    return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def analyse_correlation(
    domain_result: dict,
    per_claim_overlap: dict[str, float],
    persona_name: str,
) -> dict:
    """
    For each flagged claim in the domain result that falls within SEMANTIC_FIELDS
    (i.e., has a semantic overlap score), classify it as flagged vs. non-flagged
    and compute mean overlaps.
    """
    flagged_paths = {fc["field"] for fc in domain_result["flagged_claims"]}
    scored_paths = set(per_claim_overlap.keys())

    flagged_scored: list[tuple[str, float]] = []
    non_flagged_scored: list[tuple[str, float]] = []

    for path, score in per_claim_overlap.items():
        if path in flagged_paths:
            flagged_scored.append((path, score))
        else:
            non_flagged_scored.append((path, score))

    # flagged claims in classifier fields that have NO semantic score
    # (decision_triggers, sample_quotes, vocabulary — no source_evidence entries)
    flagged_no_semantic = [
        fc for fc in domain_result["flagged_claims"] if fc["field"] not in scored_paths
    ]

    avg_flagged = (
        sum(s for _, s in flagged_scored) / len(flagged_scored) if flagged_scored else None
    )
    avg_non_flagged = (
        sum(s for _, s in non_flagged_scored) / len(non_flagged_scored)
        if non_flagged_scored
        else None
    )

    diff = None
    if avg_flagged is not None and avg_non_flagged is not None:
        diff = round(avg_non_flagged - avg_flagged, 4)

    # STRONG signal: flagged claims ≥ 0.1 lower avg semantic overlap than non-flagged
    signal = "NONE"
    if diff is not None:
        if diff >= 0.10:
            signal = "STRONG"
        elif diff >= 0.05:
            signal = "MODERATE"
        elif diff >= 0.0:
            signal = "WEAK"
        else:
            signal = "INVERSE"  # flagged claims actually have higher overlap

    return {
        "persona": persona_name,
        "flagged_with_semantic": [
            {"field_path": p, "overlap": s} for p, s in flagged_scored
        ],
        "non_flagged_with_semantic": [
            {"field_path": p, "overlap": s} for p, s in non_flagged_scored
        ],
        "flagged_no_semantic_score": [
            {"field": fc["field"], "text": fc["text"][:80]} for fc in flagged_no_semantic
        ],
        "avg_overlap_flagged": round(avg_flagged, 4) if avg_flagged is not None else None,
        "avg_overlap_non_flagged": round(avg_non_flagged, 4) if avg_non_flagged is not None else None,
        "overlap_gap": diff,
        "signal": signal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load personas and clusters
    persona_00_raw = json.loads((WORKTREE_3_22 / "persona_00.json").read_text())
    persona_01_raw = json.loads((WORKTREE_3_22 / "persona_01.json").read_text())
    cluster_00 = json.loads((WORKTREE_3_22 / "clusters" / "cluster_00.json").read_text())
    cluster_01 = json.loads((WORKTREE_3_22 / "clusters" / "cluster_01.json").read_text())

    # The persona data is nested under "persona" key
    persona_00 = persona_00_raw["persona"]
    persona_01 = persona_01_raw["persona"]

    personas = [
        (persona_00, cluster_00, "cluster_00 (Alex DevOps)"),
        (persona_01, cluster_01, "cluster_01 (Maya Designer)"),
    ]

    results_by_cluster: list[dict] = []

    for persona, cluster, label in personas:
        # 1. Domain-rules check (replicates exp-3.22)
        domain_result = check_domain_rules(persona)

        # 2. Full semantic proxy result
        semantic_result = semantic_groundedness_proxy(persona, cluster)

        # 3. Per-claim overlap for correlation analysis
        per_claim_overlap = compute_per_claim_overlap(persona, cluster)

        # 4. Correlation analysis
        correlation = analyse_correlation(domain_result, per_claim_overlap, label)

        results_by_cluster.append(
            {
                "cluster": label,
                "domain_rules": domain_result,
                "semantic_proxy": semantic_result,
                "per_claim_overlap": per_claim_overlap,
                "correlation": correlation,
            }
        )

    # Combined signal — rank: STRONG > MODERATE > WEAK > INVERSE > NONE
    _RANK = {"STRONG": 4, "MODERATE": 3, "WEAK": 2, "INVERSE": 1, "NONE": 0}
    signals = [r["correlation"]["signal"] for r in results_by_cluster]
    combined_signal = max(signals, key=lambda s: _RANK[s])

    output = {
        "experiment": "B3-rerun-3.22-semantic",
        "description": "Re-run exp-3.22 with semantic groundedness proxy to test if domain keyword classifier correlates with actual evidence weakness",
        "clusters": results_by_cluster,
        "combined_signal": combined_signal,
        "summary": {
            "cluster_00_flagging_rate": results_by_cluster[0]["domain_rules"]["flagging_rate"],
            "cluster_01_flagging_rate": results_by_cluster[1]["domain_rules"]["flagging_rate"],
            "cluster_00_semantic_score": results_by_cluster[0]["semantic_proxy"]["semantic_score"],
            "cluster_01_semantic_score": results_by_cluster[1]["semantic_proxy"]["semantic_score"],
            "cluster_00_overlap_gap": results_by_cluster[0]["correlation"]["overlap_gap"],
            "cluster_01_overlap_gap": results_by_cluster[1]["correlation"]["overlap_gap"],
            "cluster_00_signal": results_by_cluster[0]["correlation"]["signal"],
            "cluster_01_signal": results_by_cluster[1]["correlation"]["signal"],
        },
    }

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")

    # Pretty print summary
    print("\n=== SUMMARY ===")
    for r in results_by_cluster:
        corr = r["correlation"]
        dr = r["domain_rules"]
        sp = r["semantic_proxy"]
        print(f"\n{r['cluster']}")
        print(f"  Domain-rules: {dr['sensitive_claims_found']} flagged / {dr['total_claims_checked']} total (rate={dr['flagging_rate']})")
        print(f"  Semantic score (all claims): {sp['semantic_score']} | weak pairs: {sp['weak_count']}/{sp['claim_count']}")
        print(f"  Avg overlap — flagged: {corr['avg_overlap_flagged']} | non-flagged: {corr['avg_overlap_non_flagged']}")
        print(f"  Overlap gap (non-flagged minus flagged): {corr['overlap_gap']}")
        print(f"  Signal: {corr['signal']}")
    print(f"\nCombined signal: {combined_signal}")

    return output


if __name__ == "__main__":
    main()
