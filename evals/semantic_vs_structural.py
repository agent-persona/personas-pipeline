"""Experiment 3.25 — Semantic Groundedness Validation

Compare structural checker (check_groundedness) vs semantic proxy on the same
personas. Measure divergence rate: claims that PASS structural but FAIL semantic.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synthesis"))

from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# ---------------------------------------------------------------------------
# Semantic proxy (inline — not imported from another branch)
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
                weak.append({
                    "field_path": field_path,
                    "claim": text[:100],
                    "overlap": round(overlap, 4),
                })
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
# Per-claim divergence analysis
# ---------------------------------------------------------------------------

def analyze_per_claim(persona_dict: dict, cluster_dict: dict, structural_report) -> list[dict]:
    """For each evidence-backed claim, record structural pass/fail and semantic pass/fail."""
    FIELDS = ["goals", "pains", "motivations", "objections"]

    valid_ids = {r["record_id"] for r in cluster_dict.get("sample_records", [])}
    source_evidence = persona_dict.get("source_evidence", [])

    # Mirror groundedness.py: a field_path is valid only if all its record_ids exist
    valid_evidence_paths: set[str] = {
        ev["field_path"]
        for ev in source_evidence
        if all(rid in valid_ids for rid in ev["record_ids"])
    }

    # Build semantic overlap map
    records_by_id = {r["record_id"]: r for r in cluster_dict.get("sample_records", [])}
    evidence_map = {e["field_path"]: e["record_ids"] for e in source_evidence}

    rows = []
    for field in FIELDS:
        for idx, item in enumerate(persona_dict.get(field, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            field_path = f"{field}.{idx}"
            record_ids = evidence_map.get(field_path, [])
            if not record_ids:
                continue  # no evidence entry — not part of divergence analysis

            # Structural pass: field_path is in valid_evidence_paths
            structural_pass = field_path in valid_evidence_paths

            # Semantic overlap
            claim_tokens = _claim_tokens(text)
            record_tokens: set[str] = set()
            for rid in record_ids:
                rec = records_by_id.get(rid, {})
                for v in rec.get("payload", {}).values():
                    record_tokens |= _payload_tokens(str(v))
            overlap = len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
            semantic_pass = overlap >= 0.1

            rows.append({
                "field_path": field_path,
                "claim": text[:120],
                "record_ids": record_ids,
                "structural_pass": structural_pass,
                "semantic_pass": semantic_pass,
                "overlap": round(overlap, 4),
                "diverges": structural_pass and not semantic_pass,
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_pair(persona_path: Path, cluster_path: Path):
    persona_doc = json.loads(persona_path.read_text())
    cluster_doc = json.loads(cluster_path.read_text())
    persona_dict = persona_doc["persona"]
    cluster_dict = cluster_doc
    persona_obj = PersonaV1.model_validate(persona_dict)
    cluster_obj = ClusterData.model_validate(cluster_dict)
    return persona_dict, cluster_dict, persona_obj, cluster_obj


def run():
    # Data files: prefer local output/, fall back to main repo output/
    # Worktree path: <repo>/.worktrees/<branch>/  -> ROOT.parents[1] is the main repo
    data_root = ROOT / "output"
    if not (data_root / "persona_00.json").exists():
        data_root = ROOT.parents[1] / "output"
    pairs = [
        (data_root / "persona_00.json", data_root / "clusters" / "cluster_00.json", "Alex (DevOps)"),
        (data_root / "persona_01.json", data_root / "clusters" / "cluster_01.json", "Maya (Designer)"),
    ]

    results = []

    for persona_path, cluster_path, label in pairs:
        persona_dict, cluster_dict, persona_obj, cluster_obj = load_pair(persona_path, cluster_path)

        # Structural check
        struct_report = check_groundedness(persona_obj, cluster_obj)

        # Semantic proxy
        sem_result = semantic_groundedness_proxy(persona_dict, cluster_dict)

        # Per-claim divergence
        claim_rows = analyze_per_claim(persona_dict, cluster_dict, struct_report)

        total_claims = len(claim_rows)
        diverging = [r for r in claim_rows if r["diverges"]]
        agreeing = [r for r in claim_rows if r["structural_pass"] == r["semantic_pass"]]
        divergence_rate = len(diverging) / max(total_claims, 1)
        agreement_rate = len(agreeing) / max(total_claims, 1)

        result = {
            "persona": label,
            "structural_score": struct_report.score,
            "structural_violations": struct_report.violations,
            "semantic_score": sem_result["semantic_score"],
            "claim_count": total_claims,
            "weak_count": sem_result["weak_count"],
            "weak_pairs": sem_result["weak_pairs"],
            "divergence_rate": round(divergence_rate, 4),
            "agreement_rate": round(agreement_rate, 4),
            "diverging_claims": diverging,
            "per_claim_table": claim_rows,
        }
        results.append(result)

        print(f"\n{'='*60}")
        print(f"Persona: {label}")
        print(f"  Structural score : {struct_report.score:.4f}  (violations: {len(struct_report.violations)})")
        print(f"  Semantic score   : {sem_result['semantic_score']:.4f}")
        print(f"  Claims analyzed  : {total_claims}")
        print(f"  Weak (overlap<0.1): {sem_result['weak_count']}")
        print(f"  Divergence rate  : {divergence_rate:.1%}  (struct PASS, sem FAIL)")
        print(f"  Agreement rate   : {agreement_rate:.1%}")
        print(f"\n  Per-claim table:")
        print(f"  {'field_path':<22} {'struct':>7} {'sem':>5} {'overlap':>8}  diverges")
        for row in claim_rows:
            print(
                f"  {row['field_path']:<22} {'PASS' if row['structural_pass'] else 'FAIL':>7} "
                f"{'PASS' if row['semantic_pass'] else 'FAIL':>5} {row['overlap']:>8.4f}  "
                f"{'*** DIVERGES' if row['diverges'] else ''}"
            )
        if diverging:
            print(f"\n  Most revealing weak pairs (struct PASS, sem FAIL):")
            for d in diverging:
                print(f"    {d['field_path']}: \"{d['claim'][:80]}\"")
                print(f"      overlap={d['overlap']}, records={d['record_ids']}")

    # Aggregate
    total_claims_all = sum(r["claim_count"] for r in results)
    total_diverging = sum(len(r["diverging_claims"]) for r in results)
    total_agreeing = sum(
        sum(1 for row in r["per_claim_table"] if row["structural_pass"] == row["semantic_pass"])
        for r in results
    )
    overall_divergence = total_diverging / max(total_claims_all, 1)
    overall_agreement = total_agreeing / max(total_claims_all, 1)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ACROSS BOTH PERSONAS")
    print(f"  Total claims : {total_claims_all}")
    print(f"  Total diverging (struct PASS, sem FAIL): {total_diverging} ({overall_divergence:.1%})")
    print(f"  Overall agreement rate: {overall_agreement:.1%}")
    signal = "STRONG" if overall_divergence > 0.50 else ("MODERATE" if overall_divergence > 0.25 else "WEAK")
    print(f"  Signal: {signal}")

    output = {
        "experiment": "exp-3.25-semantic-groundedness-validation",
        "hypothesis": "Structural checker passes >90% of claims that semantic proxy flags as weakly grounded",
        "personas": results,
        "aggregate": {
            "total_claims": total_claims_all,
            "total_diverging": total_diverging,
            "overall_divergence_rate": round(overall_divergence, 4),
            "overall_agreement_rate": round(overall_agreement, 4),
            "signal": signal,
        },
    }

    out_dir = ROOT / "output" / "experiments" / "exp-3.25-semantic-groundedness-validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {results_path}")
    return output


if __name__ == "__main__":
    run()
