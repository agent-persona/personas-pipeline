"""Shared metrics for all lab experiments.

Every experiment in `PRD_LAB_RESEARCH.md` declares a metric from this list.
Adding a new metric? Put it here so every problem space can pick it up
without re-implementing it.

This module is a skeleton — most functions return a placeholder. Researcher
#5 (problem space 5) owns the real implementations. The signatures below
are stable: other researchers can import them today and will get real
numbers once the bodies land.
"""

from __future__ import annotations

import re
from typing import Sequence


def schema_validity(persona_dicts: Sequence[dict], schema_cls) -> float:
    """Fraction of persona dicts that validate against `schema_cls`.

    Cheap, deterministic, no LLM calls.
    """
    from pydantic import ValidationError

    if not persona_dicts:
        return 1.0
    ok = 0
    for p in persona_dicts:
        try:
            schema_cls.model_validate(p)
            ok += 1
        except ValidationError:
            pass
    return ok / len(persona_dicts)


def groundedness_rate(reports: Sequence) -> float:
    """Mean groundedness score across a batch of `GroundednessReport`s.

    Expects the `GroundednessReport` type from
    `synthesis.engine.groundedness` (duck-typed: anything with a `.score`
    attribute works).
    """
    if not reports:
        return 1.0
    return sum(r.score for r in reports) / len(reports)


def distinctiveness(persona_embeddings: Sequence[Sequence[float]]) -> float:
    """Mean pairwise cosine distance across a set of persona embeddings.

    1.0 = maximally distinct, 0.0 = identical. Used by space 6 (distinctiveness
    floor) and space 4 (drift: did turn N drift toward turn 1 or away from it).

    TODO(space-5): implement. Today returns NaN.
    """
    return float("nan")


def cost_per_persona(total_cost_usd: float, n_personas: int) -> float:
    """Simple dollars-per-persona throughput metric."""
    if n_personas == 0:
        return 0.0
    return total_cost_usd / n_personas


# TODO(space-5): add implementations for the remaining shared metrics.
#
#   - turing_pass_rate(labels)  : human-task-worker identification rate
#   - drift(turn_embeddings)    : stylometric drift turn N vs turn 1
#   - turns_to_break(attack_log): mean turns until an adversarial prompt wins
#   - judge_rubric_score(j, p)  : Opus-as-judge per-dimension rubric score
#   - human_correlation(j, h)   : Spearman between judge and human labels


_STOPWORDS = {
    "the","a","an","and","or","of","in","for","to","with","that",
    "their","our","is","are","was","were","be","been","have","has",
    "this","it","its","at","by","from","on","as","not","but","so",
    "they","we","you","he","she","also","which","what","when","how",
    "can","will","would","could","should","may","might","do","did",
    "use","used","using","get","got","need","needs","work","works",
}

def _claim_tokens(text: str) -> set[str]:
    # Split on whitespace first, then strip punctuation
    return {
        w.lower().strip(".,;:'\"()")
        for w in text.split()
        if len(w) > 3 and w.lower().strip(".,;:'\"()") not in _STOPWORDS
    }

def _payload_tokens(text: str) -> set[str]:
    # Also split on non-alphanumeric separators to handle webhook_config, /settings/webhooks, etc.
    parts = re.split(r"[^a-zA-Z0-9]+", text)
    return {
        p.lower()
        for p in parts
        if len(p) > 3 and p.lower() not in _STOPWORDS
    }

def semantic_groundedness_proxy(persona: dict, cluster: dict) -> dict:
    """
    Vocabulary-overlap semantic groundedness check.

    For each claim in goals/pains/motivations/objections, computes token
    overlap with the payload of its cited records. Low overlap indicates
    the claim may not be supported by the cited evidence.

    Args:
        persona: PersonaV1 dict (must have source_evidence list)
        cluster: ClusterData dict (must have sample_records list)

    Returns:
        {
          "semantic_score": float,    # mean overlap [0-1] across all claim-evidence pairs
          "weak_pairs": list[dict],   # pairs with overlap < 0.1
          "claim_count": int,
          "weak_count": int,
          "coverage": float,          # fraction of claims that had any cited evidence
        }
    """
    records_by_id = {r["record_id"]: r for r in cluster.get("sample_records", [])}
    evidence_map = {
        e["field_path"]: e["record_ids"]
        for e in persona.get("source_evidence", [])
    }
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
                payload = rec.get("payload", {})
                for v in payload.values():
                    record_tokens |= _payload_tokens(str(v))
            overlap = (
                len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
            )
            overlaps.append(overlap)
            if overlap < 0.1:
                weak.append({
                    "field_path": field_path,
                    "claim": text[:100],
                    "overlap": round(overlap, 4),
                    "claim_tokens": sorted(claim_tokens)[:10],
                    "matched_tokens": sorted(claim_tokens & record_tokens),
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
