"""Experiment 6.02: Coverage gap analysis.

For each source record, computes max similarity to any persona in the set.
Records below a similarity threshold are "uncovered" — the persona set
doesn't represent them. The coverage fraction is the % of records that
align with at least one persona.

Similarity is computed by matching record signals (behaviors, pages, payload
keywords) against persona fields (goals, pains, vocabulary, firmographics).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field


# ── Tokenization ────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.lower()))


def _extract_record_words(record: dict) -> set[str]:
    """Extract all signal words from a source record."""
    words: set[str] = set()
    for b in record.get("behaviors", []):
        words |= _tokenize(b)
    for p in record.get("pages", []):
        words |= _tokenize(p)
    payload = record.get("payload", {})
    for k, v in payload.items():
        words |= _tokenize(str(k))
        words |= _tokenize(str(v))
    if record.get("source"):
        words |= _tokenize(record["source"])
    return words


def _extract_persona_words(persona: dict) -> set[str]:
    """Extract all identity words from a persona."""
    words: set[str] = set()
    for f in ("goals", "pains", "motivations", "objections", "vocabulary",
              "channels", "decision_triggers"):
        for item in persona.get(f, []):
            if isinstance(item, str):
                words |= _tokenize(item)
    for q in persona.get("sample_quotes", []):
        words |= _tokenize(q)
    words |= _tokenize(persona.get("summary", ""))
    firmo = persona.get("firmographics", {})
    for v in firmo.values():
        if isinstance(v, str):
            words |= _tokenize(v)
        elif isinstance(v, list):
            for item in v:
                words |= _tokenize(str(item))
    # Remove stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "and", "but", "or", "not", "so", "if", "it", "i", "we", "you",
        "my", "our", "they", "them", "this", "that", "what", "how", "who",
    }
    return words - stopwords


# ── Similarity ──────────────────────────────────────────────────────

def record_persona_similarity(
    record_words: set[str],
    persona_words: set[str],
) -> float:
    """Jaccard similarity between a record's signals and a persona's identity."""
    if not record_words or not persona_words:
        return 0.0
    intersection = record_words & persona_words
    union = record_words | persona_words
    return len(intersection) / len(union) if union else 0.0


# ── Coverage analysis ───────────────────────────────────────────────

@dataclass
class RecordCoverage:
    """Coverage result for a single source record."""
    record_id: str
    max_similarity: float
    best_persona: str
    is_covered: bool
    record_words: int = 0


@dataclass
class CoverageReport:
    """Full coverage analysis for a persona set against source records."""
    total_records: int
    covered_records: int
    uncovered_records: int
    coverage_fraction: float  # covered / total
    avg_similarity: float
    median_similarity: float
    min_similarity: float
    max_similarity: float
    per_record: list[RecordCoverage] = field(default_factory=list)
    uncovered_details: list[RecordCoverage] = field(default_factory=list)
    per_persona_coverage: dict[str, int] = field(default_factory=dict)


def compute_coverage(
    records: list[dict],
    personas: list[dict],
    threshold: float = 0.05,
) -> CoverageReport:
    """Compute coverage of source records by persona set.

    For each record, find the persona with highest similarity. If the
    similarity exceeds the threshold, the record is "covered."

    Args:
        records: list of source record dicts (with behaviors, pages, payload).
        personas: list of persona dicts.
        threshold: minimum similarity to count as covered (default 0.05).
    """
    # Pre-compute persona word sets
    persona_word_sets: list[tuple[str, set[str]]] = []
    for p in personas:
        name = p.get("name", "?")
        words = _extract_persona_words(p)
        persona_word_sets.append((name, words))

    per_record: list[RecordCoverage] = []
    per_persona_count: dict[str, int] = {name: 0 for name, _ in persona_word_sets}
    similarities: list[float] = []

    for rec in records:
        rec_words = _extract_record_words(rec)
        record_id = rec.get("record_id", "?")

        best_sim = 0.0
        best_persona = ""
        for name, p_words in persona_word_sets:
            sim = record_persona_similarity(rec_words, p_words)
            if sim > best_sim:
                best_sim = sim
                best_persona = name

        is_covered = best_sim >= threshold
        rc = RecordCoverage(
            record_id=record_id,
            max_similarity=best_sim,
            best_persona=best_persona,
            is_covered=is_covered,
            record_words=len(rec_words),
        )
        per_record.append(rc)
        similarities.append(best_sim)

        if is_covered and best_persona:
            per_persona_count[best_persona] = per_persona_count.get(best_persona, 0) + 1

    total = len(records)
    covered = sum(1 for r in per_record if r.is_covered)
    uncovered = [r for r in per_record if not r.is_covered]

    return CoverageReport(
        total_records=total,
        covered_records=covered,
        uncovered_records=total - covered,
        coverage_fraction=covered / total if total > 0 else 0.0,
        avg_similarity=statistics.mean(similarities) if similarities else 0.0,
        median_similarity=statistics.median(similarities) if similarities else 0.0,
        min_similarity=min(similarities) if similarities else 0.0,
        max_similarity=max(similarities) if similarities else 0.0,
        per_record=per_record,
        uncovered_details=uncovered,
        per_persona_coverage=per_persona_count,
    )
