"""Experiment 6.20: Persona deletion robustness.

Tests what happens when one persona is removed from a set and synthesis
is re-run. Measures whether the replacement persona absorbs the deleted
one's characteristics or whether coverage collapses.

Metrics:
  - **Replacement similarity**: how similar is the replacement to the
    deleted original? High = the pipeline reproduces the same archetype.
  - **Absorption rate**: what fraction of the deleted persona's key
    topics appear in the replacement?
  - **Differentiation**: does the replacement differ from the surviving
    personas, or does it collapse into a duplicate?
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.lower()))


def _persona_identity_words(persona: dict) -> set[str]:
    """Extract identity-defining words from a persona."""
    words: set[str] = set()
    for f in ("goals", "pains", "motivations", "objections", "vocabulary"):
        for item in persona.get(f, []):
            if isinstance(item, str):
                words |= _tokenize(item)
    for q in persona.get("sample_quotes", []):
        words |= _tokenize(q)
    words |= _tokenize(persona.get("summary", ""))
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "and", "but", "or", "not", "so", "if", "it", "i", "we", "you",
        "my", "our", "they", "them", "this", "that", "what", "how", "who",
    }
    return words - stopwords


@dataclass
class DeletionResult:
    """Result of deleting one persona and re-synthesizing."""
    deleted_name: str
    replacement_name: str
    surviving_names: list[str]
    # Similarity between deleted and replacement
    replacement_similarity: float  # Jaccard on identity words
    # Absorption: fraction of deleted persona's topics in replacement
    absorption_rate: float
    absorbed_words: int
    deleted_word_count: int
    # Differentiation from survivors
    survivor_similarity: float  # max Jaccard to any surviving persona
    is_duplicate_of_survivor: bool  # survivor_similarity > 0.5


@dataclass
class RobustnessReport:
    """Full deletion robustness analysis."""
    n_personas: int
    deletions: list[DeletionResult] = field(default_factory=list)
    avg_replacement_similarity: float = 0.0
    avg_absorption_rate: float = 0.0
    avg_survivor_similarity: float = 0.0
    duplicates_found: int = 0


def compute_deletion_result(
    deleted_persona: dict,
    replacement_persona: dict,
    surviving_personas: list[dict],
) -> DeletionResult:
    """Compare a replacement persona against the deleted original and survivors."""
    deleted_words = _persona_identity_words(deleted_persona)
    replacement_words = _persona_identity_words(replacement_persona)

    # Replacement similarity (Jaccard)
    if deleted_words or replacement_words:
        intersection = deleted_words & replacement_words
        union = deleted_words | replacement_words
        replacement_sim = len(intersection) / len(union) if union else 0.0
    else:
        replacement_sim = 0.0

    # Absorption rate: fraction of deleted words appearing in replacement
    if deleted_words:
        absorbed = deleted_words & replacement_words
        absorption = len(absorbed) / len(deleted_words)
    else:
        absorbed = set()
        absorption = 0.0

    # Differentiation from survivors
    max_survivor_sim = 0.0
    for survivor in surviving_personas:
        surv_words = _persona_identity_words(survivor)
        if replacement_words or surv_words:
            s_int = replacement_words & surv_words
            s_uni = replacement_words | surv_words
            sim = len(s_int) / len(s_uni) if s_uni else 0.0
            max_survivor_sim = max(max_survivor_sim, sim)

    return DeletionResult(
        deleted_name=deleted_persona.get("name", "?"),
        replacement_name=replacement_persona.get("name", "?"),
        surviving_names=[p.get("name", "?") for p in surviving_personas],
        replacement_similarity=replacement_sim,
        absorption_rate=absorption,
        absorbed_words=len(absorbed) if deleted_words else 0,
        deleted_word_count=len(deleted_words),
        survivor_similarity=max_survivor_sim,
        is_duplicate_of_survivor=max_survivor_sim > 0.5,
    )


def build_robustness_report(
    deletions: list[DeletionResult],
    n_personas: int,
) -> RobustnessReport:
    """Aggregate deletion results into a robustness report."""
    if not deletions:
        return RobustnessReport(n_personas=n_personas)

    return RobustnessReport(
        n_personas=n_personas,
        deletions=deletions,
        avg_replacement_similarity=statistics.mean([d.replacement_similarity for d in deletions]),
        avg_absorption_rate=statistics.mean([d.absorption_rate for d in deletions]),
        avg_survivor_similarity=statistics.mean([d.survivor_similarity for d in deletions]),
        duplicates_found=sum(1 for d in deletions if d.is_duplicate_of_survivor),
    )
