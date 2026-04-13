"""Experiment 6.16: Vocabulary uniqueness — Jensen-Shannon divergence.

Computes linguistic distinctiveness across personas by measuring
Jensen-Shannon divergence between every pair of vocabulary[] lists
in a tenant. Higher mean JSD indicates more linguistically diverse personas.

JSD is symmetric, bounded [0, 1] (when using base-2 log), and well-defined
even when one distribution assigns zero probability to a term.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Sequence


def _word_distribution(vocabulary: list[str]) -> Counter:
    """Build a normalized word frequency distribution from a vocabulary list.

    Each term in the vocabulary[] list is treated as one observation.
    Terms are lowercased and stripped for consistency.
    """
    return Counter(w.strip().lower() for w in vocabulary if w.strip())


def _kl_divergence(p: dict[str, float], q: dict[str, float], vocab: set[str]) -> float:
    """Kullback-Leibler divergence D_KL(P || Q) over a shared vocabulary.

    Uses base-2 logarithm so values are in bits.
    Assumes q[w] > 0 for all w where p[w] > 0 (guaranteed by the mixture M).
    """
    d = 0.0
    for w in vocab:
        pw = p.get(w, 0.0)
        qw = q.get(w, 0.0)
        if pw > 0 and qw > 0:
            d += pw * math.log2(pw / qw)
    return d


def js_divergence(vocab_a: list[str], vocab_b: list[str]) -> float:
    """Jensen-Shannon divergence between two vocabulary lists.

    Returns a value in [0, 1] (base-2 log). 0 = identical distributions,
    1 = completely non-overlapping.
    """
    if not vocab_a and not vocab_b:
        return 0.0
    if not vocab_a or not vocab_b:
        return 1.0

    counts_a = _word_distribution(vocab_a)
    counts_b = _word_distribution(vocab_b)

    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    # Normalized probability distributions
    p = {w: c / total_a for w, c in counts_a.items()}
    q = {w: c / total_b for w, c in counts_b.items()}

    # Shared vocabulary
    all_words = set(p.keys()) | set(q.keys())

    # Mixture distribution M = 0.5 * (P + Q)
    m = {w: 0.5 * (p.get(w, 0.0) + q.get(w, 0.0)) for w in all_words}

    jsd = 0.5 * _kl_divergence(p, m, all_words) + 0.5 * _kl_divergence(q, m, all_words)

    # Clamp to [0, 1] for floating-point safety
    return max(0.0, min(1.0, jsd))


def mean_pairwise_js_divergence(persona_vocabularies: Sequence[list[str]]) -> float:
    """Mean pairwise Jensen-Shannon divergence across all persona vocabulary lists.

    This is the primary metric for experiment 6.16.

    Args:
        persona_vocabularies: List of vocabulary[] lists, one per persona.

    Returns:
        Mean pairwise JSD in [0, 1]. Higher = more linguistically distinct.
        Returns 0.0 if fewer than 2 personas.
    """
    if len(persona_vocabularies) < 2:
        return 0.0

    pair_scores = []
    for i, j in combinations(range(len(persona_vocabularies)), 2):
        jsd = js_divergence(persona_vocabularies[i], persona_vocabularies[j])
        pair_scores.append(jsd)

    return sum(pair_scores) / len(pair_scores) if pair_scores else 0.0


def pairwise_js_matrix(
    persona_vocabularies: Sequence[list[str]],
    persona_names: Sequence[str] | None = None,
) -> dict:
    """Full pairwise JSD matrix with per-pair breakdown.

    Returns a dict with:
      - mean_jsd: The primary metric
      - min_jsd / max_jsd: Range of pairwise scores
      - n_personas: Number of personas compared
      - n_pairs: Number of unique pairs
      - pairs: List of {persona_a, persona_b, jsd} dicts
    """
    n = len(persona_vocabularies)
    names = list(persona_names) if persona_names else [f"persona_{i}" for i in range(n)]

    if n < 2:
        return {
            "mean_jsd": 0.0,
            "min_jsd": 0.0,
            "max_jsd": 0.0,
            "n_personas": n,
            "n_pairs": 0,
            "pairs": [],
        }

    pairs = []
    for i, j in combinations(range(n), 2):
        jsd = js_divergence(persona_vocabularies[i], persona_vocabularies[j])
        pairs.append({
            "persona_a": names[i],
            "persona_b": names[j],
            "jsd": round(jsd, 6),
        })

    scores = [p["jsd"] for p in pairs]
    return {
        "mean_jsd": round(sum(scores) / len(scores), 6),
        "min_jsd": round(min(scores), 6),
        "max_jsd": round(max(scores), 6),
        "n_personas": n,
        "n_pairs": len(pairs),
        "pairs": pairs,
    }


def vocab_overlap_stats(persona_vocabularies: Sequence[list[str]]) -> dict:
    """Complementary statistics: raw vocabulary overlap between personas.

    Useful alongside JSD for understanding whether personas share terms
    or are using completely different language.
    """
    if len(persona_vocabularies) < 2:
        return {"mean_jaccard": 0.0, "unique_ratio": 1.0, "total_unique_terms": 0}

    # Per-persona word sets (lowercased)
    word_sets = [set(w.strip().lower() for w in v if w.strip()) for v in persona_vocabularies]

    # Jaccard similarity between all pairs
    jaccard_scores = []
    for i, j in combinations(range(len(word_sets)), 2):
        union = word_sets[i] | word_sets[j]
        intersection = word_sets[i] & word_sets[j]
        jac = len(intersection) / len(union) if union else 1.0
        jaccard_scores.append(jac)

    # How many terms are unique to exactly one persona
    all_terms = set()
    for ws in word_sets:
        all_terms |= ws
    shared = set()
    for i, j in combinations(range(len(word_sets)), 2):
        shared |= (word_sets[i] & word_sets[j])
    unique_only = all_terms - shared

    return {
        "mean_jaccard": round(sum(jaccard_scores) / len(jaccard_scores), 6) if jaccard_scores else 0.0,
        "unique_ratio": round(len(unique_only) / len(all_terms), 6) if all_terms else 1.0,
        "total_unique_terms": len(all_terms),
        "terms_shared_across_personas": len(shared),
        "terms_unique_to_one_persona": len(unique_only),
    }
