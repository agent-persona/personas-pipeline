"""Shared metrics for all lab experiments.

Every experiment in `PRD_LAB_RESEARCH.md` declares a metric from this list.
Adding a new metric? Put it here so every problem space can pick it up
without re-implementing it.

Metric layer per the lab harness paragraph:
  schema validity, groundedness, distinctiveness, judge-rubric score,
  human Turing-pass rate, in-character drift, cost, latency.
"""

from __future__ import annotations

import math
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
    """
    if len(persona_embeddings) < 2:
        return 0.0

    distances = []
    for i in range(len(persona_embeddings)):
        for j in range(i + 1, len(persona_embeddings)):
            distances.append(_cosine_distance(persona_embeddings[i], persona_embeddings[j]))
    return sum(distances) / len(distances) if distances else 0.0


def vocabulary_jaccard(personas: Sequence[dict]) -> float:
    """Mean pairwise Jaccard similarity over persona vocabulary + sample_quotes.

    Used by experiment 6.16. Returns 0.0 (no overlap) to 1.0 (identical).
    Lower means MORE distinct.
    """
    word_sets = []
    for p in personas:
        words = set()
        for q in p.get("sample_quotes", []):
            if isinstance(q, str):
                words.update(q.lower().split())
        for v in p.get("vocabulary", []):
            if isinstance(v, str):
                words.update(v.lower().split())
        word_sets.append(words)

    if len(word_sets) < 2:
        return 0.0

    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            if word_sets[i] or word_sets[j]:
                jaccard_sim = len(word_sets[i] & word_sets[j]) / len(word_sets[i] | word_sets[j])
                overlaps.append(jaccard_sim)
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def gini_coefficient(scores: Sequence[float]) -> float:
    """Population Gini coefficient over persona quality scores.

    Used by experiment 6.19. Returns 0.0 (perfect equality, healthy diversity)
    to 1.0 (perfect inequality, dominant/marginal personas).
    """
    if not scores:
        return 0.0
    s = sorted(scores)
    n = len(s)
    total = sum(s)
    if total == 0:
        return 0.0
    numerator = sum((2 * i - n - 1) * x for i, x in enumerate(s, start=1))
    return numerator / (n * total)


def drift(turn_scores: Sequence[float], window: int = 3) -> dict:
    """Stylometric/character drift across a multi-turn conversation.

    Compares early window vs late window of per-turn scores.
    Returns dict with early_avg, late_avg, delta, and a `drifted` flag
    (True if delta > 0.5).
    """
    if len(turn_scores) < window * 2:
        return {"early_avg": 0.0, "late_avg": 0.0, "delta": 0.0, "drifted": False}

    early = sum(turn_scores[:window]) / window
    late = sum(turn_scores[-window:]) / window
    delta = early - late
    return {
        "early_avg": round(early, 3),
        "late_avg": round(late, 3),
        "delta": round(delta, 3),
        "drifted": delta > 0.5,
    }


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation coefficient.

    Used by experiment 5.01 (judge-human correlation) and 5.02 (cross-judge agreement).
    Returns -1.0 (perfect inverse) to 1.0 (perfect agreement).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d_sq) / (n * (n**2 - 1))


def cost_per_persona(total_cost_usd: float, n_personas: int) -> float:
    """Simple dollars-per-persona throughput metric."""
    if n_personas == 0:
        return 0.0
    return total_cost_usd / n_personas


def turing_pass_rate(human_labels: Sequence[bool]) -> float:
    """Fraction of conversations judged as human-rather-than-AI by human raters.

    Used by Pak's group 5 work. Expects bool list where True = "looks human".
    """
    if not human_labels:
        return 0.0
    return sum(1 for x in human_labels if x) / len(human_labels)


def turns_to_break(attack_logs: Sequence[Sequence[bool]]) -> float:
    """Mean number of turns until an adversarial prompt breaks character.

    `attack_logs` is a list of conversations; each conversation is a list of
    bools per turn (True = in character, False = broken). Returns the mean
    turn index of the first False per conversation.
    """
    if not attack_logs:
        return 0.0
    breaks = []
    for log in attack_logs:
        for i, in_char in enumerate(log):
            if not in_char:
                breaks.append(i + 1)
                break
        else:
            breaks.append(len(log) + 1)  # never broke
    return sum(breaks) / len(breaks)


# --- Internal helpers ---


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def _rank(values: Sequence[float]) -> list[float]:
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = float(rank)
    return ranks
