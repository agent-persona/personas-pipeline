"""Experiment 6.05: Stability across reruns."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from itertools import permutations
from statistics import mean


COMPARE_LIST_FIELDS = ("goals", "pains", "vocabulary", "sample_quotes")
COMPARE_TEXT_FIELDS = ("name", "summary")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def list_similarity(values_a: list[str], values_b: list[str]) -> float:
    words_a = set()
    words_b = set()
    for value in values_a:
        words_a |= _tokenize(value)
    for value in values_b:
        words_b |= _tokenize(value)
    return _jaccard(words_a, words_b)


def text_similarity(text_a: str, text_b: str) -> float:
    return _jaccard(_tokenize(text_a), _tokenize(text_b))


def persona_similarity(persona_a: dict, persona_b: dict) -> tuple[float, dict[str, float]]:
    per_field = {
        field: list_similarity(persona_a.get(field, []), persona_b.get(field, []))
        for field in COMPARE_LIST_FIELDS
    }
    for field in COMPARE_TEXT_FIELDS:
        per_field[field] = text_similarity(str(persona_a.get(field, "")), str(persona_b.get(field, "")))
    return mean(per_field.values()) if per_field else 0.0, per_field


@dataclass
class RunMatch:
    baseline_persona: str
    candidate_persona: str
    overall_similarity: float
    per_field_similarity: dict[str, float]


@dataclass
class RunComparison:
    run_id: str
    mean_similarity: float
    archetype_recurrence_rate: float
    matches: list[RunMatch]


def compare_run_to_baseline(
    baseline_personas: list[dict],
    candidate_personas: list[dict],
    run_id: str,
    recurrence_threshold: float = 0.45,
) -> RunComparison:
    if len(baseline_personas) != len(candidate_personas):
        raise ValueError("Expected the same persona count across reruns")

    best_total = -1.0
    best_matches: list[RunMatch] = []
    for ordering in permutations(candidate_personas):
        candidate_matches: list[RunMatch] = []
        total = 0.0
        for baseline_persona, candidate_persona in zip(baseline_personas, ordering):
            overall, per_field = persona_similarity(baseline_persona, candidate_persona)
            total += overall
            candidate_matches.append(
                RunMatch(
                    baseline_persona=baseline_persona.get("name", "unknown"),
                    candidate_persona=candidate_persona.get("name", "unknown"),
                    overall_similarity=overall,
                    per_field_similarity=per_field,
                )
            )
        if total > best_total:
            best_total = total
            best_matches = candidate_matches

    scores = [match.overall_similarity for match in best_matches]
    recurrence = (
        sum(1 for score in scores if score >= recurrence_threshold) / len(scores)
        if scores
        else 0.0
    )
    return RunComparison(
        run_id=run_id,
        mean_similarity=mean(scores) if scores else 0.0,
        archetype_recurrence_rate=recurrence,
        matches=best_matches,
    )


def comparisons_to_dict(comparisons: list[RunComparison]) -> list[dict]:
    return [asdict(comparison) for comparison in comparisons]
