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
