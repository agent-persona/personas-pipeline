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

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .judges import LLMJudge


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


def stylometric_cosine(
    replies: Sequence[str],
    reference_texts: Sequence[str],
) -> float:
    """Character-n-gram TF-IDF cosine between a reply set and a reference corpus.

    Used by experiment 1.3 (vocabulary anchoring) to measure how close
    a twin's replies sit to the original Intercom verbatim quotes the
    persona was synthesized from. Character-level n-grams are chosen
    deliberately: they pick up punctuation habits, contractions, and
    slang fragments that word-level TF-IDF would smooth over.

    Each list is concatenated into a single "document" so the result
    is the cosine between two TF-IDF vectors. Returns a scalar in
    [0, 1]; 1.0 = identical character-n-gram distributions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not replies or not reference_texts:
        return 0.0

    replies_doc = " ".join(replies)
    reference_doc = " ".join(reference_texts)
    if not replies_doc.strip() or not reference_doc.strip():
        return 0.0

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    matrix = vectorizer.fit_transform([replies_doc, reference_doc])
    sim = cosine_similarity(matrix[0], matrix[1])
    return float(sim[0][0])


async def pairing_accuracy(
    pairs: Sequence[tuple[str, str, str]],
    judge: "LLMJudge",
) -> float:
    """Fraction of (reply_a, reply_b, label) triples the judge labels correctly.

    `label` must be `"same"` or `"different"`. The judge's
    `same_speaker(a, b)` is called once per pair and compared to the
    gold label. Used by experiment 1.3 to ask: can a judge tell the
    control twin and the ablated twin apart?

    Self-preference caveat: if the judge model matches the twin model
    (both Haiku in experiment 1.3), "same" calls are inflated. Report
    this number alongside stylometric_cosine rather than alone.
    """
    if not pairs:
        return 0.0
    correct = 0
    for reply_a, reply_b, label in pairs:
        if label not in ("same", "different"):
            raise ValueError(f"label must be 'same' or 'different', got {label!r}")
        predicted_same = await judge.same_speaker(reply_a, reply_b)
        gold_same = label == "same"
        if predicted_same == gold_same:
            correct += 1
    return correct / len(pairs)


# TODO(space-5): add implementations for the remaining shared metrics.
#
#   - turing_pass_rate(labels)  : human-task-worker identification rate
#   - drift(turn_embeddings)    : stylometric drift turn N vs turn 1
#   - turns_to_break(attack_log): mean turns until an adversarial prompt wins
#   - judge_rubric_score(j, p)  : Opus-as-judge per-dimension rubric score
#   - human_correlation(j, h)   : Spearman between judge and human labels
