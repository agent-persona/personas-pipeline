"""Experiment 4.03: Drift curve measurement.

Measures stylometric and content drift at conversation checkpoints.
Computes the "half-life" of in-character behavior — the turn count at
which the persona's vocabulary overlap drops below 50% of the turn-1
baseline.

Drift is measured by:
  - **Vocabulary overlap**: fraction of persona vocabulary/quote words
    appearing in the twin's response.
  - **Content consistency**: fraction of responses that reference persona-
    specific topics (goals, pains, industry terms).
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field


# ── Drift scoring ───────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", text.lower()))


def extract_persona_words(persona: dict) -> set[str]:
    """Extract the persona's signature word set from vocabulary, quotes,
    goals, and pains."""
    words: set[str] = set()
    for term in persona.get("vocabulary", []):
        words |= _tokenize(term)
    for quote in persona.get("sample_quotes", []):
        words |= _tokenize(quote)
    for goal in persona.get("goals", []):
        words |= _tokenize(goal)
    for pain in persona.get("pains", []):
        words |= _tokenize(pain)
    # Remove ultra-common words that don't signal persona identity
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "like",
        "through", "after", "over", "between", "out", "up", "down", "off",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "it", "its", "i", "me", "my", "we", "our",
        "you", "your", "he", "she", "they", "them", "this", "that", "these",
        "those", "what", "which", "who", "whom", "how", "when", "where",
        "why", "if", "then", "else", "also", "much", "many",
    }
    return words - stopwords


def score_turn(response: str, persona_words: set[str]) -> float:
    """Vocabulary overlap between a single response and persona words.

    Returns fraction of persona_words that appear in the response.
    """
    if not persona_words:
        return 0.0
    response_words = _tokenize(response)
    overlap = persona_words & response_words
    return len(overlap) / len(persona_words)


# ── Checkpoint data ─────────────────────────────────────────────────

@dataclass
class CheckpointScore:
    """Drift score at one conversation checkpoint."""
    turn: int
    vocab_overlap: float
    response_snippet: str = ""


@dataclass
class DriftCurve:
    """Full drift curve for one persona conversation."""
    persona_name: str
    persona_word_count: int
    checkpoints: list[CheckpointScore] = field(default_factory=list)
    half_life: int = -1  # turn where overlap drops below 50% of baseline; -1 = never
    baseline_overlap: float = 0.0  # overlap at turn 1
    final_overlap: float = 0.0    # overlap at last checkpoint
    decay_rate: float = 0.0       # (baseline - final) / n_turns


def compute_drift_curve(
    responses: list[str],
    persona_words: set[str],
    checkpoints: list[int],
) -> DriftCurve:
    """Score responses at checkpoint turns and compute the drift curve."""
    curve = DriftCurve(
        persona_name="",
        persona_word_count=len(persona_words),
    )

    all_scores: list[tuple[int, float, str]] = []
    for turn_idx, response in enumerate(responses):
        turn_num = turn_idx + 1  # 1-indexed
        if turn_num in checkpoints:
            overlap = score_turn(response, persona_words)
            all_scores.append((turn_num, overlap, response[:80]))

    for turn, overlap, snippet in all_scores:
        curve.checkpoints.append(CheckpointScore(
            turn=turn,
            vocab_overlap=overlap,
            response_snippet=snippet,
        ))

    if curve.checkpoints:
        curve.baseline_overlap = curve.checkpoints[0].vocab_overlap
        curve.final_overlap = curve.checkpoints[-1].vocab_overlap

        # Decay rate: normalized drop per turn
        n_turns = curve.checkpoints[-1].turn - curve.checkpoints[0].turn
        if n_turns > 0 and curve.baseline_overlap > 0:
            curve.decay_rate = (curve.baseline_overlap - curve.final_overlap) / n_turns

        # Half-life: first turn where overlap < 50% of baseline
        threshold = curve.baseline_overlap * 0.5
        for cp in curve.checkpoints:
            if cp.vocab_overlap < threshold and curve.baseline_overlap > 0:
                curve.half_life = cp.turn
                break

    return curve


def compute_half_life(curve: DriftCurve) -> int:
    """Extract the half-life from a drift curve. -1 = never reached."""
    return curve.half_life
