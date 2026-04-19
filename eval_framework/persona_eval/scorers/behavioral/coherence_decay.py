"""D34 Multi-Turn Coherence Decay — sliding-window consistency measurement.

Trustworthiness: MEDIUM-HIGH (embedding centroid consistency is objective).
Method: Measure persona consistency in windows across long conversations.
Detect downward trend (decay) and critical turn (first window below threshold).
Expects source_context.extra_data["conversation_turns"]: list of response strings.
"""

from __future__ import annotations

from persona_eval.embeddings import Embedder
from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

WINDOW_THRESHOLD = 0.7


class CoherenceDecayScorer(BaseScorer):
    """Evaluates multi-turn coherence decay across long conversations."""

    dimension_id = "D34"
    dimension_name = "Multi-Turn Coherence Decay"
    tier = 5
    requires_set = False

    def __init__(self, window_size: int = 10) -> None:
        self._embedder: Embedder | None = None
        self.window_size = window_size

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        turns: list[str] = source_context.extra_data.get("conversation_turns", [])
        if len(turns) < self.window_size * 2:
            return self._result(
                persona, passed=True, score=1.0,
                details={
                    "skipped": True,
                    "reason": f"Need >= {self.window_size * 2} turns, got {len(turns)}",
                },
            )

        embedder = self._get_embedder()
        vecs = embedder.embed_batch(turns)
        dim = len(vecs[0])

        # Sliding window consistency
        window_scores: list[dict] = []
        step = max(1, self.window_size // 2)
        for i in range(0, len(vecs) - self.window_size + 1, step):
            window = vecs[i : i + self.window_size]
            centroid = [sum(v[d] for v in window) / len(window) for d in range(dim)]
            sims = [Embedder.vector_similarity(v, centroid) for v in window]
            mean_sim = sum(sims) / len(sims)
            window_scores.append({
                "start_turn": i,
                "end_turn": i + self.window_size,
                "mean_similarity": round(mean_sim, 4),
            })

        if len(window_scores) < 2:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Not enough windows"},
            )

        scores = [w["mean_similarity"] for w in window_scores]
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        decay = first_half - second_half

        # Find critical turn
        critical_turn = None
        for w in window_scores:
            if w["mean_similarity"] < WINDOW_THRESHOLD:
                critical_turn = w["start_turn"]
                break

        overall_score = sum(scores) / len(scores)
        passed = decay < 0.1 and overall_score >= 0.6

        return self._result(
            persona, passed=passed, score=round(min(1.0, overall_score), 4),
            details={
                "decay_magnitude": round(decay, 4),
                "first_half_avg": round(first_half, 4),
                "second_half_avg": round(second_half, 4),
                "critical_turn": critical_turn,
                "window_scores": window_scores,
                "turn_count": len(turns),
            },
        )
