"""D43 Source Data Fidelity — information retention from source to persona.

Trustworthiness: MEDIUM-HIGH (depends on quality of fact extraction).
Method: Extract facts from source data, check which survive in persona
using embedding retrieval (query = source fact, corpus = persona text).
Expects source_context.extra_data:
    "source_facts": list[str] (key facts from source data)
    "persona_text": str (the generated persona text to check against)
"""

from __future__ import annotations

import re

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Similarity threshold for considering a fact "retained" (tuned for MiniLM-L6-v2)
RETENTION_THRESHOLD = 0.35
# Pass if retention rate >= this
PASS_THRESHOLD = 0.5


class SourceDataFidelityScorer(BaseScorer):
    """Evaluates how much source data signal survives in the generated persona."""

    dimension_id = "D43"
    dimension_name = "Source Data Fidelity"
    tier = 7
    requires_set = False

    def __init__(self) -> None:
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from evaluation.testing.embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        source_facts: list[str] = source_context.extra_data.get("source_facts", [])
        persona_text: str = source_context.extra_data.get("persona_text", "")

        if not source_facts or not persona_text:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Missing source_facts or persona_text"},
            )

        embedder = self._get_embedder()

        # Split persona text into sentences (handles ., !, ?)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', persona_text) if s.strip()]
        if not sentences:
            sentences = [persona_text]

        # Batch-embed for efficiency: 2 calls instead of F*S
        from evaluation.testing.embeddings import Embedder
        fact_vecs = embedder.embed_batch(source_facts)
        sent_vecs = embedder.embed_batch(sentences)

        # For each source fact, check if it's retained in the persona
        retained = 0
        fact_scores: list[dict] = []
        for i, fact in enumerate(source_facts):
            best_sim = max(Embedder.vector_similarity(fact_vecs[i], sv) for sv in sent_vecs)
            is_retained = best_sim >= RETENTION_THRESHOLD
            if is_retained:
                retained += 1
            fact_scores.append({
                "fact": fact,
                "best_similarity": round(best_sim, 4),
                "retained": is_retained,
            })

        retention_rate = retained / len(source_facts)
        passed = retention_rate >= PASS_THRESHOLD
        score = max(0.0, min(1.0, retention_rate))

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "retention_rate": round(retention_rate, 4),
                "retained_count": retained,
                "total_facts": len(source_facts),
                "fact_scores": fact_scores,
            },
        )
