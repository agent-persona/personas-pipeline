"""D10 Lexical vs Semantic Generalization — paraphrase consistency.

Trustworthiness: MEDIUM (measures surface-level semantic consistency).
Method: Embed original and paraphrased responses, measure cosine similarity.
A consistent persona should give semantically similar answers to paraphrased questions.
"""

from __future__ import annotations

import numpy as np

from evaluation.testing.embeddings import Embedder
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext


class LexicalSemanticScorer(BaseScorer):
    """Evaluates whether a persona gives consistent answers to paraphrased questions."""

    dimension_id = "D10"
    dimension_name = "Lexical vs Semantic Generalization"
    tier = 2

    def __init__(self) -> None:
        self._embedder: Embedder | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        """Measure consistency between original and paraphrased response pairs.

        Reads from source_context.extra_data["response_pairs"]:
        a list of {"original": str, "paraphrased": str} dicts.
        """
        pairs: list[dict] = source_context.extra_data.get("response_pairs", [])

        if len(pairs) < 2:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 2 response pairs"},
            )

        embedder = self._get_embedder()
        pair_sims: list[float] = []

        for pair in pairs:
            sim = embedder.similarity(pair["original"], pair["paraphrased"])
            pair_sims.append(sim)

        mean_sim = float(np.mean(pair_sims))
        min_sim = float(np.min(pair_sims))

        score = max(0.0, min(1.0, mean_sim))
        passed = mean_sim >= 0.5

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "mean_similarity": round(mean_sim, 4),
                "min_similarity": round(min_sim, 4),
                "n_pairs": len(pairs),
                "pair_similarities": [round(s, 4) for s in pair_sims],
            },
        )
