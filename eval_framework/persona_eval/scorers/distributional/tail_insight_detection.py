"""D-NEW Tail Insight Detection — measures ability to surface rare, non-obvious findings.

Trustworthiness: EXPERIMENTAL (requires curated benchmark; semantic similarity is a proxy).
Method: Compare persona responses against a benchmark of annotated tail insights
using embedding similarity. Compute recall, novelty-weighted recall, and detection-by-tier.
Evidence: Speero/NN/g practitioner evaluations — 0% tail insight detection across all cases.
Zero academic measurement exists. See docs/plans/2026-04-09-tail-insight-detection-design.md.
"""

from __future__ import annotations

from typing import Any

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# Default similarity threshold for detecting an insight
DEFAULT_SEMANTIC_THRESHOLD = 0.65

# Pass if recall exceeds this (very low bar — current state is 0%)
RECALL_PASS_THRESHOLD = 0.10


class TailInsightDetectionScorer(BaseScorer):
    """Measures whether synthetic personas surface tail insights from a benchmark."""

    dimension_id = "D-NEW"
    dimension_name = "Tail Insight Detection"
    tier = 6
    requires_set = True

    def __init__(self) -> None:
        self._embedder = None  # Embedder, lazily imported

    def _get_embedder(self):
        if self._embedder is None:
            from persona_eval.embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D-NEW is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        # Collect benchmark and responses from extra_data.
        # Convention: exactly one context carries each key (first non-empty wins).
        benchmark: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        for ctx in source_contexts:
            b = ctx.extra_data.get("tail_insights_benchmark", [])
            if b and not benchmark:
                benchmark = b
            r = ctx.extra_data.get("persona_responses", [])
            if r and not responses:
                responses = r

        if not benchmark:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No tail_insights_benchmark provided"},
            )]

        if not responses:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No persona_responses provided"},
            )]

        # Validate required keys in benchmark and responses
        for ins in benchmark:
            if "insight_text" not in ins or "insight_id" not in ins:
                return [EvalResult(
                    dimension_id=self.dimension_id,
                    dimension_name=self.dimension_name,
                    persona_id="__set__",
                    passed=False, score=0.0,
                    details={"error": "Benchmark entries must have 'insight_id' and 'insight_text'"},
                )]
        for r in responses:
            if "response_text" not in r:
                return [EvalResult(
                    dimension_id=self.dimension_id,
                    dimension_name=self.dimension_name,
                    persona_id="__set__",
                    passed=False, score=0.0,
                    details={"error": "Response entries must have 'response_text'"},
                )]

        # Embed benchmark insights and persona responses
        embedder = self._get_embedder()
        insight_texts = [ins["insight_text"] for ins in benchmark]
        response_texts = [r["response_text"] for r in responses]

        insight_vecs = embedder.embed_batch(insight_texts)
        response_vecs = embedder.embed_batch(response_texts)

        # Compute max similarity for each insight across all responses
        max_similarities: dict[str, float] = {}
        for i, insight in enumerate(benchmark):
            max_sim = 0.0
            for j in range(len(response_vecs)):
                sim = embedder.vector_similarity(insight_vecs[i], response_vecs[j])
                if sim > max_sim:
                    max_sim = sim
            max_similarities[insight["insight_id"]] = max_sim

        # Compute detection metrics
        metrics = self._compute_detection_metrics(benchmark, max_similarities)

        # Score: use novelty-weighted recall as the primary score
        score = min(1.0, max(0.0, metrics["novelty_weighted_recall"]))
        passed = metrics["tail_insight_recall"] >= RECALL_PASS_THRESHOLD

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                **metrics,
                "recall_pass_threshold": RECALL_PASS_THRESHOLD,
            },
        )]

    def _compute_detection_metrics(
        self,
        benchmark: list[dict[str, Any]],
        max_similarities: dict[str, float],
    ) -> dict[str, Any]:
        """Pure detection logic: given similarity scores, compute recall metrics.

        Args:
            benchmark: list of insight dicts with insight_id, novelty_score, semantic_threshold
            max_similarities: dict mapping insight_id → max similarity score across all responses

        Returns:
            Detection metrics dict.
        """
        detected = []
        missed = []

        for insight in benchmark:
            iid = insight["insight_id"]
            threshold = insight.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD)
            sim = max_similarities.get(iid, 0.0)
            if sim >= threshold:
                detected.append(iid)
            else:
                missed.append(iid)

        total = len(benchmark)
        recall = len(detected) / total if total > 0 else 0.0
        detected_set = set(detected)  # O(1) membership lookups below

        # Novelty-weighted recall: Σ(detected_i × novelty_i) / Σ(novelty_i)
        total_novelty = sum(i["novelty_score"] for i in benchmark)
        detected_novelty = sum(
            i["novelty_score"] for i in benchmark if i["insight_id"] in detected_set
        )
        novelty_weighted_recall = (
            detected_novelty / total_novelty if total_novelty > 0 else recall
        )

        # Detection by novelty tier
        tiers = {
            "low (1-2)": {"detected": 0, "total": 0},
            "medium (3)": {"detected": 0, "total": 0},
            "high (4-5)": {"detected": 0, "total": 0},
        }
        for insight in benchmark:
            ns = insight["novelty_score"]
            if ns <= 2:
                tier_key = "low (1-2)"
            elif ns == 3:
                tier_key = "medium (3)"
            else:
                tier_key = "high (4-5)"
            tiers[tier_key]["total"] += 1
            if insight["insight_id"] in detected_set:
                tiers[tier_key]["detected"] += 1

        detection_by_novelty = {}
        for tier_key, counts in tiers.items():
            detection_by_novelty[tier_key] = {
                "recall": round(counts["detected"] / counts["total"], 4) if counts["total"] > 0 else 0.0,
                "detected": counts["detected"],
                "total": counts["total"],
            }

        return {
            "tail_insight_recall": round(recall, 4),
            "novelty_weighted_recall": round(novelty_weighted_recall, 4),
            "detection_by_novelty": detection_by_novelty,
            "detected_insights": detected,
            "missed_insights": missed,
            "benchmark_size": total,
            "detected_count": len(detected),
        }
