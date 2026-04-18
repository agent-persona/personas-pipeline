"""D40 Cost/Latency Bounds — token counting and latency profiling.

Trustworthiness: HIGH (objective measurement).
Method: Accept cost and latency metrics, check against configurable bounds.
Score degrades linearly when exceeding bounds.
Expects source_context.extra_data:
    "cost_usd": float (total cost for generating this persona)
    "latency_seconds": float (total generation time)
    Optional "token_counts": dict with input_tokens, output_tokens
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext


class CostLatencyScorer(BaseScorer):
    """Evaluates whether persona generation stays within cost and latency budgets."""

    dimension_id = "D40"
    dimension_name = "Cost/Latency Bounds"
    tier = 6
    requires_set = False

    def __init__(self, max_cost_per_persona: float = 1.0, max_latency_seconds: float = 60.0):
        self.max_cost = max_cost_per_persona
        self.max_latency = max_latency_seconds

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        cost = source_context.extra_data.get("cost_usd")
        latency = source_context.extra_data.get("latency_seconds")

        if cost is None and latency is None:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No cost/latency metrics in extra_data"},
            )

        cost = cost if cost is not None else 0.0
        latency = latency if latency is not None else 0.0
        token_counts = source_context.extra_data.get("token_counts", {})

        cost_ok = cost <= self.max_cost
        latency_ok = latency <= self.max_latency

        # Score: 1.0 if within bounds, degrades linearly
        cost_score = min(1.0, self.max_cost / cost) if cost > 0 else 1.0
        latency_score = min(1.0, self.max_latency / latency) if latency > 0 else 1.0
        score = max(0.0, min(1.0, (cost_score + latency_score) / 2))

        return self._result(
            persona, passed=cost_ok and latency_ok, score=round(score, 4),
            details={
                "cost_usd": round(cost, 4),
                "max_cost_usd": self.max_cost,
                "latency_seconds": round(latency, 2),
                "max_latency_seconds": self.max_latency,
                "token_counts": token_counts,
                "cost_within_budget": cost_ok,
                "latency_within_budget": latency_ok,
            },
        )
