"""D33 Strategic Reasoning — Normalized Reward Accuracy on game-theoretic tasks.

Trustworthiness: HIGH (mathematical, directly measurable).
Method: Compute NRA = (actual - random) / (optimal - random) across game results.
Evidence: GTBench 2024 (NRA ≈ -1.0 on deterministic games),
Gao/Scylla 2024 (only fine-tuned model matched humans on 11-20 game).
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# NRA threshold: above this = acceptable strategic reasoning
NRA_PASS_THRESHOLD = 0.20  # Low bar: GTBench showed most LLMs are negative


class StrategicReasoningScorer(BaseScorer):
    """Evaluates strategic reasoning via Normalized Reward Accuracy."""

    dimension_id = "D33"
    dimension_name = "Strategic Reasoning"
    tier = 5
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        games = source_context.extra_data.get("game_results", [])
        if not games:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No game_results in extra_data"},
            )

        nras = []
        per_game = []

        for game in games:
            reward = game["reward"]
            optimal = game["optimal_reward"]
            random_ = game["random_reward"]

            denom = optimal - random_
            if abs(denom) < 1e-10:
                nra = 0.0
            else:
                nra = (reward - random_) / denom

            nras.append(nra)
            per_game.append({
                "game": game["game"],
                "nra": round(nra, 4),
                "reward": reward,
                "optimal_reward": optimal,
                "random_reward": random_,
            })

        mean_nra = sum(nras) / len(nras)
        # Score: map NRA from [-1, 1] to [0, 1]
        score = max(0.0, min(1.0, (mean_nra + 1.0) / 2.0))
        passed = mean_nra >= NRA_PASS_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "mean_nra": round(mean_nra, 4),
                "per_game": per_game,
                "game_count": len(games),
                "nra_pass_threshold": NRA_PASS_THRESHOLD,
            },
        )
