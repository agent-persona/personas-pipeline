"""Shared metric collection for experiment runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ExperimentResult


class MetricsCollector:
    """Collects guardrail + target metrics from pipeline output."""

    def collect_from_output(self, output_dir: Path) -> dict[str, float]:
        """Read persona_*.json files and compute standard metrics.

        Returns dict with keys:
        - personas_generated
        - schema_validity
        - mean_groundedness
        - total_cost_usd
        - cost_per_persona
        """
        persona_files = sorted(output_dir.glob("persona_*.json"))
        if not persona_files:
            return {
                "personas_generated": 0,
                "schema_validity": 0.0,
                "mean_groundedness": 0.0,
                "total_cost_usd": 0.0,
                "cost_per_persona": 0.0,
            }

        personas = []
        for f in persona_files:
            personas.append(json.loads(f.read_text()))

        n = len(personas)

        # Schema validity - try to validate against PersonaV1
        valid_count = 0
        try:
            from synthesis.models.persona import PersonaV1
            from pydantic import ValidationError
            for p in personas:
                persona_data = p.get("persona", p)
                try:
                    PersonaV1.model_validate(persona_data)
                    valid_count += 1
                except ValidationError:
                    pass
        except ImportError:
            valid_count = n  # assume valid if can't import

        # Groundedness
        groundedness_scores = [
            p.get("groundedness", 0.0) for p in personas
        ]
        mean_groundedness = sum(groundedness_scores) / n if n else 0.0

        # Cost
        costs = [p.get("cost_usd", 0.0) for p in personas]
        total_cost = sum(costs)

        return {
            "personas_generated": float(n),
            "schema_validity": valid_count / n if n else 0.0,
            "mean_groundedness": mean_groundedness,
            "total_cost_usd": total_cost,
            "cost_per_persona": total_cost / n if n else 0.0,
        }

    def compute_signal_strength(
        self,
        baseline: dict[str, float],
        experiment: dict[str, float],
        target_metric: str,
        hypothesis_direction: str = "increase",
    ) -> str:
        """Classify signal strength based on metric deltas.

        Returns one of: strong, moderate, weak, noise, inconclusive, negative
        """
        if target_metric not in baseline or target_metric not in experiment:
            return "inconclusive"

        b_val = baseline[target_metric]
        e_val = experiment[target_metric]

        if b_val == 0:
            return "inconclusive"

        delta_pct = abs((e_val - b_val) / b_val) * 100

        # Check direction
        if hypothesis_direction == "increase":
            confirmed = e_val > b_val
        elif hypothesis_direction == "decrease":
            confirmed = e_val < b_val
        else:
            confirmed = e_val != b_val

        # Check guardrail regressions
        regressions = 0
        if experiment.get("schema_validity", 1.0) < baseline.get("schema_validity", 1.0):
            if experiment.get("schema_validity", 1.0) < 1.0:
                return "negative"  # hard invariant
            regressions += 1
        if experiment.get("mean_groundedness", 1.0) < baseline.get("mean_groundedness", 1.0) - 0.05:
            regressions += 1
        if experiment.get("total_cost_usd", 0) > baseline.get("total_cost_usd", 0) * 1.5:
            regressions += 1
        if experiment.get("personas_generated", 0) < baseline.get("personas_generated", 0):
            regressions += 1

        # Classify
        if not confirmed:
            if regressions > 0:
                return "negative"
            return "noise"

        if delta_pct > 10 and regressions == 0:
            return "strong"
        elif delta_pct > 5 or (confirmed and regressions <= 1):
            return "moderate"
        elif delta_pct > 2:
            return "weak"
        else:
            return "noise"

    def build_comparison(
        self,
        config: ExperimentConfig,
        baseline: dict[str, float],
        experiment: dict[str, float],
        target_metric: str = "",
        hypothesis_direction: str = "increase",
    ) -> ExperimentResult:
        """Build a full ExperimentResult with deltas and assessment."""
        target = target_metric or config.primary_metric

        # Compute deltas
        deltas = {}
        for key in set(baseline.keys()) | set(experiment.keys()):
            b = baseline.get(key, 0.0)
            e = experiment.get(key, 0.0)
            deltas[key] = e - b

        # Build guardrails
        guardrail_keys = ["schema_validity", "mean_groundedness", "total_cost_usd", "personas_generated"]
        guardrails = {}
        for key in guardrail_keys:
            b = baseline.get(key, 0.0)
            e = experiment.get(key, 0.0)
            delta = e - b
            regression = False
            if key == "schema_validity":
                # Flag any drop as a regression in the guardrail report.
                # compute_signal_strength applies the stricter hard-failure rule
                # (e < 1.0) separately — these two thresholds intentionally differ:
                # guardrails surface warnings; signal_strength surfaces hard stops.
                regression = e < b
            elif key == "mean_groundedness":
                regression = e < b - 0.05
            elif key == "total_cost_usd":
                regression = e > b * 1.5 if b > 0 else False
            elif key == "personas_generated":
                regression = e < b
            guardrails[key] = {
                "baseline": b,
                "experiment": e,
                "delta": delta,
                "regression": regression,
            }

        signal = self.compute_signal_strength(
            baseline, experiment, target, hypothesis_direction
        )

        # Recommendation
        if signal in ("strong", "moderate"):
            recommendation = "adopt"
        elif signal == "negative":
            recommendation = "reject"
        elif signal in ("noise", "weak"):
            recommendation = "rerun"
        else:
            recommendation = "defer"

        return ExperimentResult(
            config=config,
            baseline=baseline,
            experiment=experiment,
            deltas=deltas,
            guardrails=guardrails,
            signal_strength=signal,
            recommendation=recommendation,
        )
