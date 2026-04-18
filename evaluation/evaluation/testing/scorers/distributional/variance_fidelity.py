"""D14 Variance Fidelity — IQR comparison, K-S test.

Trustworthiness: HIGH (purely statistical, optionally compared to reference data).
Method: Compare IQR of numeric fields; K-S test for distribution shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Numeric fields: (field_name, extractor)
NUMERIC_FIELDS: list[tuple[str, Callable[[Persona], float | None]]] = [
    ("age", lambda p: float(p.age) if p.age is not None else None),
    ("experience_years", lambda p: float(p.experience_years) if p.experience_years is not None else None),
]


class VarianceFidelityScorer(BaseScorer):
    """Evaluates whether the persona set has realistic variance in numeric fields."""

    dimension_id = "D14"
    dimension_name = "Variance Fidelity"
    tier = 3
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D14 is a set-level dimension"},
        )

    def score_set(
        self,
        personas: list[Persona],
        source_contexts: list[SourceContext],
        **kwargs: Any,  # accepts reference_distributions: dict[str, {"mean": float, "std": float}]
    ) -> list[EvalResult]:
        reference_distributions: dict[str, dict] = kwargs.get("reference_distributions", {})

        if len(personas) < 20:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 20 personas"},
            )]

        iqr_scores: dict[str, float] = {}
        zero_variance_fields: list[str] = []
        ks_tests: dict[str, dict] = {}

        for field_name, extractor in NUMERIC_FIELDS:
            values = [v for p in personas if (v := extractor(p)) is not None]
            if len(values) < 5:
                continue

            arr = np.array(values)
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            std = float(np.std(arr))

            if iqr == 0 and std == 0:
                zero_variance_fields.append(field_name)
                iqr_scores[field_name] = 0.0
            else:
                mean = float(np.mean(arr))
                cv = std / mean if mean != 0 else 0.0
                iqr_scores[field_name] = round(min(1.0, cv / 0.3), 4)

            if field_name in reference_distributions:
                ref = reference_distributions[field_name]
                ref_mean = ref.get("mean", float(np.mean(arr)))
                ref_std = ref.get("std", float(np.std(arr)))
                if ref_std > 0:
                    ks_stat, ks_p = stats.kstest(arr, "norm", args=(ref_mean, ref_std))
                    ks_tests[field_name] = {
                        "statistic": round(float(ks_stat), 4),
                        "p_value": round(float(ks_p), 4),
                    }

        if not iqr_scores:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No numeric fields"},
            )]

        mean_score = sum(iqr_scores.values()) / len(iqr_scores)
        passed = len(zero_variance_fields) == 0 and mean_score >= 0.3

        details: dict[str, Any] = {
            "iqr_scores": iqr_scores,
            "zero_variance_fields": zero_variance_fields,
            "persona_count": len(personas),
        }
        if ks_tests:
            details["ks_tests"] = ks_tests

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(mean_score, 4),
            details=details,
        )]
