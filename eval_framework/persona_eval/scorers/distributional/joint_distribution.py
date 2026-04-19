"""D18 Joint Distribution Fidelity — correlation matrix comparison.

Trustworthiness: HIGH (mathematical, but requires good reference data).
Method: Compute attribute correlations via Cramer's V, detect stereotypical over-correlation.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# Categorical fields: (field_name, extractor)
CATEGORICAL_FIELDS: list[tuple[str, Callable[[Persona], str]]] = [
    ("gender", lambda p: p.gender),
    ("education", lambda p: p.education),
    ("income_bracket", lambda p: p.income_bracket),
    ("lifestyle", lambda p: p.lifestyle),
    ("marital_status", lambda p: p.marital_status),
    ("comm.tone", lambda p: p.communication_style.tone),
    ("comm.formality", lambda p: p.communication_style.formality),
    ("emotional.baseline_mood", lambda p: p.emotional_profile.baseline_mood),
]

STEREOTYPE_THRESHOLD = 0.7  # Cramer's V > 0.7 = suspiciously high correlation
REGRESSION_WRONG_THRESHOLD = 0.10  # absolute error > 0.10 = "wrong"
REGRESSION_SIGN_FLIP_WARN = 0.10   # >10% sign flips = concerning (Bisbee found 32%)


def _cramers_v(x: list[str], y: list[str]) -> float:
    """Compute Cramer's V statistic for two categorical variables (pure Python)."""
    n = len(x)
    if n != len(y) or n < 5:
        return 0.0

    x_cats = sorted(set(x))
    y_cats = sorted(set(y))
    if len(x_cats) < 2 or len(y_cats) < 2:
        return 0.0

    # Build contingency table
    x_idx = {c: i for i, c in enumerate(x_cats)}
    y_idx = {c: i for i, c in enumerate(y_cats)}
    table = [[0] * len(y_cats) for _ in range(len(x_cats))]

    for xi, yi in zip(x, y):
        table[x_idx[xi]][y_idx[yi]] += 1

    # Row and column sums
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[r][c] for r in range(len(x_cats))) for c in range(len(y_cats))]

    # Chi-squared statistic
    chi2 = 0.0
    for r in range(len(x_cats)):
        for c in range(len(y_cats)):
            expected = (row_sums[r] * col_sums[c]) / n
            if expected > 0:
                chi2 += (table[r][c] - expected) ** 2 / expected

    k = min(len(x_cats), len(y_cats))
    if k <= 1:
        return 0.0

    return (chi2 / (n * (k - 1))) ** 0.5


class JointDistributionScorer(BaseScorer):
    """Evaluates joint distribution fidelity — flags stereotypical attribute correlations."""

    dimension_id = "D18"
    dimension_name = "Joint Distribution Fidelity"
    tier = 3
    requires_set = True

    def _regression_analysis(self, source_contexts: list[SourceContext]) -> dict[str, Any]:
        """Compare real vs synthetic regression coefficients (Bisbee 2024 methodology)."""
        ref = None
        for ctx in source_contexts:
            ref = ctx.extra_data.get("regression_reference")
            if ref:
                break
        if not ref:
            return {}

        coefficients = ref.get("coefficients", [])
        if not coefficients:
            return {}

        wrong_count = 0
        sign_flip_count = 0
        abs_errors = []

        for coeff in coefficients:
            real = coeff["real_coeff"]
            synthetic = coeff["synthetic_coeff"]
            abs_error = abs(real - synthetic)
            abs_errors.append(abs_error)

            if abs_error > REGRESSION_WRONG_THRESHOLD:
                wrong_count += 1
            if (real > 0 and synthetic < 0) or (real < 0 and synthetic > 0):
                sign_flip_count += 1

        n = len(coefficients)
        return {
            "regression_wrong_rate": round(wrong_count / n, 4),
            "regression_sign_flip_rate": round(sign_flip_count / n, 4),
            "regression_mae": round(sum(abs_errors) / n, 4),
            "regression_coefficient_count": n,
        }

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D18 is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        if len(personas) < 50:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 50 for stable Cramer's V"},
            )]

        # Extract all categorical fields per-persona (preserve index alignment)
        field_data: dict[str, list[str | None]] = {}
        for field_name, extractor in CATEGORICAL_FIELDS:
            vals: list[str | None] = []
            for p in personas:
                v = extractor(p)
                vals.append(v if v and v.strip() else None)
            non_none = sum(1 for v in vals if v is not None)
            if non_none >= 10:
                field_data[field_name] = vals

        if len(field_data) < 2:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 2 categorical fields with data"},
            )]

        # Compute pairwise Cramer's V (pair values per-persona, filter missing)
        correlation_matrix: dict[str, float] = {}
        stereotypical_pairs: list[dict] = []
        all_vs: list[float] = []

        for (name_a, vals_a), (name_b, vals_b) in combinations(field_data.items(), 2):
            paired_a, paired_b = [], []
            for a, b in zip(vals_a, vals_b):
                if a is not None and b is not None:
                    paired_a.append(a)
                    paired_b.append(b)
            if len(paired_a) < 5:
                continue
            v = _cramers_v(paired_a, paired_b)
            pair_key = f"{name_a} x {name_b}"
            correlation_matrix[pair_key] = round(v, 4)
            all_vs.append(v)

            if v > STEREOTYPE_THRESHOLD:
                stereotypical_pairs.append({"pair": pair_key, "cramers_v": round(v, 4)})

        mean_v = sum(all_vs) / len(all_vs) if all_vs else 0.0
        # Lower correlation = better (less stereotypical)
        score = max(0.0, 1.0 - mean_v)
        if stereotypical_pairs:
            score *= 0.5  # heavy penalty for stereotypical correlations

        passed = len(stereotypical_pairs) == 0 and mean_v < 0.5

        details: dict[str, Any] = {
            "correlation_matrix": correlation_matrix,
            "mean_cramers_v": round(mean_v, 4),
            "stereotypical_pairs": stereotypical_pairs,
            "persona_count": len(personas),
        }

        regression = self._regression_analysis(source_contexts)
        details.update(regression)

        if regression.get("regression_sign_flip_rate", 0) > REGRESSION_SIGN_FLIP_WARN:
            passed = False

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details=details,
        )]
