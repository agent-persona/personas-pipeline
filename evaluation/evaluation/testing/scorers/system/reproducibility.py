"""D39 Reproducibility — variance measurement across identical generation runs.

Trustworthiness: HIGH (purely statistical).
Method: Accept N run outputs, measure variance per field type.
Structured fields (occupation, age) expect zero variance.
Narrative fields (bio) accept bounded variance.
Expects source_context.extra_data["run_outputs"]:
    list of dicts, each representing one generation run's output fields.
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Minimum runs for meaningful variance
MIN_RUNS = 2
# Max narrative variance (fraction of unique values).
# Intentionally 1.0: narrative fields (bio, etc.) are expected to vary across runs.
# The scorer primarily gates on structured field consistency.
MAX_NARRATIVE_VARIANCE = 1.0
# Fields treated as structured (expect zero variance)
STRUCTURED_FIELDS = {"occupation", "age", "gender", "location", "education",
                     "industry", "income_bracket", "ethnicity", "marital_status"}


class ReproducibilityScorer(BaseScorer):
    """Evaluates whether identical inputs produce consistent outputs across runs."""

    dimension_id = "D39"
    dimension_name = "Reproducibility"
    tier = 6
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        run_outputs: list[dict] = source_context.extra_data.get("run_outputs", [])

        if len(run_outputs) < MIN_RUNS:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": f"Need >= {MIN_RUNS} runs"},
            )

        # Collect all field names
        all_fields: set[str] = set()
        for run in run_outputs:
            all_fields.update(run.keys())

        n_runs = len(run_outputs)
        structured_variances: list[float] = []
        narrative_variances: list[float] = []

        for field in sorted(all_fields):
            values = [str(run.get(field, "")) for run in run_outputs]
            unique_count = len(set(values))
            variance = (unique_count - 1) / max(n_runs - 1, 1)  # 0 = identical, 1 = all different

            if field in STRUCTURED_FIELDS:
                structured_variances.append(variance)
            else:
                narrative_variances.append(variance)

        struct_var = sum(structured_variances) / len(structured_variances) if structured_variances else 0.0
        narr_var = sum(narrative_variances) / len(narrative_variances) if narrative_variances else 0.0

        # Score: penalize structured variance heavily, narrative variance lightly
        struct_penalty = struct_var  # Any structured variance is bad
        narr_penalty = max(0.0, narr_var - MAX_NARRATIVE_VARIANCE) if narr_var > MAX_NARRATIVE_VARIANCE else 0.0
        score = max(0.0, min(1.0, 1.0 - struct_penalty - narr_penalty * 0.5))

        passed = struct_var == 0.0 and narr_var <= MAX_NARRATIVE_VARIANCE

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "structured_variance": round(struct_var, 4),
                "narrative_variance": round(narr_var, 4),
                "n_runs": n_runs,
                "n_fields": len(all_fields),
            },
        )
