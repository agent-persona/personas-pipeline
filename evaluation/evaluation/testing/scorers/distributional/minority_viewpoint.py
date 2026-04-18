"""D16 Minority Viewpoint Preservation — within-group entropy.

Trustworthiness: HIGH (requires reference data, measurement is straightforward).
Method: For each demographic subgroup, measure opinion diversity within that group.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Group personas by this field
GROUPING_FIELD: tuple[str, Callable[[Persona], str]] = (
    "gender", lambda p: p.gender
)

# Opinion fields to measure diversity within each group
OPINION_FIELDS: list[tuple[str, Callable[[Persona], str]]] = [
    ("education", lambda p: p.education),
    ("lifestyle", lambda p: p.lifestyle),
    ("income_bracket", lambda p: p.income_bracket),
    ("comm.tone", lambda p: p.communication_style.tone),
    ("comm.formality", lambda p: p.communication_style.formality),
    ("emotional.baseline_mood", lambda p: p.emotional_profile.baseline_mood),
]


def _within_group_entropy(values: list[str]) -> float:
    """Normalized Shannon entropy within a group. 0 = no diversity, 1 = max diversity."""
    if len(values) < 2:
        return 0.0
    counter = Counter(values)
    n = len(values)
    num_cats = len(counter)
    if num_cats <= 1:
        return 0.0
    max_ent = math.log2(num_cats)
    if max_ent == 0:
        return 0.0
    ent = -sum((c / n) * math.log2(c / n) for c in counter.values())
    return ent / max_ent


class MinorityViewpointScorer(BaseScorer):
    """Evaluates whether minority subgroups maintain opinion diversity."""

    dimension_id = "D16"
    dimension_name = "Minority Viewpoint Preservation"
    tier = 3
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D16 is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        if len(personas) < 30:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 30 for meaningful within-group entropy (groups of 10+ each)"},
            )]

        # Group personas by the grouping field
        group_name, group_extractor = GROUPING_FIELD
        groups: dict[str, list[Persona]] = {}
        for p in personas:
            val = group_extractor(p)
            if val and val.strip():
                groups.setdefault(val, []).append(p)

        # Compute within-group entropy for each opinion field
        group_entropies: dict[str, dict[str, float]] = {}
        for group_val, group_personas in groups.items():
            if len(group_personas) < 3:
                continue
            field_entropies: dict[str, float] = {}
            for field_name, extractor in OPINION_FIELDS:
                vals = [extractor(p) for p in group_personas]
                vals = [v for v in vals if v and v.strip()]
                if len(vals) >= 2:
                    field_entropies[field_name] = round(_within_group_entropy(vals), 4)
            if field_entropies:
                group_entropies[group_val] = field_entropies

        if not group_entropies:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No groups with enough data"},
            )]

        # Average entropy across all groups and fields
        all_ents: list[float] = []
        for fields in group_entropies.values():
            all_ents.extend(fields.values())

        mean_ent = sum(all_ents) / len(all_ents) if all_ents else 0.0
        passed = mean_ent >= 0.3

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(mean_ent, 4),
            details={
                "within_group_entropy": group_entropies,
                "mean_entropy": round(mean_ent, 4),
                "persona_count": len(personas),
            },
        )]
