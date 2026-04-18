"""D15 Structural Aggregation Consistency — split-half stability test.

Trustworthiness: HIGH (deterministic comparison).
Method: Split persona set into random halves, compare attribute distributions
between halves. Stable distributions = consistent generation. Also checks
per-group distribution similarity to detect demographic skew.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Group personas by these fields
GROUPING_FIELDS: list[tuple[str, Callable[[Persona], str]]] = [
    ("gender", lambda p: p.gender),
    ("education", lambda p: p.education),
    ("income_bracket", lambda p: p.income_bracket),
]

# Compute distributions of these fields within/across groups
DISTRIBUTION_FIELDS: list[tuple[str, Callable[[Persona], str]]] = [
    ("occupation", lambda p: p.occupation),
    ("industry", lambda p: p.industry),
    ("lifestyle", lambda p: p.lifestyle),
]

N_SPLITS = 5  # number of random split-half trials
SPLIT_SEED = 42


def _jensen_shannon_divergence(dist_a: Counter, dist_b: Counter) -> float:
    """Compute Jensen-Shannon divergence between two distributions. Returns 0-1."""
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    if not all_keys:
        return 0.0

    total_a = sum(dist_a.values()) or 1
    total_b = sum(dist_b.values()) or 1

    p = {k: dist_a.get(k, 0) / total_a for k in all_keys}
    q = {k: dist_b.get(k, 0) / total_b for k in all_keys}

    # M = average distribution
    m = {k: (p[k] + q[k]) / 2 for k in all_keys}

    def kl(dist: dict[str, float], ref: dict[str, float]) -> float:
        return sum(
            dist[k] * math.log2(dist[k] / ref[k])
            for k in all_keys
            if dist[k] > 0 and ref[k] > 0
        )

    return (kl(p, m) + kl(q, m)) / 2


class AggregationConsistencyScorer(BaseScorer):
    """Check structural consistency via split-half stability and cross-group comparison."""

    dimension_id = "D15"
    dimension_name = "Structural Aggregation Consistency"
    tier = 3
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D15 is a set-level dimension"},
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
                details={"skipped": True, "reason": "Need >= 50 for stable split-half (25 per half)", "min_recommended": 50},
            )]

        split_half_scores = self._split_half_consistency(personas)
        cross_group_results = self._cross_group_consistency(personas)

        # Combine both measures
        all_scores = split_half_scores + [r["consistency"] for r in cross_group_results]
        if not all_scores:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True},
            )]

        mean_consistency = sum(all_scores) / len(all_scores)
        passed = mean_consistency >= 0.7

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(mean_consistency, 4),
            details={
                "consistency_ratio": round(mean_consistency, 4),
                "split_half_scores": [round(s, 4) for s in split_half_scores],
                "group_comparisons": cross_group_results,
            },
        )]

    def _split_half_consistency(self, personas: list[Persona]) -> list[float]:
        """Split set into random halves, compare distributions. Returns per-trial scores."""
        rng = random.Random(SPLIT_SEED)
        scores: list[float] = []

        for _ in range(N_SPLITS):
            shuffled = list(personas)
            rng.shuffle(shuffled)
            mid = len(shuffled) // 2
            half_a, half_b = shuffled[:mid], shuffled[mid:]

            trial_jsd: list[float] = []
            for field_name, extractor in DISTRIBUTION_FIELDS:
                dist_a = Counter(extractor(p) for p in half_a if extractor(p))
                dist_b = Counter(extractor(p) for p in half_b if extractor(p))
                if dist_a and dist_b:
                    jsd = _jensen_shannon_divergence(dist_a, dist_b)
                    trial_jsd.append(jsd)

            if trial_jsd:
                # JSD of 0 = identical distributions, 1 = maximally different
                # Convert to consistency score: 1 - mean_jsd
                scores.append(1.0 - (sum(trial_jsd) / len(trial_jsd)))

        return scores

    def _cross_group_consistency(self, personas: list[Persona]) -> list[dict]:
        """Compare per-group distributions to overall distribution."""
        results: list[dict] = []

        for group_name, group_extractor in GROUPING_FIELDS:
            groups: dict[str, list[Persona]] = {}
            for p in personas:
                val = group_extractor(p)
                if val and val.strip():
                    groups.setdefault(val, []).append(p)

            # Skip if fewer than 2 groups with meaningful size
            meaningful_groups = {k: v for k, v in groups.items() if len(v) >= 3}
            if len(meaningful_groups) < 2:
                continue

            for field_name, field_extractor in DISTRIBUTION_FIELDS:
                overall_dist = Counter(
                    field_extractor(p) for p in personas if field_extractor(p)
                )
                if not overall_dist:
                    continue

                group_jsds: list[float] = []
                for group_val, group_personas in meaningful_groups.items():
                    group_dist = Counter(
                        field_extractor(p) for p in group_personas if field_extractor(p)
                    )
                    if group_dist:
                        jsd = _jensen_shannon_divergence(group_dist, overall_dist)
                        group_jsds.append(jsd)

                if group_jsds:
                    mean_jsd = sum(group_jsds) / len(group_jsds)
                    consistency = round(1.0 - mean_jsd, 4)
                    results.append({
                        "group_by": group_name,
                        "field": field_name,
                        "consistency": consistency,
                    })

        return results
