"""Suite runner — orchestrates all scorers with tier gating."""

from __future__ import annotations

import time

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext


class SuiteRunner:
    """Orchestrates evaluation across all dimensions.

    Respects tier gating: Tier 1 must pass before Tier 2+ runs.
    Handles both per-persona and set-level scorers.
    """

    def __init__(self, scorers: list[BaseScorer]):
        self.scorers = scorers
        self._tiers: dict[int, list[BaseScorer]] = {}
        self._dim_tiers: dict[str, int] = {}
        for s in scorers:
            self._tiers.setdefault(s.tier, []).append(s)
            self._dim_tiers[s.dimension_id] = s.tier

    def run(
        self,
        persona: Persona,
        source_context: SourceContext,
        tier_filter: int | None = None,
    ) -> list[EvalResult]:
        """Run per-persona scorers with tier gating.

        Set-level scorers (requires_set=True) are skipped — use run_set() for those.
        """
        all_results: list[EvalResult] = []

        for tier in sorted(self._tiers):
            if tier_filter is not None and tier != tier_filter:
                continue

            scorers = self._tiers[tier]

            # Tier gating: Tier 1 must pass before higher tiers run
            if tier > 1 and tier_filter is None:
                tier1_results = [r for r in all_results if self._dim_tiers.get(r.dimension_id) == 1]
                if tier1_results and not all(r.passed for r in tier1_results):
                    for s in scorers:
                        if s.requires_set:
                            continue  # Set-level gating handled by run_full()
                        all_results.append(EvalResult(
                            dimension_id=s.dimension_id,
                            dimension_name=s.dimension_name,
                            persona_id=persona.id,
                            passed=False,
                            score=0.0,
                            details={"skipped": True, "reason": "Tier 1 gating failure"},
                        ))
                    continue

            for scorer in scorers:
                if scorer.requires_set:
                    continue  # Skip set-level scorers in per-persona run
                start = time.time()
                try:
                    result = scorer.score(persona, source_context)
                    result.details["elapsed_seconds"] = round(time.time() - start, 3)
                    all_results.append(result)
                except Exception as e:
                    all_results.append(EvalResult(
                        dimension_id=scorer.dimension_id,
                        dimension_name=scorer.dimension_name,
                        persona_id=persona.id,
                        passed=False,
                        score=0.0,
                        errors=[str(e)],
                    ))

        return all_results

    def run_set(
        self,
        personas: list[Persona],
        source_contexts: list[SourceContext],
        tier_filter: int | None = None,
    ) -> list[EvalResult]:
        """Run set-level scorers on a collection of personas.

        Per-persona scorers are skipped — use run() for those.
        """
        all_results: list[EvalResult] = []

        for tier in sorted(self._tiers):
            if tier_filter is not None and tier != tier_filter:
                continue

            scorers = self._tiers[tier]

            for scorer in scorers:
                if not scorer.requires_set:
                    continue  # Skip per-persona scorers in set run
                start = time.time()
                try:
                    results = scorer.score_set(personas, source_contexts)
                    elapsed = round(time.time() - start, 3)
                    for r in results:
                        r.details["elapsed_seconds"] = elapsed
                    all_results.extend(results)
                except Exception as e:
                    all_results.append(EvalResult(
                        dimension_id=scorer.dimension_id,
                        dimension_name=scorer.dimension_name,
                        persona_id="__set__",
                        passed=False,
                        score=0.0,
                        errors=[str(e)],
                    ))

        return all_results

    def run_full(
        self,
        personas: list[Persona],
        source_contexts: list[SourceContext],
        tier_filter: int | None = None,
    ) -> list[EvalResult]:
        """Run all scorers — per-persona on each persona, then set-level on the full set.

        Tier gating applies: if any Tier 1 scorer fails on any persona,
        higher tiers are skipped entirely.
        """
        all_results: list[EvalResult] = []
        tier1_passed = True

        # Per-persona scoring
        for persona, ctx in zip(personas, source_contexts):
            results = self.run(persona, ctx, tier_filter=tier_filter)
            all_results.extend(results)
            # Check Tier 1 gating across all personas
            for r in results:
                if self._dim_tiers.get(r.dimension_id) == 1 and not r.passed:
                    tier1_passed = False

        # Set-level scoring (only if Tier 1 passed)
        if tier1_passed:
            set_results = self.run_set(personas, source_contexts, tier_filter=tier_filter)
            all_results.extend(set_results)
        else:
            # Skip all set-level scorers
            for scorer in self.scorers:
                if scorer.requires_set:
                    if tier_filter is not None and scorer.tier != tier_filter:
                        continue
                    all_results.append(EvalResult(
                        dimension_id=scorer.dimension_id,
                        dimension_name=scorer.dimension_name,
                        persona_id="__set__",
                        passed=False,
                        score=0.0,
                        details={"skipped": True, "reason": "Tier 1 gating failure"},
                    ))

        return all_results
