"""D9 Knowledge Boundary Awareness — out-of-domain probe battery.

Trustworthiness: MEDIUM (requires reliable model of what the persona should know).
Method: Check responses to out-of-domain questions for appropriate uncertainty markers.
"""

from __future__ import annotations

import re

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Signals of appropriate uncertainty
UNCERTAINTY_MARKERS = [
    re.compile(r"\b(not sure|don'?t know|not my area|outside my expertise|can'?t speak to)\b", re.IGNORECASE),
    re.compile(r"\b(not really|wouldn'?t know|not qualified|not familiar|don'?t have experience)\b", re.IGNORECASE),
    re.compile(r"\b(I'?d have to|you'?d be better off asking|I'?m not the right person)\b", re.IGNORECASE),
]


class KnowledgeBoundaryScorer(BaseScorer):
    """Evaluates whether a persona shows appropriate uncertainty on out-of-domain topics."""

    dimension_id = "D9"
    dimension_name = "Knowledge Boundary Awareness"
    tier = 2

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        """Check if persona shows appropriate uncertainty on out-of-domain questions.

        Reads from source_context.extra_data["ood_responses"]:
        a list of {"domain": str, "response": str} dicts.
        """
        ood_responses: list[dict] = source_context.extra_data.get("ood_responses", [])

        if not ood_responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No out-of-domain responses to evaluate"},
            )

        probe_results = []
        for item in ood_responses:
            response = item["response"]
            shows_uncertainty = any(p.search(response) for p in UNCERTAINTY_MARKERS)
            probe_results.append({
                "domain": item["domain"],
                "shows_uncertainty": shows_uncertainty,
                "response_snippet": response[:100],
            })

        appropriate_count = sum(1 for r in probe_results if r["shows_uncertainty"])
        score = appropriate_count / len(probe_results) if probe_results else 0.0
        passed = score >= 0.6

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "probe_results": probe_results,
                "appropriate_uncertainty_rate": round(score, 4),
            },
        )
