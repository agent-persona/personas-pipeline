"""D26 Empathetic Responsiveness — empathy detection in responses to emotional prompts.

Trustworthiness: MEDIUM (keyword-based empathy detection is approximate).
Method: Scan responses to emotional prompts for empathy markers.
Expects source_context.extra_data["empathy_probes"]: list of {"prompt": str, "response": str}.
"""

from __future__ import annotations

import re

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

EMPATHY_MARKERS = re.compile(
    r"\b(sorry|understand|feel|must be|that sounds|difficult|tough|hard|"
    r"support|care|hear|empathize|sympathize|compassion|concern|"
    r"how are you|are you ok|holding up|here for you|listen|"
    r"I can only imagine|that.s awful|my heart goes out|"
    r"challenging|overwhelming|painful)\b",
    re.IGNORECASE,
)

DISMISSIVE_MARKERS = re.compile(
    r"\b(get over it|stop complaining|toughen up|not a big deal|everyone|"
    r"statistically|you should have|obviously|just deal|move on|"
    r"that.s life|grow up|suck it up|why would you)\b",
    re.IGNORECASE,
)

# Pass if empathy rate >= this
EMPATHY_THRESHOLD = 0.5


class EmpatheticResponsivenessScorer(BaseScorer):
    """Evaluates whether responses show appropriate empathy to emotional prompts."""

    dimension_id = "D26"
    dimension_name = "Empathetic Responsiveness"
    tier = 5
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        probes: list[dict] = source_context.extra_data.get("empathy_probes", [])
        if not probes:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No empathy_probes in extra_data"},
            )

        empathetic_count = 0
        for probe in probes:
            response = probe.get("response", "")
            emp_hits = len(EMPATHY_MARKERS.findall(response))
            dis_hits = len(DISMISSIVE_MARKERS.findall(response))
            if emp_hits > dis_hits:
                empathetic_count += 1

        empathy_rate = empathetic_count / len(probes)
        score = empathy_rate
        passed = empathy_rate >= EMPATHY_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "empathy_rate": round(empathy_rate, 4),
                "empathetic_count": empathetic_count,
                "probe_count": len(probes),
            },
        )
