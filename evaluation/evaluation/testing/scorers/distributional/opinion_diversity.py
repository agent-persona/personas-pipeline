"""D13 Opinion Diversity — variation ratio, entropy, modal collapse detection.

Trustworthiness: HIGH (mathematically grounded, directly testable).
Method: Compute variation ratio and Shannon entropy across categorical fields.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Categorical fields to measure diversity across: (attribute_name, extractor)
DIVERSITY_FIELDS: list[tuple[str, Callable[[Persona], str]]] = [
    ("gender", lambda p: p.gender),
    ("education", lambda p: p.education),
    ("occupation", lambda p: p.occupation),
    ("industry", lambda p: p.industry),
    ("income_bracket", lambda p: p.income_bracket),
    ("ethnicity", lambda p: p.ethnicity),
    ("marital_status", lambda p: p.marital_status),
    ("lifestyle", lambda p: p.lifestyle),
    ("comm.tone", lambda p: p.communication_style.tone),
    ("comm.formality", lambda p: p.communication_style.formality),
    ("comm.vocabulary_level", lambda p: p.communication_style.vocabulary_level),
    ("emotional.baseline_mood", lambda p: p.emotional_profile.baseline_mood),
]

MODAL_COLLAPSE_THRESHOLD = 0.80  # >80% same answer = modal collapse
ENTRENCHMENT_THRESHOLD = 0.40  # >40% of questions invariant = concerning (CLAIMSIM found 50%)
MODAL_RESPONSE_THRESHOLD = 0.80  # >80% same response within a question = entrenched


def _variation_ratio(values: list[str]) -> float:
    """Fraction of values that are NOT the mode. 0 = all same, ~1 = perfectly uniform."""
    if not values:
        return 0.0
    counter = Counter(values)
    mode_count = counter.most_common(1)[0][1]
    return 1.0 - (mode_count / len(values))


def _shannon_entropy(values: list[str]) -> float:
    """Normalized Shannon entropy. 0 = no diversity, 1 = maximum diversity."""
    if not values:
        return 0.0
    counter = Counter(values)
    n = len(values)
    num_categories = len(counter)
    if num_categories <= 1:
        return 0.0
    max_entropy = math.log2(num_categories)
    if max_entropy == 0:
        return 0.0
    entropy = -sum((count / n) * math.log2(count / n) for count in counter.values())
    return entropy / max_entropy


class OpinionDiversityScorer(BaseScorer):
    """Evaluates opinion/attribute diversity across a persona set."""

    dimension_id = "D13"
    dimension_name = "Opinion Diversity"
    tier = 3
    requires_set = True

    def _detect_entrenchment(
        self, source_contexts: list[SourceContext]
    ) -> dict[str, Any]:
        """Detect RLHF entrenchment: questions where demographics don't affect response."""
        all_responses: dict[str, list[str]] = {}  # question_id -> [responses]
        has_data = False
        for ctx in source_contexts:
            survey = ctx.extra_data.get("survey_responses", [])
            if survey:
                has_data = True
            for item in survey:
                qid = item.get("question_id")
                response = item.get("response")
                if qid is None or response is None:
                    continue
                all_responses.setdefault(qid, []).append(response)

        if not has_data:
            return {"entrenchment_skipped": True}

        entrenched_questions = []
        for qid, responses in all_responses.items():
            if len(responses) < 3:
                continue
            counter = Counter(responses)
            mode_fraction = counter.most_common(1)[0][1] / len(responses)
            if mode_fraction >= MODAL_RESPONSE_THRESHOLD:
                entrenched_questions.append({
                    "question_id": qid,
                    "modal_response": counter.most_common(1)[0][0],
                    "modal_fraction": round(mode_fraction, 4),
                })

        total_questions = len([q for q, r in all_responses.items() if len(r) >= 3])
        entrenchment_rate = len(entrenched_questions) / total_questions if total_questions > 0 else 0.0

        return {
            "entrenchment_rate": round(entrenchment_rate, 4),
            "entrenched_questions": entrenched_questions,
            "total_survey_questions": total_questions,
            "entrenchment_threshold": ENTRENCHMENT_THRESHOLD,
        }

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D13 is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        if len(personas) < 20:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True,
                score=1.0,
                details={"skipped": True, "reason": "Need >= 20 for reliable entropy estimation"},
            )]

        variation_ratios: dict[str, float] = {}
        entropies: dict[str, float] = {}
        modal_collapse_fields: list[dict] = []

        for field_name, extractor in DIVERSITY_FIELDS:
            values = []
            for p in personas:
                val = extractor(p)
                if val and val.strip():
                    values.append(val)

            if len(values) < 2:
                continue

            vr = _variation_ratio(values)
            ent = _shannon_entropy(values)

            variation_ratios[field_name] = round(vr, 4)
            entropies[field_name] = round(ent, 4)

            counter = Counter(values)
            mode_count = counter.most_common(1)[0][1]
            if mode_count / len(values) > MODAL_COLLAPSE_THRESHOLD:
                modal_collapse_fields.append({
                    "field": field_name,
                    "modal_value": counter.most_common(1)[0][0],
                    "modal_fraction": round(mode_count / len(values), 4),
                })

        if not variation_ratios:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True,
                score=1.0,
                details={"skipped": True, "reason": "No comparable fields"},
            )]

        mean_entropy = sum(entropies.values()) / len(entropies)
        mean_vr = sum(variation_ratios.values()) / len(variation_ratios)
        score = (mean_entropy + mean_vr) / 2

        passed = len(modal_collapse_fields) == 0 and score >= 0.3

        details: dict[str, Any] = {
            "variation_ratios": variation_ratios,
            "entropies": entropies,
            "mean_entropy": round(mean_entropy, 4),
            "mean_variation_ratio": round(mean_vr, 4),
            "modal_collapse_fields": modal_collapse_fields,
            "persona_count": len(personas),
        }

        entrenchment = self._detect_entrenchment(source_contexts)
        details.update(entrenchment)

        if entrenchment.get("entrenchment_rate", 0) >= ENTRENCHMENT_THRESHOLD:
            passed = False

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details=details,
        )]
