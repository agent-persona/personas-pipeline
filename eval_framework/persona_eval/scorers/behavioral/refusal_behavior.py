"""D29 Refusal Behavior — out-of-scope question handling quality.

Trustworthiness: MEDIUM-HIGH (embedding-based uncertainty detection).
Method: Build an uncertainty prototype vector from canonical uncertain phrases,
then measure cosine similarity of each response to that prototype.
OOD responses similar to prototype = appropriate refusal.
In-scope responses dissimilar from prototype = appropriate confidence.
Falls back to keyword matching if embedder unavailable.
Expects source_context.extra_data["scope_responses"]:
    list of {"question": str, "response": str, "in_scope": bool}.
"""

from __future__ import annotations

import re

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

UNCERTAINTY_MARKERS = [
    re.compile(r"\b(I'?m not sure|I don'?t (really )?know|outside my expertise)\b", re.IGNORECASE),
    re.compile(r"\b(not (really )?my (area|field)|can'?t speak to|don'?t have experience)\b", re.IGNORECASE),
    re.compile(r"\b(I'?d have to|you'?d (need|want) to ask|I'?m not (qualified|the right person))\b", re.IGNORECASE),
]

# Canonical phrases that represent uncertainty / appropriate epistemic humility
_UNCERTAINTY_PHRASES = [
    "I'm not sure about that",
    "That's outside my area of expertise",
    "I don't really know enough about this",
    "I'm not qualified to answer that",
    "You'd want to ask a specialist",
    "I can't speak to that topic",
    "I don't have experience in that field",
    "That's not really my area",
    "I lack sufficient background to answer reliably",
    "I wouldn't feel comfortable advising on that",
    "I'm not the right person to ask about that",
    "I'd have to defer to an expert on this",
]

# Cosine sim thresholds
UNCERTAINTY_SIM_THRESHOLD = 0.55   # OOD response at or above → appropriate refusal
CONFIDENCE_SIM_THRESHOLD = 0.40    # in-scope response below → appropriate confidence

# Pass if this fraction of OOS questions get appropriate refusals
REFUSAL_THRESHOLD = 0.5


def _get_embedder():
    from persona_eval.embeddings import Embedder
    return Embedder()


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


def _centroid(vecs: list[list[float]]) -> list[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    total = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            total[i] += x
    n = len(vecs)
    return [x / n for x in total]


class RefusalBehaviorScorer(BaseScorer):
    """Evaluates quality of refusal on out-of-scope questions."""

    dimension_id = "D29"
    dimension_name = "Refusal Behavior"
    tier = 5
    requires_set = False

    def __init__(self) -> None:
        super().__init__()
        self._uncertainty_proto: list[float] | None = None

    def _build_uncertainty_proto(self, embedder) -> list[float]:
        """Compute and cache the normalized uncertainty prototype vector."""
        if self._uncertainty_proto is not None:
            return self._uncertainty_proto
        phrase_vecs = embedder.embed_batch(_UNCERTAINTY_PHRASES)
        centroid = _centroid(phrase_vecs)
        self._uncertainty_proto = _normalize(centroid)
        return self._uncertainty_proto

    def _score_embedding(
        self, persona: Persona, responses: list[dict], oos_items: list[dict]
    ) -> EvalResult:
        embedder = _get_embedder()
        proto = self._build_uncertainty_proto(embedder)

        in_scope_items = [r for r in responses if r.get("in_scope", True)]

        # Embed all response texts in one batch for efficiency
        all_items = oos_items + in_scope_items
        all_texts = [item.get("response", "") for item in all_items]
        all_vecs = embedder.embed_batch(all_texts)

        oos_details = []
        appropriate_oos = 0
        for i, item in enumerate(oos_items):
            vec = _normalize(all_vecs[i])
            # proto is already normalized; dot product = cosine similarity
            sim = sum(a * b for a, b in zip(vec, proto))
            appropriate = sim >= UNCERTAINTY_SIM_THRESHOLD
            if appropriate:
                appropriate_oos += 1
            resp = item.get("response", "")
            oos_details.append({
                "response": resp[:80],
                "sim": round(sim, 4),
                "appropriate": appropriate,
            })

        appropriate_in_scope = 0
        for j, item in enumerate(in_scope_items):
            idx = len(oos_items) + j
            vec = _normalize(all_vecs[idx])
            sim = sum(a * b for a, b in zip(vec, proto))
            if sim < CONFIDENCE_SIM_THRESHOLD:
                appropriate_in_scope += 1

        total = len(responses)
        oos_count = len(oos_items)
        in_scope_count = len(in_scope_items)
        oos_refusal_rate = appropriate_oos / oos_count if oos_count > 0 else 1.0
        combined_appropriate = appropriate_oos + appropriate_in_scope
        score = combined_appropriate / total if total > 0 else 1.0
        passed = oos_refusal_rate >= REFUSAL_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "method": "embedding",
                "appropriate_refusal_rate": round(oos_refusal_rate, 4),
                "oos_refusal_rate": round(oos_refusal_rate, 4),
                "oos_count": oos_count,
                "appropriate_oos": appropriate_oos,
                "appropriate_refusals": appropriate_oos,
                "appropriate_in_scope": appropriate_in_scope,
                "in_scope_count": in_scope_count,
                "total_responses": total,
                "oos_details": oos_details,
            },
        )

    def _score_keyword(
        self, persona: Persona, responses: list[dict], oos_items: list[dict],
        fallback_reason: str | None = None,
    ) -> EvalResult:
        appropriate_count = 0
        for item in oos_items:
            response = item.get("response", "")
            has_uncertainty = any(m.search(response) for m in UNCERTAINTY_MARKERS)
            if has_uncertainty:
                appropriate_count += 1

        refusal_rate = appropriate_count / len(oos_items)
        passed = refusal_rate >= REFUSAL_THRESHOLD

        details = {
            "method": "keyword_fallback",
            "appropriate_refusal_rate": round(refusal_rate, 4),
            "oos_refusal_rate": round(refusal_rate, 4),
            "oos_count": len(oos_items),
            "appropriate_oos": appropriate_count,
            "appropriate_refusals": appropriate_count,
            "total_responses": len(responses),
        }
        if fallback_reason is not None:
            details["fallback_reason"] = fallback_reason

        return self._result(
            persona, passed=passed, score=round(refusal_rate, 4),
            details=details,
        )

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        responses: list[dict] = source_context.extra_data.get("scope_responses", [])
        if not responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No scope_responses in extra_data"},
            )

        oos_items = [r for r in responses if not r.get("in_scope", True)]
        if not oos_items:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No out-of-scope questions"},
            )

        try:
            return self._score_embedding(persona, responses, oos_items)
        except Exception as exc:
            return self._score_keyword(
                persona, responses, oos_items,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
