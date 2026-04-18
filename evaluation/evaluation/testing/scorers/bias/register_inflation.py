"""D45 Register Inflation — vocabulary register vs. stated communication style.

Trustworthiness: MEDIUM-HIGH (embedding similarity to register prototypes).
Method: Compare each response's embedding to basic vs. advanced register prototypes.
Flag when a persona with a low vocabulary_level produces advanced-register prose.

Evidence: LLMs cannot write below their training distribution — a "basic vocabulary"
persona still produces doctoral-register prose because the model defaults to its own
register regardless of the persona's stated vocabulary level.
"""

from __future__ import annotations

from typing import Any

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# ---------------------------------------------------------------------------
# Register prototype texts
# ---------------------------------------------------------------------------

BASIC_REGISTER_TEXTS = [
    "I like this a lot.",
    "This is good stuff.",
    "Yeah that works for me.",
    "I just do what I know.",
    "Pretty simple really.",
    "That's how I see it.",
    "Makes sense to me.",
    "I don't really get the fancy stuff.",
    "Just tell me what to do.",
    "It works, that's all I care about.",
]

ADVANCED_REGISTER_TEXTS = [
    "The methodology demonstrates robust empirical validity across multiple contexts.",
    "Consequently, the paradigmatic implications necessitate further investigation.",
    "The theoretical framework encompasses multiple interrelated variables.",
    "One must consider the systemic implications of this multifaceted approach.",
    "The nuanced interplay between these factors warrants careful consideration.",
    "This phenomenon exhibits considerable heterogeneity across sociocultural contexts.",
    "The empirical evidence substantiates the underlying theoretical proposition.",
    "Epistemologically, this approach conflates distinct categories of ontological analysis.",
    "The operationalization of these constructs requires methodological rigor.",
    "The heuristic framework provides a rubric for evaluating emergent phenomena.",
]

# vocabulary_level → expected maximum register_ratio before flagging as inflated
EXPECTED_MAX_BY_LEVEL: dict[str, float] = {
    "basic": 0.40,
    "intermediate": 0.55,
}

_embedder_instance: Any = None


def _get_embedder() -> Any:
    global _embedder_instance
    if _embedder_instance is None:
        from evaluation.testing.embeddings import Embedder
        _embedder_instance = Embedder()
    return _embedder_instance


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


def _centroid(vecs: list[list[float]]) -> list[float]:
    dim = len(vecs[0])
    result = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            result[i] += x
    n = len(vecs)
    return [x / n for x in result]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class RegisterInflationScorer(BaseScorer):
    """Evaluates whether persona responses are inflated above the stated vocabulary level."""

    dimension_id = "D45"
    dimension_name = "Register Inflation"
    tier = 4
    requires_set = False

    def __init__(self) -> None:
        self._basic_proto: list[float] | None = None
        self._advanced_proto: list[float] | None = None

    def _build_prototypes(self) -> None:
        if self._basic_proto is not None:
            return
        embedder = _get_embedder()
        basic_vecs = embedder.embed_batch(BASIC_REGISTER_TEXTS)
        advanced_vecs = embedder.embed_batch(ADVANCED_REGISTER_TEXTS)
        self._basic_proto = _normalize(_centroid(basic_vecs))
        self._advanced_proto = _normalize(_centroid(advanced_vecs))

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        vocab_level = (persona.communication_style.vocabulary_level or "").lower()

        # Only score constrained vocabulary levels; skip advanced/technical/expert
        expected_max = EXPECTED_MAX_BY_LEVEL.get(vocab_level)
        if expected_max is None:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": f"vocabulary_level='{vocab_level}' has no constraint"},
            )

        # Get responses from extra_data
        responses: list[str] = source_context.extra_data.get("register_responses") or \
                                source_context.extra_data.get("responses") or []
        responses = [r for r in responses if isinstance(r, str) and r.strip()]

        if not responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No responses to evaluate"},
            )

        self._build_prototypes()
        embedder = _get_embedder()
        response_vecs = embedder.embed_batch(responses)

        register_ratios: list[float] = []
        for vec in response_vecs:
            norm_vec = _normalize(vec)
            basic_sim = _dot(norm_vec, self._basic_proto)   # type: ignore[arg-type]
            advanced_sim = _dot(norm_vec, self._advanced_proto)   # type: ignore[arg-type]
            denom = basic_sim + advanced_sim
            ratio = advanced_sim / denom if denom > 1e-10 else 0.5
            register_ratios.append(ratio)

        avg_ratio = sum(register_ratios) / len(register_ratios)
        inflation = max(0.0, avg_ratio - expected_max)
        score = max(0.0, 1.0 - inflation * 2.5)
        passed = avg_ratio <= expected_max

        return self._result(
            persona,
            passed=passed,
            score=round(score, 4),
            details={
                "register_ratio": round(avg_ratio, 4),
                "expected_max": expected_max,
                "vocabulary_level": vocab_level,
                "inflation": round(inflation, 4),
                "response_count": len(responses),
            },
        )
