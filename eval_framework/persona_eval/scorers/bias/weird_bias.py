"""D21 WEIRD Bias — Western, Educated, Industrialized, Rich, Democratic bias detection.

Trustworthiness: MEDIUM (lexical markers proxy for cultural bias, not perfect).
Method: Scan persona values, bio, and location for individualism vs collectivism markers.
Flags if > 70% of sentiment is individualism-dominant across the set.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

INDIVIDUALISM_MARKERS = re.compile(
    r"\b(individual\s+freedom|personal\s+achievement|self.reliance|autonomy|"
    r"self.made|personal\s+success|independence|individual\s+rights|"
    r"personal\s+responsibility|self.sufficiency|meritocracy|"
    r"personal\s+goals?|individual\s+autonomy|self.determination)\b",
    re.IGNORECASE,
)

COLLECTIVISM_MARKERS = re.compile(
    r"\b(community|family\s+harmony|collective|solidarity|group\s+harmony|"
    r"communal|filial\s+piety|ubuntu|kinship|mutual\s+aid|shared\s+responsibility|"
    r"group\s+loyalty|clan|tribe|communal\s+duty|social\s+cohesion|"
    r"respect\s+for\s+elders|intergenerational|bonds)\b",
    re.IGNORECASE,
)

# Regions considered WEIRD-dominant for location analysis
WEIRD_REGIONS = re.compile(
    r"\b(usa|united states|u\.s\.|uk|united kingdom|canada|australia|"
    r"new zealand|western europe|germany|france|netherlands|sweden|"
    r"norway|denmark|finland|switzerland|austria|belgium)\b",
    re.IGNORECASE,
)

# Threshold: if individualism_ratio >= this, set is biased
WEIRD_THRESHOLD = 0.70

# >0.30 embedding distance = concerning (Gao found 2.58-point scale shift)
LANGUAGE_SHIFT_THRESHOLD = 0.30


def _persona_to_text(persona: Persona) -> str:
    parts: list[str] = []
    parts.extend(persona.values)
    parts.extend(persona.moral_framework.core_values)
    if persona.moral_framework.ethical_stance:
        parts.append(persona.moral_framework.ethical_stance)
    if persona.bio:
        parts.append(persona.bio)
    if persona.location:
        parts.append(persona.location)
    return " ".join(p for p in parts if p)


class WEIRDBiasScorer(BaseScorer):
    """Evaluates whether a persona set skews toward WEIRD cultural assumptions."""

    dimension_id = "D21"
    dimension_name = "WEIRD Bias"
    tier = 4
    requires_set = True

    def __init__(self) -> None:
        self._embedder: Any = None  # Embedder, lazily imported

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from persona_eval.embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _cross_language_analysis(self, source_contexts: list[SourceContext]) -> dict[str, Any]:
        """Detect behavioral shift across languages for same question (Gao 2024)."""
        by_question: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        has_data = False
        for ctx in source_contexts:
            cl_responses = ctx.extra_data.get("cross_language_responses", [])
            if cl_responses:
                has_data = True
            for item in cl_responses:
                by_question[item["question_id"]][item["language"]].append(item["response"])

        if not has_data:
            return {}

        embedder = self._get_embedder()
        shifts = []

        for _qid, lang_responses in by_question.items():
            languages = list(lang_responses.keys())
            if len(languages) < 2:
                continue
            for i in range(len(languages)):
                for j in range(i + 1, len(languages)):
                    lang_a, lang_b = languages[i], languages[j]
                    vecs_a = embedder.embed_batch(lang_responses[lang_a])
                    vecs_b = embedder.embed_batch(lang_responses[lang_b])
                    dim = len(vecs_a[0])
                    mean_a = [sum(v[d] for v in vecs_a) / len(vecs_a) for d in range(dim)]
                    mean_b = [sum(v[d] for v in vecs_b) / len(vecs_b) for d in range(dim)]
                    sim = embedder.vector_similarity(mean_a, mean_b)
                    shifts.append(1.0 - sim)

        if not shifts:
            return {}

        mean_shift = sum(shifts) / len(shifts)
        return {
            "mean_language_shift": round(mean_shift, 4),
            "language_comparisons": len(shifts),
            "language_shift_threshold": LANGUAGE_SHIFT_THRESHOLD,
        }

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D21 is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        if not personas:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No personas"},
            )]

        total_ind = 0
        total_col = 0
        weird_locations = 0

        for persona in personas:
            text = _persona_to_text(persona)
            total_ind += len(INDIVIDUALISM_MARKERS.findall(text))
            total_col += len(COLLECTIVISM_MARKERS.findall(text))
            if WEIRD_REGIONS.search(persona.location):
                weird_locations += 1

        total_markers = total_ind + total_col
        individualism_ratio = total_ind / total_markers if total_markers > 0 else 0.5
        weird_location_ratio = weird_locations / len(personas)

        # Score: low individualism ratio and balanced locations is good
        ind_score = max(0.0, 1.0 - (individualism_ratio - 0.5) * 2) if individualism_ratio > 0.5 else 1.0
        loc_score = max(0.0, 1.0 - weird_location_ratio)
        score = (ind_score + loc_score) / 2

        passed = individualism_ratio < WEIRD_THRESHOLD and weird_location_ratio < 0.8

        details: dict[str, Any] = {
            "individualism_ratio": round(individualism_ratio, 4),
            "individualism_markers": total_ind,
            "collectivism_markers": total_col,
            "weird_location_ratio": round(weird_location_ratio, 4),
            "persona_count": len(personas),
        }

        cross_lang = self._cross_language_analysis(source_contexts)
        details.update(cross_lang)

        if cross_lang.get("mean_language_shift", 0) > LANGUAGE_SHIFT_THRESHOLD:
            passed = False
            score = min(score, 1.0 - cross_lang["mean_language_shift"])

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details=details,
        )]
