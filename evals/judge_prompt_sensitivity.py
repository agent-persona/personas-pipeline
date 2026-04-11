"""Experiment 5.12: Judge prompt sensitivity."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from statistics import mean, pstdev

DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)

DIMENSION_BLURBS = {
    "grounded": "claims trace cleanly to evidence and cited records",
    "distinctive": "this feels like a specific person, not a generic average",
    "coherent": "fields and voice fit together without contradiction",
    "actionable": "the persona produces usable product or go-to-market guidance",
    "voice_fidelity": "quotes and vocabulary sound like one consistent speaker",
}


@dataclass
class VariantScore:
    persona_id: str
    variant: str
    scores: dict[str, float]
    overall: float
    rationale: str


@dataclass
class SensitivitySummary:
    n_personas: int
    n_variants: int
    per_persona_overall_variance: dict[str, float]
    per_dimension_cv: dict[str, float]
    variant_mean_overall: dict[str, float]
    most_sensitive_dimension: str
    least_sensitive_dimension: str


VARIANT_SPECS = {
    "formal_baseline": {
        "tone": "formal",
        "numbered": False,
        "negative": False,
        "terse": False,
        "order": DIMENSIONS,
    },
    "casual": {
        "tone": "casual",
        "numbered": False,
        "negative": False,
        "terse": False,
        "order": DIMENSIONS,
    },
    "numbered_checklist": {
        "tone": "formal",
        "numbered": True,
        "negative": False,
        "terse": False,
        "order": DIMENSIONS,
    },
    "negative_framing": {
        "tone": "formal",
        "numbered": False,
        "negative": True,
        "terse": False,
        "order": DIMENSIONS,
    },
    "reordered": {
        "tone": "formal",
        "numbered": False,
        "negative": False,
        "terse": False,
        "order": ("voice_fidelity", "actionable", "coherent", "distinctive", "grounded"),
    },
    "terse": {
        "tone": "formal",
        "numbered": False,
        "negative": False,
        "terse": True,
        "order": DIMENSIONS,
    },
}


def _parse_response(text: str) -> tuple[dict[str, float], float, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return ({dim: float("nan") for dim in DIMENSIONS}, float("nan"), cleaned[:240])
        data = json.loads(match.group())
    scores = {dim: float(data.get(dim, float("nan"))) for dim in DIMENSIONS}
    return scores, float(data.get("overall", float("nan"))), str(data.get("rationale", ""))


def build_system_prompt(variant: str) -> str:
    spec = VARIANT_SPECS[variant]
    lines = []
    if spec["tone"] == "casual":
        lines.append(
            "You're a sharp reviewer scoring customer personas. Keep the reasoning tight and consistent."
        )
    else:
        lines.append(
            "You are an expert persona evaluator scoring synthesized customer personas."
        )
    lines.append("Use a 1-5 scale where 1 is very poor and 5 is excellent.")
    lines.append("")
    lines.append("Dimensions:")
    for idx, dim in enumerate(spec["order"], start=1):
        prefix = f"{idx}. " if spec["numbered"] else "- "
        if spec["negative"]:
            text = f"penalize when {DIMENSION_BLURBS[dim]} is missing"
        elif spec["terse"]:
            text = DIMENSION_BLURBS[dim]
        else:
            text = f"score high when {DIMENSION_BLURBS[dim]}"
        lines.append(f"{prefix}{dim}: {text}")
    lines.extend(
        [
            "",
            "Respond with JSON only:",
            "{",
            '  "grounded": <1-5>,',
            '  "distinctive": <1-5>,',
            '  "coherent": <1-5>,',
            '  "actionable": <1-5>,',
            '  "voice_fidelity": <1-5>,',
            '  "overall": <1-5>,',
            '  "rationale": "short justification"',
            "}",
        ]
    )
    return "\n".join(lines)


def build_prompt(persona: dict) -> str:
    return "Score this persona.\n\nPERSONA:\n" + json.dumps(persona, indent=2, default=str)


async def score_variant(backend, persona_id: str, persona: dict, variant: str) -> VariantScore:
    response = await backend.score(
        system=build_system_prompt(variant),
        prompt=build_prompt(persona),
    )
    scores, overall, rationale = _parse_response(response)
    return VariantScore(
        persona_id=persona_id,
        variant=variant,
        scores=scores,
        overall=overall,
        rationale=rationale,
    )


def summarize_scores(results: list[VariantScore]) -> SensitivitySummary:
    persona_ids = sorted({result.persona_id for result in results})
    per_persona_overall_variance: dict[str, float] = {}
    for persona_id in persona_ids:
        values = [result.overall for result in results if result.persona_id == persona_id]
        avg = mean(values) if values else 0.0
        per_persona_overall_variance[persona_id] = mean(
            (value - avg) ** 2 for value in values
        ) if values else 0.0

    per_dimension_cv: dict[str, float] = {}
    for dim in DIMENSIONS:
        values = [result.scores[dim] for result in results]
        avg = mean(values) if values else 0.0
        per_dimension_cv[dim] = (pstdev(values) / avg) if values and avg else 0.0

    variant_mean_overall = {
        variant: mean(result.overall for result in results if result.variant == variant)
        for variant in VARIANT_SPECS
    }

    most_sensitive_dimension = max(per_dimension_cv, key=per_dimension_cv.get)
    least_sensitive_dimension = min(per_dimension_cv, key=per_dimension_cv.get)

    return SensitivitySummary(
        n_personas=len(persona_ids),
        n_variants=len(VARIANT_SPECS),
        per_persona_overall_variance=per_persona_overall_variance,
        per_dimension_cv=per_dimension_cv,
        variant_mean_overall=variant_mean_overall,
        most_sensitive_dimension=most_sensitive_dimension,
        least_sensitive_dimension=least_sensitive_dimension,
    )


def results_to_dict(results: list[VariantScore], summary: SensitivitySummary) -> dict:
    return {
        "results": [asdict(result) for result in results],
        "summary": asdict(summary),
    }
