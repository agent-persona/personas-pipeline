"""Shared metrics for all lab experiments.

Every experiment in `PRD_LAB_RESEARCH.md` declares a metric from this list.
Adding a new metric? Put it here so every problem space can pick it up
without re-implementing it.

This module is a skeleton — most functions return a placeholder. Researcher
#5 (problem space 5) owns the real implementations. The signatures below
are stable: other researchers can import them today and will get real
numbers once the bodies land.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

DEFAULT_STABILITY_FIELDS = [
    "name",
    "summary",
    "goals",
    "pains",
    "motivations",
    "objections",
]

FIELD_INTERDEPENDENCE_RULES = {
    "role_goal_alignment": {
        "anchor_paths": (
            "summary",
            "firmographics.industry",
            "firmographics.role_titles",
        ),
        "dependent_paths": (
            "goals",
            "pains",
            "motivations",
            "objections",
            "decision_triggers",
        ),
    },
    "tech_voice_alignment": {
        "anchor_paths": (
            "summary",
            "firmographics.tech_stack_signals",
        ),
        "dependent_paths": (
            "vocabulary",
            "sample_quotes",
            "channels",
        ),
    },
    "journey_action_alignment": {
        "anchor_paths": (
            "journey_stages.stage",
            "journey_stages.key_actions",
            "journey_stages.content_preferences",
        ),
        "dependent_paths": (
            "goals",
            "decision_triggers",
            "channels",
        ),
    },
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "with",
}

_AGE_BAND_RULES = (
    {"min": 0, "max": 5, "sentence_range": (0.0, 8.0), "long_word_ratio": 0.06},
    {"min": 6, "max": 12, "sentence_range": (4.0, 14.0), "long_word_ratio": 0.10},
    {"min": 13, "max": 17, "sentence_range": (6.0, 20.0), "long_word_ratio": 0.16},
    {"min": 18, "max": 25, "sentence_range": (8.0, 24.0), "long_word_ratio": 0.22},
    {"min": 26, "max": 60, "sentence_range": (8.0, 28.0), "long_word_ratio": 0.26},
    {"min": 61, "max": 120, "sentence_range": (6.0, 24.0), "long_word_ratio": 0.24},
)

_WORK_CONTEXT_TERMS = {
    "budget",
    "buyer",
    "career",
    "client",
    "customers",
    "demo",
    "deployment",
    "engineering",
    "executive",
    "finance",
    "pipeline",
    "procurement",
    "quarter",
    "revenue",
    "roadmap",
    "roi",
    "sales",
    "stakeholder",
    "vendor",
}

_SENIORITY_TERMS = {
    "architect",
    "chief",
    "cto",
    "director",
    "executive",
    "founder",
    "head",
    "lead",
    "manager",
    "principal",
    "senior",
    "staff",
    "vp",
}

_CAPABILITY_LEVEL_FIELDS = (
    "factual_knowledge",
    "procedural_skill",
    "taste_judgment",
    "creativity",
    "speed",
    "consistency",
    "error_recovery",
    "teaching_ability",
    "tool_fluency",
    "confidence_calibration",
)

_RECENCY_SPEED_CAPS = {
    "active": 5.0,
    "rusty": 4.0,
    "dormant": 3.2,
    "atrophied": 2.6,
}

_STAGE_SKILL_WINDOWS = {
    1: (0.0, 1.8),
    2: (0.5, 3.1),
    3: (1.8, 4.4),
    4: (3.1, 5.0),
}

_BEHAVIOR_MODIFIER_KEYS = {
    "warmth",
    "dominance",
    "disclosure",
    "disclosure_level",
    "self_monitoring",
    "spontaneity",
    "trust_level",
    "vulnerability",
    "reputation_concern",
    "teaching_quality_modifier",
    "cognitive_load_modifier",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mean_defined(values: Sequence[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _score_range(value: float, lower: float, upper: float, tolerance: float = 1.0) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return _clamp(1.0 - ((lower - value) / max(tolerance, 1e-6)))
    return _clamp(1.0 - ((value - upper) / max(tolerance, 1e-6)))


def _parse_age_range(age_range: str | None) -> tuple[int, int] | None:
    if not isinstance(age_range, str):
        return None
    match = re.fullmatch(r"\s*(\d{1,3})-(\d{1,3})\s*", age_range)
    if not match:
        return None
    lower, upper = int(match.group(1)), int(match.group(2))
    if lower > upper:
        return None
    return lower, upper


def _persona_age(persona: dict) -> int | None:
    age = persona.get("age")
    if isinstance(age, int):
        return age
    demographics = persona.get("demographics")
    age_range = None
    if isinstance(demographics, dict):
        age_range = demographics.get("age_range")
    if age_range is None:
        age_range = persona.get("age_range")
    parsed = _parse_age_range(age_range)
    if parsed is None:
        return None
    lower, upper = parsed
    return (lower + upper) // 2


def _age_range_string(persona: dict) -> str | None:
    age_range = persona.get("age_range")
    if isinstance(age_range, str):
        return age_range
    demographics = persona.get("demographics")
    if isinstance(demographics, dict):
        nested = demographics.get("age_range")
        if isinstance(nested, str):
            return nested
    return None


def _iter_strings(value) -> list[str]:
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, str):
        items.append(value)
        return items
    if isinstance(value, list):
        for item in value:
            items.extend(_iter_strings(item))
        return items
    if isinstance(value, dict):
        for nested in value.values():
            items.extend(_iter_strings(nested))
        return items
    return items


def _persona_text_fragments(persona: dict) -> list[str]:
    fragments: list[str] = []
    fields = (
        "name",
        "summary",
        "goals",
        "pains",
        "motivations",
        "objections",
        "channels",
        "vocabulary",
        "decision_triggers",
        "sample_quotes",
    )
    for field_name in fields:
        fragments.extend(_iter_strings(persona.get(field_name)))
    demographics = persona.get("demographics")
    if isinstance(demographics, dict):
        fragments.extend(_iter_strings(demographics))
    firmographics = persona.get("firmographics")
    if isinstance(firmographics, dict):
        fragments.extend(_iter_strings(firmographics))
    return fragments


def _persona_text_blob(persona: dict) -> str:
    return "\n".join(_persona_text_fragments(persona)).lower()


def _phrase_hits(text: str, phrases: Sequence[str]) -> int:
    normalized = text.lower()
    hits = 0
    for phrase in phrases:
        cleaned = phrase.strip().lower()
        if cleaned and cleaned in normalized:
            hits += 1
    return hits


def _token_hits(tokens: set[str], phrases: Sequence[str]) -> int:
    hits = 0
    for phrase in phrases:
        phrase_tokens = _normalize_tokens(phrase)
        if phrase_tokens and phrase_tokens <= tokens:
            hits += 1
    return hits


def _age_band_rule(age: int) -> dict[str, float]:
    for rule in _AGE_BAND_RULES:
        if rule["min"] <= age <= rule["max"]:
            return rule
    return _AGE_BAND_RULES[-1]


def _quote_stats(persona: dict) -> tuple[float, float]:
    quotes = persona.get("sample_quotes")
    if not isinstance(quotes, list) or not quotes:
        return float("nan"), float("nan")
    lengths: list[float] = []
    long_word_ratios: list[float] = []
    for quote in quotes:
        if not isinstance(quote, str):
            continue
        words = re.findall(r"[A-Za-z0-9']+", quote)
        if not words:
            continue
        lengths.append(float(len(words)))
        long_words = sum(1 for word in words if len(word) >= 8)
        long_word_ratios.append(long_words / len(words))
    return _mean_defined(lengths), _mean_defined(long_word_ratios)


def schema_validity(persona_dicts: Sequence[dict], schema_cls) -> float:
    """Fraction of persona dicts that validate against `schema_cls`.

    Cheap, deterministic, no LLM calls.
    """
    from pydantic import ValidationError

    if not persona_dicts:
        return 1.0
    ok = 0
    for p in persona_dicts:
        try:
            schema_cls.model_validate(p)
            ok += 1
        except ValidationError:
            pass
    return ok / len(persona_dicts)


def groundedness_rate(reports: Sequence) -> float:
    """Mean groundedness score across a batch of `GroundednessReport`s.

    Expects the `GroundednessReport` type from
    `synthesis.engine.groundedness` (duck-typed: anything with a `.score`
    attribute works).
    """
    if not reports:
        return 1.0
    return sum(r.score for r in reports) / len(reports)


def distinctiveness(persona_embeddings: Sequence[Sequence[float]]) -> float:
    """Mean pairwise cosine distance across a set of persona embeddings.

    1.0 = maximally distinct, 0.0 = identical. Used by space 6 (distinctiveness
    floor) and space 4 (drift: did turn N drift toward turn 1 or away from it).
    """
    n = len(persona_embeddings)
    if n < 2:
        return float("nan")

    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a = persona_embeddings[i]
            b = persona_embeddings[j]
            dot = sum(ai * bi for ai, bi in zip(a, b))
            norm_a = math.sqrt(sum(ai * ai for ai in a))
            norm_b = math.sqrt(sum(bi * bi for bi in b))
            if norm_a == 0 or norm_b == 0:
                distances.append(1.0)
            else:
                sim = dot / (norm_a * norm_b)
                sim = max(-1.0, min(1.0, sim))
                distances.append(1.0 - sim)

    return sum(distances) / len(distances)


def cost_per_persona(total_cost_usd: float, n_personas: int) -> float:
    """Simple dollars-per-persona throughput metric."""
    if n_personas == 0:
        return 0.0
    return total_cost_usd / n_personas


def _population_stdev(values: Sequence[float]) -> float:
    """Population standard deviation for already-clean numeric values."""
    if len(values) < 2:
        return float("nan")
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def summarize_metric_runs(metric_runs: Sequence[dict[str, float | None]]) -> dict[str, object]:
    """Summarize a list of per-run metric dictionaries.

    The result is intentionally JSON-friendly so runners can write it straight
    into baseline artifacts without another transformation layer.
    """
    metric_names = sorted({metric for run in metric_runs for metric in run.keys()})
    means: dict[str, float | None] = {}
    stdevs: dict[str, float | None] = {}
    mins: dict[str, float | None] = {}
    maxs: dict[str, float | None] = {}
    counts: dict[str, int] = {}

    for metric_name in metric_names:
        values: list[float] = []
        for run in metric_runs:
            value = run.get(metric_name)
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            values.append(float(value))

        counts[metric_name] = len(values)
        if not values:
            means[metric_name] = None
            stdevs[metric_name] = None
            mins[metric_name] = None
            maxs[metric_name] = None
            continue

        means[metric_name] = sum(values) / len(values)
        stdev = _population_stdev(values)
        stdevs[metric_name] = None if math.isnan(stdev) else stdev
        mins[metric_name] = min(values)
        maxs[metric_name] = max(values)

    return {
        "num_runs": len(metric_runs),
        "means": means,
        "stdevs": stdevs,
        "mins": mins,
        "maxs": maxs,
        "counts": counts,
    }


def developmental_fit(persona: dict) -> float:
    """Heuristic age realism score across V1 and V2 persona shapes."""
    age_range = _age_range_string(persona)
    if age_range is None:
        return 0.0
    parsed = _parse_age_range(age_range)
    if parsed is None:
        return 0.0

    age = _persona_age(persona)
    if age is None:
        return 1.0

    lower, upper = parsed
    range_score = 1.0 if lower <= age <= upper else 0.0

    rule = _age_band_rule(age)
    mean_sentence_length, long_word_ratio = _quote_stats(persona)
    quote_length_score = (
        0.75
        if math.isnan(mean_sentence_length)
        else _score_range(
            mean_sentence_length,
            rule["sentence_range"][0],
            rule["sentence_range"][1],
            tolerance=8.0,
        )
    )
    long_word_score = (
        0.75
        if math.isnan(long_word_ratio)
        else _score_range(
            long_word_ratio,
            0.0,
            rule["long_word_ratio"],
            tolerance=0.12,
        )
    )
    quote_score = (quote_length_score + long_word_score) / 2.0

    persona_tokens = _values_to_token_set(_persona_text_fragments(persona))
    work_hits = len(persona_tokens & _WORK_CONTEXT_TERMS)
    senior_hits = len(persona_tokens & _SENIORITY_TERMS)
    stage_score = 1.0
    if age < 13:
        stage_score = _clamp(1.0 - (0.25 * work_hits) - (0.45 * senior_hits))
    elif age < 18:
        stage_score = _clamp(1.0 - (0.15 * work_hits) - (0.35 * senior_hits))
    elif age < 25 and senior_hits:
        stage_score = _clamp(1.0 - (0.18 * senior_hits))

    return _clamp((0.45 * range_score) + (0.35 * quote_score) + (0.20 * stage_score))


def historical_fit(persona: dict) -> float:
    """Heuristic historical/cohort plausibility score for V2 personas."""
    birth_year = persona.get("birth_year")
    eval_year = persona.get("eval_year")
    if not isinstance(birth_year, int) or not isinstance(eval_year, int):
        return float("nan")

    age = _persona_age(persona)
    if age is None:
        age_consistency = 0.6
    else:
        expected_age = max(0, eval_year - birth_year)
        age_consistency = _clamp(1.0 - (abs(expected_age - age) / 8.0))

    cohort = persona.get("cohort")
    snapshot = persona.get("tech_familiarity_snapshot")
    snapshot_alignment = 0.7
    if isinstance(cohort, dict):
        cohort_tech = cohort.get("tech_familiarity", {})
        if isinstance(cohort_tech, dict) and isinstance(snapshot, dict):
            comparisons: list[float] = []
            for field_name in ("grew_up_with", "adopted_as_adult", "never_used"):
                cohort_values = cohort_tech.get(field_name, [])
                snapshot_values = snapshot.get(field_name, [])
                if isinstance(cohort_values, list) and isinstance(snapshot_values, list):
                    left = {str(item).lower() for item in cohort_values}
                    right = {str(item).lower() for item in snapshot_values}
                    if not left and not right:
                        comparisons.append(1.0)
                    elif left or right:
                        comparisons.append(len(left & right) / len(left | right))
            if comparisons:
                snapshot_alignment = sum(comparisons) / len(comparisons)

    text_blob = _persona_text_blob(persona)
    persona_tokens = _values_to_token_set(_persona_text_fragments(persona))
    slang_score = 1.0
    tech_score = 0.7
    event_memory_score = 1.0

    if isinstance(cohort, dict):
        slang_profile = cohort.get("slang_compatibility", {})
        if isinstance(slang_profile, dict):
            unknown_slang = slang_profile.get("unknown_slang", [])
            recognized_slang = slang_profile.get("recognized_slang", [])
            active_slang = slang_profile.get("active_slang", [])
            unknown_hits = _phrase_hits(text_blob, unknown_slang if isinstance(unknown_slang, list) else [])
            positive_hits = _phrase_hits(
                text_blob,
                (
                    list(active_slang) if isinstance(active_slang, list) else []
                ) + (
                    list(recognized_slang) if isinstance(recognized_slang, list) else []
                ),
            )
            slang_score = _clamp(1.0 - (0.35 * unknown_hits) + (0.05 * positive_hits))

        tech_familiarity = cohort.get("tech_familiarity", {})
        if isinstance(tech_familiarity, dict):
            native = tech_familiarity.get("grew_up_with", [])
            adopted = tech_familiarity.get("adopted_as_adult", [])
            never_used = tech_familiarity.get("never_used", [])
            supported_hits = _token_hits(
                persona_tokens,
                (
                    list(native) if isinstance(native, list) else []
                ) + (
                    list(adopted) if isinstance(adopted, list) else []
                ),
            )
            impossible_hits = _phrase_hits(text_blob, never_used if isinstance(never_used, list) else [])
            tech_score = _clamp(0.7 + (0.08 * supported_hits) - (0.3 * impossible_hits))

        years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text_blob)]
        for year in years:
            if year > eval_year:
                event_memory_score -= 0.4
            elif age is not None and year < birth_year + 5:
                event_memory_score -= 0.2
        event_memory_score = _clamp(event_memory_score)

    return _clamp(
        (0.25 * age_consistency)
        + (0.35 * snapshot_alignment)
        + (0.15 * tech_score)
        + (0.15 * slang_score)
        + (0.10 * event_memory_score)
    )


def capability_coherence(persona: dict) -> float:
    """Deterministic coherence score for the capability matrix."""
    matrix = persona.get("capability_matrix")
    if not isinstance(matrix, dict) or not matrix:
        return float("nan")

    persona_tokens = _values_to_token_set(_persona_text_fragments(persona))
    age = _persona_age(persona)
    domain_scores: list[float] = []

    for domain_name, profile in matrix.items():
        if not isinstance(profile, dict):
            continue

        levels = [
            float(profile[field_name])
            for field_name in _CAPABILITY_LEVEL_FIELDS
            if isinstance(profile.get(field_name), (int, float))
        ]
        if not levels:
            continue
        avg_level = sum(levels) / len(levels)

        identity = str(profile.get("identity_salience", "competent"))
        motivation = float(profile.get("motivation", 0.0))
        identity_score = 1.0
        if identity == "core" and motivation < 3.0:
            identity_score = 0.4 + (0.2 * motivation)
        elif identity == "avoidance":
            identity_score = _clamp(1.0 - (motivation / 5.0))
        elif identity == "chore" and motivation > 4.0:
            identity_score = 0.5

        confidence_level = str(profile.get("confidence_level", "inferred"))
        evidence = profile.get("evidence")
        evidence_present = isinstance(evidence, str) and bool(evidence.strip())
        confidence_score = 1.0
        if confidence_level == "grounded" and not evidence_present:
            confidence_score = 0.25
        elif confidence_level == "guessed" and evidence_present:
            confidence_score = 0.7

        support_score = 0.55
        domain_token_overlap = len(_normalize_tokens(str(domain_name)) & persona_tokens)
        if domain_token_overlap:
            support_score += 0.25
        if evidence_present and _normalize_tokens(evidence) & persona_tokens:
            support_score += 0.20
        support_score = _clamp(support_score)

        experience_score = 0.65
        experience = profile.get("experience")
        if isinstance(experience, dict):
            years_exposed = int(experience.get("years_exposed", 0))
            years_practiced = int(experience.get("years_practiced", 0))
            practice_intensity = float(experience.get("deliberate_practice_intensity", 0.0))
            success_rate = float(experience.get("success_rate", 0.0))
            stage = int(experience.get("unconscious_competence_stage", 1))
            recency = str(experience.get("recency", "active"))

            stage_low, stage_high = _STAGE_SKILL_WINDOWS.get(stage, (0.0, 5.0))
            stage_score = _score_range(avg_level, stage_low, stage_high, tolerance=1.3)
            practice_expectation = min(5.0, 0.6 + (0.22 * years_practiced) + (0.35 * practice_intensity))
            practice_score = _score_range(avg_level, max(0.0, practice_expectation - 1.8), min(5.0, practice_expectation + 1.1), tolerance=1.5)
            calibration_score = _clamp(1.0 - abs((profile.get("confidence_calibration", 0.0) / 5.0) - success_rate))
            exposure_score = 1.0
            if age is not None and years_exposed > age:
                exposure_score = _clamp(1.0 - ((years_exposed - age) / 10.0))
            if age is not None and years_practiced > age:
                exposure_score = min(exposure_score, _clamp(1.0 - ((years_practiced - age) / 10.0)))
            recency_cap = _RECENCY_SPEED_CAPS.get(recency, 5.0)
            recency_score = _score_range(
                float(profile.get("speed", 0.0)),
                0.0,
                recency_cap,
                tolerance=1.0,
            )
            experience_score = _mean_defined(
                (stage_score, practice_score, calibration_score, exposure_score, recency_score)
            )

        condition_score = 0.7
        conditions = profile.get("conditions")
        if isinstance(conditions, dict):
            floor = float(conditions.get("floor", 0.0))
            ceiling = float(conditions.get("ceiling", 5.0))
            in_bounds = _score_range(avg_level, floor, ceiling, tolerance=1.0)
            stress_modifier = float(conditions.get("stress_modifier", 0.0))
            fatigue_modifier = float(conditions.get("fatigue_modifier", 0.0))
            modifier_score = 1.0
            if avg_level >= 4.0 and stress_modifier <= -1.8 and fatigue_modifier <= -1.8:
                modifier_score = 0.8
            condition_score = (in_bounds + modifier_score) / 2.0

        domain_scores.append(
            _mean_defined(
                (
                    support_score,
                    identity_score,
                    confidence_score,
                    experience_score,
                    condition_score,
                )
            )
        )

    return _mean_defined(domain_scores)


def relational_realism(persona: dict) -> float:
    """Deterministic realism score for V2 relational context."""
    relational_self = persona.get("relational_self")
    if not isinstance(relational_self, dict):
        return float("nan")

    relationship_profiles = relational_self.get("relationship_profiles", {})
    trait_distributions = relational_self.get("trait_distributions", {})
    if_then_signatures = relational_self.get("if_then_signatures", [])
    group_profile = relational_self.get("group_profile")

    profile_score = 0.3
    if isinstance(relationship_profiles, dict) and relationship_profiles:
        profile_score = _clamp(0.45 + (0.15 * min(len(relationship_profiles), 3)))

    differentiation_score = 0.4
    if isinstance(relationship_profiles, dict) and len(relationship_profiles) >= 2:
        vectors: list[tuple[float, ...]] = []
        for profile in relationship_profiles.values():
            if not isinstance(profile, dict):
                continue
            vectors.append(
                (
                    float(profile.get("warmth", 0.0)),
                    float(profile.get("dominance", 0.0)),
                    float(profile.get("disclosure_level", 0.0)),
                    float(profile.get("self_monitoring", 0.0)),
                    float(profile.get("trust_level", 0.0)),
                    float(profile.get("reputation_concern", 0.0)),
                )
            )
        pairwise: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance = sum(abs(a - b) for a, b in zip(vectors[i], vectors[j])) / len(vectors[i])
                pairwise.append(distance)
        if pairwise:
            differentiation_score = _score_range(sum(pairwise) / len(pairwise), 0.10, 0.42, tolerance=0.22)

    heuristic_checks: list[float] = []
    if isinstance(relationship_profiles, dict):
        close_friend = relationship_profiles.get("close_friend")
        boss = relationship_profiles.get("boss")
        stranger = relationship_profiles.get("stranger")
        rival = relationship_profiles.get("rival")

        if isinstance(close_friend, dict) and isinstance(boss, dict):
            heuristic_checks.append(
                1.0
                if float(close_friend.get("disclosure_level", 0.0)) >= float(boss.get("disclosure_level", 0.0))
                else 0.0
            )
            heuristic_checks.append(
                1.0
                if float(boss.get("self_monitoring", 0.0)) >= float(close_friend.get("self_monitoring", 0.0))
                else 0.0
            )
        if isinstance(close_friend, dict) and isinstance(stranger, dict):
            heuristic_checks.append(
                1.0
                if float(close_friend.get("trust_level", 0.0)) >= float(stranger.get("trust_level", 0.0))
                else 0.0
            )
            heuristic_checks.append(
                1.0
                if float(close_friend.get("warmth", 0.0)) >= float(stranger.get("warmth", 0.0))
                else 0.0
            )
        if isinstance(rival, dict) and isinstance(close_friend, dict):
            heuristic_checks.append(
                1.0
                if float(rival.get("trust_level", 0.0)) <= float(close_friend.get("trust_level", 0.0))
                else 0.0
            )
    heuristic_score = _mean_defined(heuristic_checks) if heuristic_checks else 0.6

    trait_score = 0.6
    if isinstance(trait_distributions, dict) and trait_distributions:
        scores: list[float] = []
        for trait in trait_distributions.values():
            if not isinstance(trait, dict):
                continue
            mean = float(trait.get("mean", 0.0))
            floor = float(trait.get("floor", 0.0))
            ceiling = float(trait.get("ceiling", 5.0))
            variance = float(trait.get("variance", 0.0))
            scores.append(
                _mean_defined(
                    (
                        _score_range(mean, floor, ceiling, tolerance=1.0),
                        _score_range(variance, 0.05, 2.0, tolerance=1.0),
                    )
                )
            )
        if scores:
            trait_score = sum(scores) / len(scores)

    if_then_score = 0.5
    if isinstance(if_then_signatures, list) and if_then_signatures:
        scores = []
        for signature in if_then_signatures:
            if not isinstance(signature, dict):
                continue
            condition = signature.get("condition", {})
            modifiers = signature.get("behavior_modifiers", {})
            strength = float(signature.get("strength", 0.0))
            has_condition = isinstance(condition, dict) and bool(condition)
            relevant_modifiers = (
                isinstance(modifiers, dict)
                and bool(set(modifiers.keys()) & _BEHAVIOR_MODIFIER_KEYS)
            )
            evidence_present = isinstance(signature.get("evidence"), str) and bool(signature["evidence"].strip())
            scores.append(
                _mean_defined(
                    (
                        1.0 if has_condition else 0.0,
                        1.0 if relevant_modifiers else 0.0,
                        _score_range(strength, 0.3, 1.0, tolerance=0.4),
                        1.0 if evidence_present else 0.0,
                    )
                )
            )
        if scores:
            if_then_score = sum(scores) / len(scores)

    group_score = 0.65
    if isinstance(group_profile, dict):
        values = [
            float(group_profile.get(field_name, 0.0))
            for field_name in (
                "conformity_increase",
                "signaling_increase",
                "nuance_decrease",
                "in_group_warmth_boost",
                "out_group_guardedness",
                "audience_size_sensitivity",
                "leadership_emergence",
                "deference_to_consensus",
            )
            if isinstance(group_profile.get(field_name), (int, float))
        ]
        risk_shift = float(group_profile.get("risk_taking_shift", 0.0))
        non_zero = sum(1 for value in values if value > 0.05)
        group_score = _mean_defined(
            (
                _clamp(non_zero / max(len(values), 1)),
                _score_range(abs(risk_shift), 0.0, 0.7, tolerance=0.5),
            )
        )

    return _clamp(
        (0.20 * profile_score)
        + (0.20 * differentiation_score)
        + (0.20 * heuristic_score)
        + (0.15 * trait_score)
        + (0.15 * if_then_score)
        + (0.10 * group_score)
    )


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings. Handles semantic rewording better than exact match."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _normalize_tokens(value: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }
    return tokens


def _extract_path_values(persona: dict, path: str) -> list:
    current_values = [persona]
    for part in path.split("."):
        next_values: list = []
        for value in current_values:
            if isinstance(value, dict):
                child = value.get(part)
                if child is None:
                    continue
                if isinstance(child, list):
                    next_values.extend(child)
                else:
                    next_values.append(child)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        child = item.get(part)
                        if child is None:
                            continue
                        if isinstance(child, list):
                            next_values.extend(child)
                        else:
                            next_values.append(child)
        current_values = next_values
    return current_values


def _values_to_token_set(values: list) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            tokens.update(_normalize_tokens(value))
            continue
        if isinstance(value, list):
            tokens.update(_values_to_token_set(value))
            continue
        if isinstance(value, dict):
            for nested_value in value.values():
                tokens.update(_values_to_token_set([nested_value]))
            continue
        tokens.update(_normalize_tokens(str(value)))
    return tokens


def _directional_token_overlap(anchor_tokens: set[str], dependent_tokens: set[str]) -> float:
    if not anchor_tokens or not dependent_tokens:
        return float("nan")
    return len(anchor_tokens & dependent_tokens) / len(anchor_tokens)


def field_interdependence_breakdown(persona: dict) -> dict[str, object]:
    """Estimate whether dependent fields cohere with anchor fields.

    This is a deterministic heuristic. It is intentionally lightweight so
    mutation-based harnesses can run without another LLM in the loop.
    """
    rule_scores: dict[str, dict[str, object]] = {}
    overall_scores: list[float] = []

    for rule_name, rule in FIELD_INTERDEPENDENCE_RULES.items():
        anchor_values: list = []
        dependent_values: list = []
        for path in rule["anchor_paths"]:
            anchor_values.extend(_extract_path_values(persona, path))
        for path in rule["dependent_paths"]:
            dependent_values.extend(_extract_path_values(persona, path))

        anchor_tokens = _values_to_token_set(anchor_values)
        dependent_tokens = _values_to_token_set(dependent_values)
        score = _directional_token_overlap(anchor_tokens, dependent_tokens)
        if not math.isnan(score):
            overall_scores.append(score)

        rule_scores[rule_name] = {
            "score": None if math.isnan(score) else score,
            "anchor_token_count": len(anchor_tokens),
            "dependent_token_count": len(dependent_tokens),
            "shared_tokens": sorted(anchor_tokens & dependent_tokens),
        }

    overall = (
        sum(overall_scores) / len(overall_scores)
        if overall_scores
        else float("nan")
    )
    return {
        "overall": overall,
        "rules": rule_scores,
    }


def field_interdependence(persona: dict) -> float:
    """Overall deterministic field interdependence score."""
    return field_interdependence_breakdown(persona)["overall"]


def _field_similarity(val_a, val_b) -> float:
    """Compare two field values with type-aware similarity.

    - Lists: mean pairwise Jaccard of sorted items (order-independent)
    - Strings: Jaccard token similarity
    - Dicts: recursive mean field similarity
    - Exact match fallback for other types
    """
    if val_a is None and val_b is None:
        return 1.0
    if val_a is None or val_b is None:
        return 0.0

    # Both are lists — compare items order-independently
    if isinstance(val_a, list) and isinstance(val_b, list):
        if not val_a and not val_b:
            return 1.0
        if not val_a or not val_b:
            return 0.0
        # Convert items to strings, sort, compare pairwise
        strs_a = sorted(str(x) for x in val_a)
        strs_b = sorted(str(x) for x in val_b)
        n = max(len(strs_a), len(strs_b))
        total_sim = 0.0
        for i in range(min(len(strs_a), len(strs_b))):
            total_sim += _jaccard_similarity(strs_a[i], strs_b[i])
        return total_sim / n

    # Both are dicts — recursive field comparison
    if isinstance(val_a, dict) and isinstance(val_b, dict):
        all_keys = set(val_a.keys()) | set(val_b.keys())
        if not all_keys:
            return 1.0
        return sum(_field_similarity(val_a.get(k), val_b.get(k)) for k in all_keys) / len(all_keys)

    # Strings — Jaccard token similarity
    if isinstance(val_a, str) and isinstance(val_b, str):
        return _jaccard_similarity(val_a, val_b)

    # Fallback: exact match for numbers, bools, etc.
    return 1.0 if val_a == val_b else 0.0


def _stability_similarities_by_field(
    persona_runs: list[list[dict]],
    key_fields: list[str],
) -> dict[str, list[float]]:
    """Collect pairwise similarities for each tracked field across all runs."""
    similarities_by_field = {field_name: [] for field_name in key_fields}

    n_personas = len(persona_runs[0])
    for persona_idx in range(n_personas):
        for field_name in key_fields:
            values = []
            for run in persona_runs:
                if persona_idx < len(run):
                    values.append(run[persona_idx].get(field_name))
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    similarities_by_field[field_name].append(
                        _field_similarity(values[i], values[j]),
                    )

    return similarities_by_field


def stability(persona_runs: list[list[dict]], key_fields: list[str] | None = None) -> float:
    """Measure rerun stability: how much do key fields vary across identical runs?

    Uses token-level Jaccard similarity (not exact string match) so semantically
    equivalent rewrites score high. Lists are compared order-independently.

    Args:
        persona_runs: list of runs, each run is a list of persona dicts (same source each time).
                      All runs should have the same length and ordering.
        key_fields: which top-level fields to compare. Defaults to core content fields.

    Returns:
        Mean field-level similarity across runs (1.0 = perfectly stable, 0.0 = total drift).
        Returns NaN if fewer than 2 runs.
    """
    if len(persona_runs) < 2:
        return float("nan")

    if key_fields is None:
        key_fields = DEFAULT_STABILITY_FIELDS

    n_personas = len(persona_runs[0])
    if n_personas == 0:
        return float("nan")

    similarities_by_field = _stability_similarities_by_field(persona_runs, key_fields)
    similarities = [
        similarity
        for field_scores in similarities_by_field.values()
        for similarity in field_scores
    ]

    if not similarities:
        return float("nan")
    return sum(similarities) / len(similarities)


def stability_breakdown(
    persona_runs: list[list[dict]],
    key_fields: list[str] | None = None,
) -> dict:
    """Return overall stability plus per-field detail for inspection/debugging."""
    if key_fields is None:
        key_fields = DEFAULT_STABILITY_FIELDS

    num_runs = len(persona_runs)
    num_personas = len(persona_runs[0]) if persona_runs else 0

    if num_runs < 2 or num_personas == 0:
        return {
            "overall": float("nan"),
            "num_runs": num_runs,
            "num_personas": num_personas,
            "fields": {},
        }

    similarities_by_field = _stability_similarities_by_field(persona_runs, key_fields)
    fields: dict[str, dict[str, float | int]] = {}

    for field_name, scores in similarities_by_field.items():
        if not scores:
            fields[field_name] = {"similarity": float("nan"), "comparisons": 0}
            continue
        fields[field_name] = {
            "similarity": sum(scores) / len(scores),
            "comparisons": len(scores),
        }

    all_scores = [
        similarity
        for field_scores in similarities_by_field.values()
        for similarity in field_scores
    ]

    return {
        "overall": sum(all_scores) / len(all_scores) if all_scores else float("nan"),
        "num_runs": num_runs,
        "num_personas": num_personas,
        "fields": fields,
    }


def pairwise_preference_rate(wins: int, total: int) -> float:
    """Win rate from pairwise judge comparisons.

    Computed from judge results, not persona content.
    """
    if total == 0:
        return float("nan")
    return wins / total


def run_core_metrics(
    personas: list[dict],
    schema_cls=None,
    groundedness_reports: list | None = None,
    embeddings: list[list[float]] | None = None,
    persona_runs: list[list[dict]] | None = None,
) -> dict[str, float]:
    """Run 6 baseline metrics and return a summary dict.

    The 6 metrics computed here:
    1. groundedness, 2. developmental_fit, 3. historical_fit,
    4. capability_coherence, 5. relational_realism, 6. stability.
    Pairwise preference is computed at comparison time from judge results.
    """
    results: dict[str, float] = {}

    if schema_cls is not None:
        results["schema_validity"] = schema_validity(personas, schema_cls)

    if groundedness_reports is not None:
        results["groundedness"] = groundedness_rate(groundedness_reports)

    if embeddings is not None:
        results["distinctiveness"] = distinctiveness(embeddings)

    if personas:
        dev_scores = [developmental_fit(p) for p in personas]
        hist_scores = [historical_fit(p) for p in personas]
        capability_scores = [capability_coherence(p) for p in personas]
        relational_scores = [relational_realism(p) for p in personas]
        results["developmental_fit"] = _mean_defined(dev_scores)
        results["historical_fit"] = _mean_defined(hist_scores)
        results["capability_coherence"] = _mean_defined(capability_scores)
        results["relational_realism"] = _mean_defined(relational_scores)

    # Stability requires multiple runs of the same pipeline
    if persona_runs is not None:
        results["stability"] = stability(persona_runs)
    else:
        results["stability"] = float("nan")

    return results


# TODO(space-5): add implementations for the remaining shared metrics.
#
#   - turing_pass_rate(labels)  : human-task-worker identification rate
#   - drift(turn_embeddings)    : stylometric drift turn N vs turn 1
#   - turns_to_break(attack_log): mean turns until an adversarial prompt wins
#   - judge_rubric_score(j, p)  : Opus-as-judge per-dimension rubric score
#   - human_correlation(j, h)   : Spearman between judge and human labels
