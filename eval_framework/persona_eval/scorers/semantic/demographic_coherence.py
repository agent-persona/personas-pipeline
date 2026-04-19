"""D7 Demographic Coherence — co-occurrence plausibility checking.

Trustworthiness: HIGH (when reference data exists).
Method: Check attribute combinations against plausibility rules.
"""

from __future__ import annotations

from collections.abc import Callable

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext


def _check_phd_age(p: Persona) -> tuple[bool, str]:
    if p.age and p.education and "phd" in p.education.lower() and p.age < 24:
        return (False, f"PhD at age {p.age} is extremely rare")
    return (True, "")


def _check_ceo_age(p: Persona) -> tuple[bool, str]:
    if p.age and p.occupation and p.age < 25:
        if any(t in p.occupation.lower() for t in ["ceo", "chief", "vp", "vice president", "director"]):
            return (False, f"C-suite role '{p.occupation}' at age {p.age} is implausible")
    return (True, "")


def _check_widowed_young(p: Persona) -> tuple[bool, str]:
    if p.age and p.marital_status and p.age < 25 and "widow" in p.marital_status.lower():
        return (False, f"Widowed at age {p.age} is statistically very rare")
    return (True, "")


def _check_income_education_mismatch(p: Persona) -> tuple[bool, str]:
    if p.age is None or not p.education or not p.income_bracket:
        return (True, "")
    if "high school" in p.education.lower() and p.income_bracket == "high" and p.age < 30:
        return (False, "High income with only high school education at young age is uncommon")
    return (True, "")


def _check_experience_vs_age(p: Persona) -> tuple[bool, str]:
    if p.age and p.experience_years is not None:
        min_working_age = 16
        max_possible = p.age - min_working_age
        if p.experience_years > max_possible:
            return (False, f"{p.experience_years} years experience at age {p.age} is impossible (max {max_possible})")
    return (True, "")


COHERENCE_RULES: list[Callable[[Persona], tuple[bool, str]]] = [
    _check_phd_age,
    _check_ceo_age,
    _check_widowed_young,
    _check_income_education_mismatch,
    _check_experience_vs_age,
]


class DemographicCoherenceScorer(BaseScorer):
    """Evaluates whether persona demographics are internally plausible."""

    dimension_id = "D7"
    dimension_name = "Demographic Coherence"
    tier = 2

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        anomalies: list[dict] = []
        checked = 0

        for rule_fn in COHERENCE_RULES:
            checked += 1
            passed, msg = rule_fn(persona)
            if not passed:
                anomalies.append({"rule": rule_fn.__name__, "message": msg})

        if checked == 0:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True},
            )

        score = max(0.0, (checked - len(anomalies)) / checked)
        passed = len(anomalies) == 0

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={"anomalies": anomalies, "rules_checked": checked},
        )
