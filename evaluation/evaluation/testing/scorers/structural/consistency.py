from __future__ import annotations
import re
from typing import Callable
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Each rule is a (check_fn, error_message) tuple.
# check_fn returns True when the persona VIOLATES the constraint.
Rule = tuple[Callable[[Persona], bool], str]

_SENIOR_TITLES = re.compile(r"\b(senior|principal|staff|lead|director|vp|cto|ceo|coo|chief)\b", re.I)
_ENTRY_TITLES = re.compile(r"\b(junior|entry.?level|intern|associate|graduate|trainee)\b", re.I)


def _has_senior_title(p: Persona) -> bool:
    return bool(_SENIOR_TITLES.search(p.occupation))


def _has_entry_title(p: Persona) -> bool:
    return bool(_ENTRY_TITLES.search(p.occupation))


RULES: list[Rule] = [
    # Senior title but very few years of experience
    (
        lambda p: _has_senior_title(p) and p.experience_years is not None and p.experience_years < 3,
        "Senior-level title but experience_years < 3",
    ),
    # Entry-level title but many years of experience
    (
        lambda p: _has_entry_title(p) and p.experience_years is not None and p.experience_years > 10,
        "Entry-level title but experience_years > 10",
    ),
    # Age too young for stated experience
    (
        lambda p: (
            p.age is not None
            and p.experience_years is not None
            and (p.age - p.experience_years) < 14
        ),
        "Age implies work started before age 14",
    ),
    # Claims budget-conscious but expensive lifestyle markers
    (
        lambda p: (
            "budget" in " ".join(p.values + p.behaviors).lower()
            and "luxury" in (p.lifestyle + " " + p.bio).lower()
        ),
        "Claims budget-conscious but lifestyle indicates luxury spending",
    ),
    # Introvert + large social leadership behaviors
    (
        lambda p: (
            "introvert" in " ".join(p.personality_traits).lower()
            and "public speaking" in " ".join(p.behaviors + p.goals).lower()
            and "extrovert" not in " ".join(p.personality_traits).lower()
        ),
        "Introvert personality but public speaking listed as core behavior/goal without reconciliation",
    ),
]


class ConsistencyScorer(BaseScorer):
    dimension_id = "D3"
    dimension_name = "Internal Logical Consistency"
    tier = 1

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        violations: list[str] = []
        for check_fn, message in RULES:
            try:
                if check_fn(persona):
                    violations.append(message)
            except Exception as e:
                violations.append(f"Rule check error: {e}")

        passed = len(violations) == 0
        score = max(0.0, 1.0 - len(violations) * 0.25)
        return self._result(
            persona,
            passed=passed,
            score=score,
            details={"violations": violations},
            errors=violations,
        )
