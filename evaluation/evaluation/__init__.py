"""Evaluation scaffold for lab research problem space 5.

This package is intentionally a skeleton — it exists so every other lab
experiment has a single, stable place to import judges and metrics from.
Researcher #5 owns filling this out.
"""

from .metrics import (
    schema_validity,
    groundedness_rate,
    distinctiveness,
    cost_per_persona,
)
from .judges import LLMJudge, JudgeScore
from .golden_set import GoldenTenant, load_golden_set

__all__ = [
    "schema_validity",
    "groundedness_rate",
    "distinctiveness",
    "cost_per_persona",
    "LLMJudge",
    "JudgeScore",
    "GoldenTenant",
    "load_golden_set",
]
