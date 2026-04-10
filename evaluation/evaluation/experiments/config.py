"""Experiment configuration and result dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExperimentConfig:
    """Specification for a single experiment run."""
    number: str                    # "2.16"
    title: str                     # "prompt-compression"
    description: str
    files: list[str]
    primary_metric: str            # metric name from metrics.py or "new:<name>"
    branch_name: str               # "exp-2.16-prompt-compression"
    hypothesis: str = ""
    problem_space: int = 0

@dataclass
class ExperimentResult:
    """Complete result of an experiment run including comparison to baseline."""
    config: ExperimentConfig
    baseline: dict[str, float] = field(default_factory=dict)
    experiment: dict[str, float] = field(default_factory=dict)
    variants: list[dict[str, Any]] | None = None
    deltas: dict[str, float] = field(default_factory=dict)
    guardrails: dict[str, dict[str, Any]] = field(default_factory=dict)
    signal_strength: str = "inconclusive"
    recommendation: str = "defer"
    findings: str = ""
    run_metadata: dict[str, Any] = field(default_factory=dict)
