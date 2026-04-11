"""Experiment 5.11: Reference-based vs reference-free judging.

Provides prompt builders for two judging modes:
  - FREE mode: standard rubric-only scoring (no reference persona)
  - REFERENCE mode: rubric + a proxy reference persona as calibration anchor

Also includes comparison statistics to measure whether reference-mode
reduces score variance and/or anchors scores.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field


# ── Proxy reference persona (clearly labeled) ────────────────────────

PROXY_REFERENCE_PERSONA = {
    "_meta": "PROXY REFERENCE — hand-crafted calibration anchor, NOT a validated gold persona",
    "name": "Platform Engineering Lead — Automation-First (PROXY)",
    "summary": (
        "A senior platform engineer at a mid-size fintech who treats manual "
        "processes as bugs. Spends most of her week on CI/CD and fighting for "
        "engineering standards across product teams."
    ),
    "demographics": {
        "age_range": "32-38",
        "gender_distribution": "predominantly female",
        "location_signals": ["SF Bay Area", "remote-first"],
    },
    "firmographics": {
        "company_size": "100-200",
        "industry": "fintech",
        "role_titles": ["Senior Platform Engineer", "DevOps Lead"],
        "tech_stack_signals": ["Terraform", "GitHub Actions", "Datadog", "AWS EKS"],
    },
    "goals": [
        "Automate deployment pipelines to achieve <10min deploy cycles",
        "Consolidate observability into a single dashboard",
        "Reduce CI flakiness from 12% to under 2% by Q3",
    ],
    "pains": [
        "5+ hours/week debugging CI failures from flaky tests other teams wrote",
        "No deployment standardization across teams",
        "Rate limits on bulk export API break nightly pipeline twice a month",
    ],
    "motivations": [
        "Prove platform engineering deserves headcount",
        "Infrastructure-as-code is non-negotiable",
    ],
    "objections": [
        "Won't adopt tools without a Terraform provider",
        "Resistant to per-seat pricing for CI bots",
    ],
    "channels": ["HashiCorp forums", "DevOps Weekly", "#platform-eng Slack"],
    "vocabulary": ["toil", "blast radius", "golden path", "SLO", "error budget"],
    "decision_triggers": [
        "Terraform provider on day one",
        "SOC 2 docs available",
        "Bulk API >10k row support",
    ],
    "sample_quotes": [
        "If I can't terraform it, it doesn't exist in my infrastructure.",
        "That's not engineering, that's archaeology.",
    ],
    "source_evidence": [
        {"claim": "Automate deployment pipelines", "record_ids": ["proxy_001", "proxy_002"], "field_path": "goals.0", "confidence": 0.9},
        {"claim": "CI flakiness debugging", "record_ids": ["proxy_003"], "field_path": "pains.0", "confidence": 0.95},
        {"claim": "Platform eng headcount", "record_ids": ["proxy_001"], "field_path": "motivations.0", "confidence": 0.8},
    ],
}

# Expected quality level for the proxy (used in anchoring detection)
PROXY_EXPECTED_OVERALL = 4.5  # We expect this proxy to score ~4.5/5


# ── Prompt builders ───────────────────────────────────────────────────

def build_free_prompt(persona: dict) -> str:
    """Build a standard reference-free judging prompt."""
    return (
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


def build_reference_prompt(persona: dict) -> str:
    """Build a reference-based judging prompt with proxy anchor."""
    return (
        "Below is a PROXY REFERENCE persona (hand-crafted calibration anchor, "
        "NOT a validated gold standard). Use it as a quality benchmark — the "
        "reference represents approximately 4.5/5 quality. Score the TARGET "
        "persona relative to this reference.\n\n"
        "=== PROXY REFERENCE PERSONA (approximate quality: 4.5/5) ===\n"
        + json.dumps(PROXY_REFERENCE_PERSONA, indent=2, default=str)
        + "\n\n"
        "=== TARGET PERSONA TO SCORE ===\n"
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


# ── Comparison statistics ─────────────────────────────────────────────

@dataclass
class ModeStats:
    """Aggregate statistics for one judging mode."""
    mode: str
    scores: list[float] = field(default_factory=list)
    dimension_scores: dict[str, list[float]] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.scores) if self.scores else float("nan")

    @property
    def std(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def n(self) -> int:
        return len(self.scores)


@dataclass
class ComparisonResult:
    """Result of comparing free vs reference judging modes."""
    free_stats: ModeStats
    ref_stats: ModeStats
    variance_reduction_ratio: float = 0.0
    mean_delta: float = 0.0
    spearman_rho: float = float("nan")
    anchoring_detected: bool = False
    anchoring_evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "free_mode": {
                "mean": round(self.free_stats.mean, 3),
                "std": round(self.free_stats.std, 3),
                "n": self.free_stats.n,
                "scores": [round(s, 2) for s in self.free_stats.scores],
            },
            "reference_mode": {
                "mean": round(self.ref_stats.mean, 3),
                "std": round(self.ref_stats.std, 3),
                "n": self.ref_stats.n,
                "scores": [round(s, 2) for s in self.ref_stats.scores],
            },
            "variance_reduction_ratio": round(self.variance_reduction_ratio, 3),
            "mean_delta": round(self.mean_delta, 3),
            "spearman_rho": round(self.spearman_rho, 3) if not math.isnan(self.spearman_rho) else None,
            "anchoring_detected": self.anchoring_detected,
            "anchoring_evidence": self.anchoring_evidence,
        }


def _rank(values: list[float]) -> list[float]:
    """Compute ranks for a list of values (average ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank for tied group
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
    return rho


def compare_modes(
    free_scores: list[float],
    ref_scores: list[float],
    anchor_quality: float = PROXY_EXPECTED_OVERALL,
) -> ComparisonResult:
    """Compare free vs reference judging modes.

    Args:
        free_scores: overall scores from free (reference-free) mode
        ref_scores: overall scores from reference mode (same personas, same order)
        anchor_quality: the declared quality of the proxy reference
    """
    free_stats = ModeStats(mode="free", scores=free_scores)
    ref_stats = ModeStats(mode="reference", scores=ref_scores)

    # Variance reduction
    free_var = free_stats.std ** 2 if free_stats.std > 0 else 0.0
    ref_var = ref_stats.std ** 2 if ref_stats.std > 0 else 0.0
    if free_var > 0:
        variance_reduction = 1.0 - (ref_var / free_var)
    else:
        variance_reduction = 0.0

    # Mean delta
    mean_delta = ref_stats.mean - free_stats.mean

    # Rank correlation
    rho = spearman_correlation(free_scores, ref_scores)

    # Anchoring detection: do ref-mode scores cluster around the anchor quality?
    # If >60% of ref scores are within 0.5 of the anchor, we flag anchoring
    if ref_scores:
        near_anchor = sum(1 for s in ref_scores if abs(s - anchor_quality) <= 0.5)
        anchor_pct = near_anchor / len(ref_scores)
        anchoring_detected = anchor_pct > 0.6
        anchoring_evidence = (
            f"{near_anchor}/{len(ref_scores)} ({anchor_pct:.0%}) of reference-mode "
            f"scores are within 0.5 of anchor quality ({anchor_quality})"
        )
    else:
        anchoring_detected = False
        anchoring_evidence = "No scores to analyze"

    return ComparisonResult(
        free_stats=free_stats,
        ref_stats=ref_stats,
        variance_reduction_ratio=variance_reduction,
        mean_delta=mean_delta,
        spearman_rho=rho,
        anchoring_detected=anchoring_detected,
        anchoring_evidence=anchoring_evidence,
    )
