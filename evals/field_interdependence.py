"""Experiment 1.07: Field interdependence — ablation harness.

Generates a control persona via the standard pipeline, then ablates each
removable field (set to []) and re-scores the degraded copy with the judge.
Builds a dependency matrix: rows = removed field, columns = scoring dimension,
cells = score delta from control.

Fields are classified as:
  - load-bearing: removal drops 2+ dimensions by >= 0.5
  - decorative:   removal has minimal impact on scores
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from evals.judge_helper_1_07 import JudgeScore, LLMJudge

# Fields eligible for ablation (list-type fields on PersonaV1)
ABLATABLE_FIELDS = [
    "goals",
    "pains",
    "motivations",
    "objections",
    "channels",
    "vocabulary",
    "decision_triggers",
    "sample_quotes",
    "journey_stages",
]

LOAD_BEARING_THRESHOLD = 0.5   # score drop to count as significant
LOAD_BEARING_MIN_DIMS = 2      # min dimensions that must drop


@dataclass
class AblationResult:
    """Result of ablating one field from a persona."""
    field_name: str
    control_scores: dict[str, float]
    ablated_scores: dict[str, float]
    deltas: dict[str, float]       # ablated - control (negative = worse)
    control_overall: float = 0.0
    ablated_overall: float = 0.0
    overall_delta: float = 0.0
    classification: str = ""       # "load-bearing" or "decorative"
    rationale: str = ""


@dataclass
class FieldInterdependenceReport:
    """Full experiment report."""
    control_persona_name: str = ""
    control_scores: dict[str, float] = field(default_factory=dict)
    control_overall: float = 0.0
    ablations: list[AblationResult] = field(default_factory=list)
    load_bearing_fields: list[str] = field(default_factory=list)
    decorative_fields: list[str] = field(default_factory=list)
    dependency_matrix: dict[str, dict[str, float]] = field(default_factory=dict)


def ablate_persona(persona_dict: dict, field_name: str) -> dict:
    """Return a copy of persona_dict with field_name set to []."""
    ablated = copy.deepcopy(persona_dict)
    if field_name in ablated:
        ablated[field_name] = []
    return ablated


def classify_ablation(deltas: dict[str, float]) -> str:
    """Classify based on how many dimensions dropped significantly."""
    sig_drops = sum(1 for d in deltas.values() if d <= -LOAD_BEARING_THRESHOLD)
    return "load-bearing" if sig_drops >= LOAD_BEARING_MIN_DIMS else "decorative"


async def run_ablation_study(
    persona_dict: dict,
    judge: LLMJudge,
    fields: list[str] | None = None,
) -> FieldInterdependenceReport:
    """Run the full ablation study on a single persona.

    1. Score the control persona
    2. For each field, ablate it and re-score
    3. Compute deltas and classify
    """
    fields = fields or ABLATABLE_FIELDS
    report = FieldInterdependenceReport()

    report.control_persona_name = persona_dict.get("name", "unknown")

    # Score control
    print(f"  Scoring control persona: {report.control_persona_name}")
    control_score = await judge.score_persona(persona_dict)
    report.control_scores = control_score.dimensions
    report.control_overall = control_score.overall

    # Ablate each field
    for field_name in fields:
        if field_name not in persona_dict:
            print(f"  Skipping {field_name} (not in persona)")
            continue

        original_value = persona_dict.get(field_name, [])
        if not isinstance(original_value, list) or len(original_value) == 0:
            print(f"  Skipping {field_name} (empty or non-list)")
            continue

        print(f"  Ablating: {field_name} (removing {len(original_value)} items)")
        ablated = ablate_persona(persona_dict, field_name)
        ablated_score = await judge.score_persona(ablated)

        deltas = {}
        for dim in control_score.dimensions:
            ctrl = control_score.dimensions.get(dim, 0)
            abl = ablated_score.dimensions.get(dim, 0)
            deltas[dim] = abl - ctrl

        classification = classify_ablation(deltas)

        result = AblationResult(
            field_name=field_name,
            control_scores=control_score.dimensions,
            ablated_scores=ablated_score.dimensions,
            deltas=deltas,
            control_overall=control_score.overall,
            ablated_overall=ablated_score.overall,
            overall_delta=ablated_score.overall - control_score.overall,
            classification=classification,
            rationale=ablated_score.rationale,
        )
        report.ablations.append(result)
        report.dependency_matrix[field_name] = deltas

        if classification == "load-bearing":
            report.load_bearing_fields.append(field_name)
        else:
            report.decorative_fields.append(field_name)

        status = "LOAD-BEARING" if classification == "load-bearing" else "decorative"
        print(f"    -> {status} (overall delta: {result.overall_delta:+.1f})")

    return report


def format_dependency_matrix(report: FieldInterdependenceReport) -> str:
    """Format the dependency matrix as a readable table."""
    if not report.ablations:
        return "No ablation results."

    dims = list(report.control_scores.keys())
    lines: list[str] = []

    # Header
    header = f"{'Field removed':<22}"
    for d in dims:
        header += f"{d:>14}"
    header += f"{'overall':>14}  {'class':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    # Control row
    ctrl_row = f"{'(control)':.<22}"
    for d in dims:
        ctrl_row += f"{report.control_scores[d]:>14.1f}"
    ctrl_row += f"{report.control_overall:>14.1f}  {'---':>14}"
    lines.append(ctrl_row)
    lines.append("")

    # Delta rows
    for abl in report.ablations:
        row = f"{abl.field_name:.<22}"
        for d in dims:
            delta = abl.deltas.get(d, 0)
            row += f"{delta:>+14.1f}"
        row += f"{abl.overall_delta:>+14.1f}  {abl.classification:>14}"
        lines.append(row)

    lines.append("")
    lines.append(f"Load-bearing fields: {', '.join(report.load_bearing_fields) or 'none'}")
    lines.append(f"Decorative fields:   {', '.join(report.decorative_fields) or 'none'}")

    return "\n".join(lines)


def report_to_dict(report: FieldInterdependenceReport) -> dict:
    """Serialize report to JSON-friendly dict."""
    return {
        "control_persona_name": report.control_persona_name,
        "control_scores": report.control_scores,
        "control_overall": report.control_overall,
        "ablations": [
            {
                "field_name": a.field_name,
                "control_scores": a.control_scores,
                "ablated_scores": a.ablated_scores,
                "deltas": a.deltas,
                "control_overall": a.control_overall,
                "ablated_overall": a.ablated_overall,
                "overall_delta": a.overall_delta,
                "classification": a.classification,
                "rationale": a.rationale,
            }
            for a in report.ablations
        ],
        "dependency_matrix": report.dependency_matrix,
        "load_bearing_fields": report.load_bearing_fields,
        "decorative_fields": report.decorative_fields,
    }
