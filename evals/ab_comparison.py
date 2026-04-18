"""A/B comparison utilities for humanization experiment."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageComparison:
    stage_name: str
    baseline_scores: dict[str, float]
    humanized_scores: dict[str, float]
    deltas: dict[str, float]


def compare_scores(baseline: dict, humanized: dict) -> dict:
    """Compute per-dimension deltas between baseline and humanized scores."""
    all_keys = sorted(set(list(baseline.keys()) + list(humanized.keys())))
    deltas = {}
    for key in all_keys:
        b = baseline.get(key, 0.0)
        h = humanized.get(key, 0.0)
        deltas[key] = round(h - b, 3)
    return deltas


def format_comparison_table(comparisons: list[StageComparison]) -> str:
    """Pretty ASCII table showing baseline vs humanized vs delta for each stage."""
    if not comparisons:
        return "(no comparisons to display)"

    # Collect all dimension keys across all comparisons
    all_dims: list[str] = []
    for comp in comparisons:
        for d in comp.baseline_scores:
            if d not in all_dims:
                all_dims.append(d)
        for d in comp.humanized_scores:
            if d not in all_dims:
                all_dims.append(d)

    # Column widths
    stage_w = max(len("Stage"), max(len(c.stage_name) for c in comparisons))
    dim_w = max(len(d) for d in all_dims) if all_dims else 10
    num_w = 8  # "baseline", "humaniz", "delta" headers

    lines: list[str] = []

    # Header
    header = f"{'Stage':<{stage_w}}  {'Dimension':<{dim_w}}  {'Baseline':>{num_w}}  {'Humanized':>{num_w}}  {'Delta':>{num_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for comp in comparisons:
        first = True
        for dim in all_dims:
            b = comp.baseline_scores.get(dim, float("nan"))
            h = comp.humanized_scores.get(dim, float("nan"))
            d = comp.deltas.get(dim, float("nan"))

            stage_label = comp.stage_name if first else ""
            delta_str = f"{d:+.2f}" if not _is_nan(d) else "n/a"
            b_str = f"{b:.2f}" if not _is_nan(b) else "n/a"
            h_str = f"{h:.2f}" if not _is_nan(h) else "n/a"

            lines.append(
                f"{stage_label:<{stage_w}}  {dim:<{dim_w}}  {b_str:>{num_w}}  {h_str:>{num_w}}  {delta_str:>{num_w}}"
            )
            first = False
        lines.append("")

    return "\n".join(lines)


def format_findings_md(experiment_data: dict) -> str:
    """Generate FINDINGS.md content from experiment data."""
    lines = [
        "# Experiment: Humanization A/B",
        "",
        "## Hypothesis",
        "Humanized personas (with backstory, speech patterns, emotional triggers)",
        "produce twin chat replies that read more like real humans than baseline personas.",
        "",
        "## Method",
        "1. Shared ingest + segmentation on tenant_acme_corp",
        "2. Baseline: synthesize v1 -> score with default judge -> twin chat -> score replies",
        "3. Humanized: synthesize v2 -> score with humanized judge -> twin chat (humanized prompt) -> score replies",
        "4. Compare per-stage scores and twin reply humanness",
        "",
    ]

    if "model_synthesis" in experiment_data:
        lines.append(f"- Synthesis model: {experiment_data['model_synthesis']}")
    if "model_judge" in experiment_data:
        lines.append(f"- Judge model: {experiment_data['model_judge']}")
    lines.append("")

    # Persona comparison
    if "comparison_table" in experiment_data:
        lines.extend([
            "## Persona Score Comparison",
            "",
            "```",
            experiment_data["comparison_table"],
            "```",
            "",
        ])

    # Twin reply comparison
    if "twin_comparison_table" in experiment_data:
        lines.extend([
            "## Twin Reply Humanness Comparison",
            "",
            "```",
            experiment_data["twin_comparison_table"],
            "```",
            "",
        ])

    # Per-persona details
    if "persona_results" in experiment_data:
        lines.append("## Per-Persona Details")
        lines.append("")
        for pr in experiment_data["persona_results"]:
            name = pr.get("name", "unknown")
            lines.append(f"### {name}")
            lines.append("")
            if "baseline_overall" in pr and "humanized_overall" in pr:
                delta = pr["humanized_overall"] - pr["baseline_overall"]
                lines.append(
                    f"- Persona score: {pr['baseline_overall']:.2f} -> "
                    f"{pr['humanized_overall']:.2f} (delta: {delta:+.2f})"
                )
            if "baseline_twin_overall" in pr and "humanized_twin_overall" in pr:
                delta = pr["humanized_twin_overall"] - pr["baseline_twin_overall"]
                lines.append(
                    f"- Twin humanness: {pr['baseline_twin_overall']:.2f} -> "
                    f"{pr['humanized_twin_overall']:.2f} (delta: {delta:+.2f})"
                )
            lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "TBD after reviewing results.",
        "",
        "## Decision",
        "",
        "TBD — adopt / reject / iterate.",
    ])

    return "\n".join(lines) + "\n"


def _is_nan(v: float) -> bool:
    try:
        return v != v  # NaN != NaN
    except (TypeError, ValueError):
        return True
