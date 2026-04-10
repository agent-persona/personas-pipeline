"""Run experiment 5.11: reference-based vs free judging.

Judge the same personas twice:
- free: persona only
- reference-based: persona plus source-cluster context
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from evaluation.judges import LLMJudge, SCORE_DIMENSIONS, build_judge_backend_from_settings  # noqa: E402
from synthesis.config import settings  # noqa: E402

from research_utils import build_reference_context, load_cluster_map, load_persona_entries, load_record_map  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "evaluation" / "baselines" / "p1_baseline.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "baselines" / "reference_vs_free_judging.json"


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return None
    return round(statistics.mean(clean), 6)


async def main(input_path: Path, output_path: Path, *, temperature: float | None) -> None:
    settings.validate_runtime_settings()
    artifact, entries = load_persona_entries(input_path, temperature=temperature)
    cluster_map = load_cluster_map()
    record_map = load_record_map()
    backend = build_judge_backend_from_settings(model=settings.resolved_judge_model)
    judge = LLMJudge(backend=backend)

    rows: list[dict] = []
    overall_free: list[float] = []
    overall_reference: list[float] = []
    delta_by_dimension: dict[str, list[float]] = {dimension: [] for dimension in SCORE_DIMENSIONS}

    for entry in entries:
        persona = entry["persona"]
        reference_context = build_reference_context(entry, cluster_map, record_map)
        free_result = await judge.score_persona_with_context(persona)
        reference_result = await judge.score_persona_with_context(
            persona,
            reference_context=reference_context,
        )

        if not math.isnan(free_result.overall):
            overall_free.append(free_result.overall)
        if not math.isnan(reference_result.overall):
            overall_reference.append(reference_result.overall)

        dimension_deltas: dict[str, float | None] = {}
        for dimension in SCORE_DIMENSIONS:
            free_value = free_result.dimensions.get(dimension, float("nan"))
            reference_value = reference_result.dimensions.get(dimension, float("nan"))
            delta = (
                reference_value - free_value
                if not math.isnan(free_value) and not math.isnan(reference_value)
                else float("nan")
            )
            dimension_deltas[dimension] = None if math.isnan(delta) else round(delta, 6)
            if not math.isnan(delta):
                delta_by_dimension[dimension].append(delta)

        rows.append({
            "persona_name": entry.get("name"),
            "cluster_id": entry.get("cluster_id"),
            "free": {
                "overall": free_result.overall,
                "dimensions": free_result.dimensions,
                "rationale": free_result.rationale,
            },
            "reference_based": {
                "overall": reference_result.overall,
                "dimensions": reference_result.dimensions,
                "rationale": reference_result.rationale,
            },
            "delta": {
                "overall": None
                if math.isnan(free_result.overall) or math.isnan(reference_result.overall)
                else round(reference_result.overall - free_result.overall, 6),
                "dimensions": dimension_deltas,
            },
            "reference_context": reference_context,
        })

    report = {
        "experiment_id": "5.11",
        "name": "Reference-based vs free judging",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_artifact": str(input_path),
        "source_artifact_type": artifact.get("experiment_id", artifact.get("version", "unknown")),
        "selected_temperature": temperature,
        "judge_model": settings.resolved_judge_model,
        "aggregate": {
            "free_overall_mean": _mean(overall_free),
            "reference_overall_mean": _mean(overall_reference),
            "overall_delta_mean": (
                None
                if not overall_free or not overall_reference
                else round(statistics.mean(overall_reference) - statistics.mean(overall_free), 6)
            ),
            "dimension_delta_means": {
                dimension: _mean(values)
                for dimension, values in delta_by_dimension.items()
            },
        },
        "rows": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment 5.11 reference-based vs free judging")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Baseline or temp sweep artifact")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save report JSON")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="If input is a temp sweep artifact, select this temperature instead of best_temperature",
    )
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, temperature=args.temperature))
