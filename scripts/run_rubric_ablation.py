"""Run experiment 5.05: rubric ablation.

Score the same frozen personas with the full rubric and with one dimension
removed at a time. Compare mean scores and persona rank stability.
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

from research_utils import load_persona_entries  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "evaluation" / "baselines" / "p1_baseline.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "baselines" / "rubric_ablation.json"


def _spearman(rank_a: dict[str, int], rank_b: dict[str, int]) -> float | None:
    names = sorted(set(rank_a) & set(rank_b))
    n = len(names)
    if n < 2:
        return None
    diff_sum = sum((rank_a[name] - rank_b[name]) ** 2 for name in names)
    return 1 - ((6 * diff_sum) / (n * (n**2 - 1)))


def _rankings(rows: list[dict]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: row["overall"], reverse=True)
    return {row["persona_name"]: index + 1 for index, row in enumerate(ordered)}


async def main(input_path: Path, output_path: Path, *, temperature: float | None) -> None:
    settings.validate_runtime_settings()
    artifact, entries = load_persona_entries(input_path, temperature=temperature)
    backend = build_judge_backend_from_settings(model=settings.resolved_judge_model)
    judge = LLMJudge(backend=backend)

    rubric_variants: list[tuple[str, tuple[str, ...]]] = [("full", SCORE_DIMENSIONS)]
    for dimension in SCORE_DIMENSIONS:
        rubric_variants.append(
            (f"minus_{dimension}", tuple(d for d in SCORE_DIMENSIONS if d != dimension)),
        )

    variant_rows: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []

    for variant_name, dimensions in rubric_variants:
        rows: list[dict] = []
        for entry in entries:
            persona = entry["persona"]
            result = await judge.score_persona_with_context(persona, dimensions=dimensions)
            rows.append({
                "persona_name": entry.get("name"),
                "cluster_id": entry.get("cluster_id"),
                "overall": result.overall,
                "dimensions": result.dimensions,
                "rationale": result.rationale,
            })
        variant_rows[variant_name] = rows

    full_rank = _rankings(variant_rows["full"])
    full_means = [row["overall"] for row in variant_rows["full"] if not math.isnan(row["overall"])]
    full_mean = statistics.mean(full_means) if full_means else float("nan")

    for variant_name, rows in variant_rows.items():
        means = [row["overall"] for row in rows if not math.isnan(row["overall"])]
        mean_overall = statistics.mean(means) if means else float("nan")
        rank_corr = _spearman(full_rank, _rankings(rows))
        summary_rows.append({
            "variant": variant_name,
            "dimensions": list(next(dim for name, dim in rubric_variants if name == variant_name)),
            "mean_overall": None if math.isnan(mean_overall) else round(mean_overall, 6),
            "delta_vs_full": None if math.isnan(mean_overall) or math.isnan(full_mean) else round(mean_overall - full_mean, 6),
            "rank_correlation_vs_full": None if rank_corr is None else round(rank_corr, 6),
        })

    report = {
        "experiment_id": "5.05",
        "name": "Rubric ablation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_artifact": str(input_path),
        "source_artifact_type": artifact.get("experiment_id", artifact.get("version", "unknown")),
        "selected_temperature": temperature,
        "judge_model": settings.resolved_judge_model,
        "variants": summary_rows,
        "variant_rows": variant_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment 5.05 rubric ablation")
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
