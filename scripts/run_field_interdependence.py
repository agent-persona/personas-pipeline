"""Run experiment 1.07: field interdependence harness.

Build deterministic negative controls by swapping dependent field bundles
between personas, then measure how much the field-interdependence score drops.
Optional judge mode can compare original vs mutated personas pairwise.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from evaluation.judges import LLMJudge, build_judge_backend_from_settings  # noqa: E402
from evaluation.metrics import field_interdependence_breakdown  # noqa: E402
from synthesis.config import settings  # noqa: E402

from research_utils import load_persona_entries  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "evaluation" / "baselines" / "p1_baseline.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "baselines" / "field_interdependence.json"

MUTATION_GROUPS = {
    "swap_firmographics": ("firmographics",),
    "swap_goals_bundle": ("goals", "pains", "motivations", "objections", "decision_triggers"),
    "swap_voice_bundle": ("channels", "vocabulary", "sample_quotes"),
}


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def _swap_fields(base_persona: dict, donor_persona: dict, fields: tuple[str, ...]) -> dict:
    mutated = copy.deepcopy(base_persona)
    for field_name in fields:
        mutated[field_name] = copy.deepcopy(donor_persona.get(field_name))
    return mutated


async def _judge_preference(original: dict, mutated: dict) -> str:
    backend = build_judge_backend_from_settings(model=settings.resolved_judge_model)
    judge = LLMJudge(backend=backend)
    result = await judge.pairwise(original, mutated)
    return result.winner


async def main(input_path: Path, output_path: Path, *, temperature: float | None, run_judge: bool) -> None:
    settings.validate_runtime_settings()
    artifact, entries = load_persona_entries(input_path, temperature=temperature)
    if len(entries) < 2:
        raise ValueError("Field interdependence harness needs at least two personas to swap bundles")

    report_rows: list[dict] = []
    aggregate: dict[str, dict[str, float | int | None]] = {}

    for mutation_name, fields in MUTATION_GROUPS.items():
        drops: list[float] = []
        judge_original_wins = 0
        judge_mutant_wins = 0
        judge_ties = 0

        for idx, entry in enumerate(entries):
            donor = entries[(idx + 1) % len(entries)]
            original_persona = entry["persona"]
            mutated_persona = _swap_fields(original_persona, donor["persona"], fields)

            original_breakdown = field_interdependence_breakdown(original_persona)
            mutated_breakdown = field_interdependence_breakdown(mutated_persona)
            original_score = original_breakdown["overall"]
            mutated_score = mutated_breakdown["overall"]
            score_drop = (
                original_score - mutated_score
                if not math.isnan(original_score) and not math.isnan(mutated_score)
                else float("nan")
            )
            if not math.isnan(score_drop):
                drops.append(score_drop)

            judge_winner = None
            if run_judge:
                judge_winner = await _judge_preference(original_persona, mutated_persona)
                if judge_winner == "a":
                    judge_original_wins += 1
                elif judge_winner == "b":
                    judge_mutant_wins += 1
                else:
                    judge_ties += 1

            report_rows.append({
                "mutation": mutation_name,
                "fields": list(fields),
                "persona_name": entry.get("name"),
                "cluster_id": entry.get("cluster_id"),
                "donor_name": donor.get("name"),
                "original_score": None if math.isnan(original_score) else round(original_score, 6),
                "mutated_score": None if math.isnan(mutated_score) else round(mutated_score, 6),
                "score_drop": None if math.isnan(score_drop) else round(score_drop, 6),
                "original_breakdown": original_breakdown,
                "mutated_breakdown": mutated_breakdown,
                "judge_pairwise_winner": judge_winner,
            })

        aggregate[mutation_name] = {
            "num_trials": len(entries),
            "mean_score_drop": round(sum(drops) / len(drops), 6) if drops else None,
            "max_score_drop": round(max(drops), 6) if drops else None,
            "min_score_drop": round(min(drops), 6) if drops else None,
            "judge_original_win_rate": round(judge_original_wins / len(entries), 6) if run_judge else None,
            "judge_mutant_win_rate": round(judge_mutant_wins / len(entries), 6) if run_judge else None,
            "judge_tie_rate": round(judge_ties / len(entries), 6) if run_judge else None,
        }

    report = {
        "experiment_id": "1.07",
        "name": "Field interdependence harness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_artifact": str(input_path),
        "source_artifact_type": artifact.get("experiment_id", artifact.get("version", "unknown")),
        "selected_temperature": temperature,
        "model": settings.resolved_judge_model if run_judge else None,
        "run_judge": run_judge,
        "mutations": aggregate,
        "rows": report_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"Saved: {output_path}")
    for mutation_name, stats in aggregate.items():
        print(
            f"{mutation_name}: "
            f"drop={_fmt(stats['mean_score_drop'])} "
            f"judge_original_win_rate={_fmt(stats['judge_original_win_rate'])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment 1.07 field interdependence harness")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Baseline or temp sweep artifact")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save report JSON")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="If input is a temp sweep artifact, select this temperature instead of best_temperature",
    )
    parser.add_argument(
        "--run-judge",
        action="store_true",
        help="Also run pairwise judge on original vs mutated personas",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.input,
            args.output,
            temperature=args.temperature,
            run_judge=args.run_judge,
        ),
    )
