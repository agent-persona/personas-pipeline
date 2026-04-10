"""Export the selected winner from temp_sweep.json as a baseline-like artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = REPO_ROOT / "evaluation" / "baselines" / "temp_sweep.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "baselines" / "p1_temp_winner.json"


def main(input_path: Path, output_path: Path, *, temperature: float | None) -> None:
    report = json.loads(input_path.read_text())
    selected_temperature = temperature
    if selected_temperature is None:
        selected_temperature = report.get("selection", {}).get("best_temperature")
    if selected_temperature is None:
        raise ValueError("No best_temperature found; pass --temperature explicitly")

    winner = None
    for result in report.get("results", []):
        if result.get("temperature") == selected_temperature:
            winner = result
            break
    if winner is None:
        raise ValueError(f"Temperature {selected_temperature} not found in {input_path}")
    if winner.get("status") == "failed":
        raise ValueError(f"Temperature {selected_temperature} failed and cannot be frozen")

    baseline = {
        "version": f"p1-temp-{selected_temperature}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": report.get("model"),
        "source_experiment_id": report.get("experiment_id"),
        "source_temperature": selected_temperature,
        "run_metadata": report.get("run_metadata", {}),
        "num_personas": winner.get("num_personas"),
        "total_cost_usd": winner.get("total_cost_usd"),
        "runs": winner.get("runs", []),
        "per_persona": winner.get("per_persona", []),
        "stability_breakdown": winner.get("stability_breakdown"),
        "aggregate": winner.get("aggregate", {}),
        "aggregate_summary": winner.get("aggregate_summary", {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2, default=str))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze winning temp sweep result as a baseline artifact")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Temp sweep report")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Baseline-like output path")
    parser.add_argument("--temperature", type=float, default=None, help="Explicit temperature to export")
    args = parser.parse_args()
    main(args.input, args.output, temperature=args.temperature)
