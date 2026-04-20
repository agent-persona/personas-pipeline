"""Experiment run registry.

Per the lab harness paragraph in `PRD_LAB_RESEARCH.md`:

    "A run registry where every experiment carries an experiment_id, a config
     diff, a written hypothesis, a control, a sample size, a result, and a
     decision (adopt / reject / defer). No ad-hoc runs."

Append-only JSONL store, keyed by experiment_id. Every experiment in spaces 1-6
should write to this registry rather than dropping CSVs in scratch directories.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "registry.jsonl"


@dataclass
class ExperimentRun:
    """One row in the run registry — the seven mandated fields plus metadata."""

    experiment_id: str  # e.g. "1.09"
    title: str
    group: int  # 1-6 (problem space)
    size: str  # "S" | "M" | "L"
    hypothesis: str
    control: str
    variant: str
    sample_size: int = 0
    metrics: dict = field(default_factory=dict)
    result: str = ""
    decision: str = "pending"  # "adopt" | "reject" | "defer" | "pending"
    config_diff: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: int = 0
    cost_usd: float = 0.0
    owner: str = "yash"
    tenant_id: str = "tenant_acme_corp"


def save_run(run: ExperimentRun, path: Optional[Path] = None) -> None:
    """Append an experiment run to the registry."""
    target = path or REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a") as f:
        f.write(json.dumps(asdict(run)) + "\n")


def load_runs(path: Optional[Path] = None) -> list[dict]:
    """Load all experiment runs from the registry."""
    target = path or REGISTRY_PATH
    if not target.exists():
        return []
    runs = []
    for line in target.read_text().strip().split("\n"):
        if line:
            runs.append(json.loads(line))
    return runs


def latest_per_experiment(path: Optional[Path] = None) -> dict[str, dict]:
    """Return only the latest run per experiment_id (preferring entries with metrics)."""
    runs = load_runs(path)
    out: dict[str, dict] = {}
    for r in runs:
        eid = r["experiment_id"]
        # Prefer runs that have actual metric data over earlier failed attempts
        if eid not in out:
            out[eid] = r
        elif r.get("metrics") and not out[eid].get("metrics"):
            out[eid] = r
        elif r.get("metrics") and out[eid].get("metrics"):
            out[eid] = r  # latest wins among successful runs
    return out


def get_run(experiment_id: str, path: Optional[Path] = None) -> dict | None:
    """Get the latest run for a specific experiment."""
    return latest_per_experiment(path).get(experiment_id)


def summary_table(path: Optional[Path] = None) -> str:
    """Format the registry as a printable summary table."""
    runs = latest_per_experiment(path)
    if not runs:
        return "No experiments recorded."

    lines = [f"{'ID':<6} {'Title':<40} {'Decision':<10} {'Result':<60}"]
    lines.append("-" * 120)
    for eid in sorted(runs.keys()):
        r = runs[eid]
        title = r["title"][:40]
        decision = r["decision"]
        result = r.get("result", "")[:60]
        lines.append(f"{eid:<6} {title:<40} {decision:<10} {result}")
    return "\n".join(lines)
