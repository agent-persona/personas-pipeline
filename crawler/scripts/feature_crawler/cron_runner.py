#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "feature_crawler" / "config" / "crawl_jobs.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "feature_crawler" / "logs"


def load_jobs(config_path: Path) -> list[dict[str, object]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("config must include a jobs array")
    normalized: list[dict[str, object]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        command = job.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError(f"job {job.get('name', 'unknown')} has invalid command")
        normalized.append(job)
    return normalized


def run_job(job: dict[str, object], log_dir: Path, dry_run: bool) -> int:
    name = str(job.get("name") or "unnamed-job")
    command = [str(item) for item in job["command"]]
    print(f"[{timestamp()}] job={name} command={' '.join(command)}")
    if dry_run:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{slug(name)}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{timestamp()}] START {name}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"[{timestamp()}] END {name} exit={result.returncode}\n")
    return result.returncode


def timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    compact = "".join(chars).strip("-")
    while "--" in compact:
        compact = compact.replace("--", "-")
    return compact or "job"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scheduled crawler jobs from config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--job", action="append", default=[], help="Only run selected job name(s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    jobs = load_jobs(config_path)
    selected = set(args.job)
    exit_code = 0
    for job in jobs:
        enabled = bool(job.get("enabled", True))
        name = str(job.get("name") or "unnamed-job")
        if not enabled:
            continue
        if selected and name not in selected:
            continue
        result = run_job(job, Path(args.log_dir).expanduser().resolve(), args.dry_run)
        if result != 0:
            exit_code = result
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
