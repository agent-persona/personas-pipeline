"""Orchestrates parallel benchmark runs across all runtime-changing branches.

For each branch:
  1. Create an isolated git worktree at .worktrees/<branch>
  2. Copy benchmark/ and synthesis/.env into it
  3. Run python benchmark/run.py --output <results-dir>
  4. Move the results back to benchmark/results/<branch>/ on dev
  5. Remove the worktree

Concurrency is bounded by MAX_PARALLEL to stay within API rate limits and
avoid exhausting local disk/IO.

Usage:
  python benchmark/sweep.py --classification benchmark/branch_classification.json \
       --output-root benchmark/results \
       --parallel 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
WORKTREE_ROOT = REPO_ROOT / ".worktrees"


async def run_branch(
    branch: str,
    output_root: Path,
    semaphore: asyncio.Semaphore,
    skip_existing: bool = True,
) -> dict:
    """Run the benchmark for a single branch in an isolated worktree."""
    result_dir = output_root / branch
    report = {"branch": branch, "status": "pending", "elapsed_s": 0, "error": ""}

    if skip_existing and (result_dir / "summary.json").exists():
        report["status"] = "skipped_existing"
        return report

    async with semaphore:
        t0 = time.monotonic()
        wt_path = WORKTREE_ROOT / branch.replace("/", "_")
        try:
            # Create worktree (remove if stale)
            if wt_path.exists():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt_path)],
                    cwd=REPO_ROOT, capture_output=True,
                )
            r = subprocess.run(
                ["git", "worktree", "add", "--detach", str(wt_path), f"origin/{branch}"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            if r.returncode != 0:
                report["status"] = "worktree_failed"
                report["error"] = r.stderr.strip()[:200]
                return report

            # Copy benchmark/ from dev (current directory)
            shutil.copytree(REPO_ROOT / "benchmark", wt_path / "benchmark",
                            dirs_exist_ok=True)
            # Copy .env for API key
            env_src = REPO_ROOT / "synthesis" / ".env"
            env_dst = wt_path / "synthesis" / ".env"
            if env_src.exists() and env_dst.parent.exists():
                shutil.copy2(env_src, env_dst)

            # Run benchmark in the worktree
            local_out = wt_path / "bench-out"
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "benchmark/run.py", "--output", str(local_out),
                cwd=wt_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1500)
            except asyncio.TimeoutError:
                proc.kill()
                report["status"] = "timeout"
                return report

            if proc.returncode != 0:
                report["status"] = "run_failed"
                report["error"] = (stderr.decode(errors="replace")[-500:]
                                   if stderr else "unknown")
                report["stdout_tail"] = (stdout.decode(errors="replace")[-500:]
                                        if stdout else "")
                return report

            # Move results back to dev
            result_dir.mkdir(parents=True, exist_ok=True)
            for f in local_out.iterdir():
                shutil.copy2(f, result_dir / f.name)
            report["status"] = "ok"

        except Exception as e:
            report["status"] = "exception"
            report["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            # Clean up worktree
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=REPO_ROOT, capture_output=True,
            )
            report["elapsed_s"] = time.monotonic() - t0

    return report


async def main(
    classification_file: Path,
    output_root: Path,
    parallel: int,
    branch_filter: list[str] | None = None,
) -> None:
    data = json.loads(classification_file.read_text())
    runtime_branches = [
        d["branch"] for d in data
        if d["type"] == "runtime-changing"
    ]

    if branch_filter:
        runtime_branches = [b for b in runtime_branches if b in branch_filter]

    print(f"Sweeping {len(runtime_branches)} runtime-changing branches "
          f"(parallel={parallel})")
    print(f"Output: {output_root}")
    print()

    output_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(parallel)

    t_start = time.monotonic()
    progress = {"completed": 0, "total": len(runtime_branches)}

    async def run_with_progress(b: str) -> dict:
        r = await run_branch(b, output_root, semaphore)
        progress["completed"] += 1
        status = r["status"]
        elapsed = r.get("elapsed_s", 0)
        tag = {
            "ok": "[OK]",
            "skipped_existing": "[SKIP]",
            "timeout": "[TIMEOUT]",
            "run_failed": "[FAIL]",
            "worktree_failed": "[WT-FAIL]",
            "exception": "[EXC]",
        }.get(status, "[?]")
        print(
            f"  {progress['completed']:>3}/{progress['total']} {tag:<10} "
            f"{b:<45} ({elapsed:.0f}s)",
            flush=True,
        )
        if status not in ("ok", "skipped_existing"):
            err = r.get("error", "")[:150]
            print(f"        -> {err}", flush=True)
        return r

    reports = await asyncio.gather(*[run_with_progress(b) for b in runtime_branches])

    total = time.monotonic() - t_start
    counts: dict[str, int] = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print()
    print(f"Sweep done in {total:.0f}s ({total/60:.1f} min)")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")

    # Save sweep log
    log_path = output_root / "_sweep_log.json"
    log_path.write_text(json.dumps({
        "branches": reports,
        "total_elapsed_s": total,
        "parallel": parallel,
    }, indent=2, default=str))
    print(f"Sweep log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--branch", action="append", help="Restrict to specific branches")
    args = parser.parse_args()

    asyncio.run(main(
        args.classification, args.output_root,
        args.parallel, branch_filter=args.branch,
    ))
