"""Auto-merge schema-additive and eval-only branches.

For each branch:
  1. Find files that are NEW on the branch (don't exist on main)
  2. Skip noise (docs/, output/, benchmark/, .gitignore, *.md root files)
  3. Bring those files into the current branch via `git checkout origin/<branch> -- <file>`
  4. Commit with a descriptive message naming the branch + verdict

Skips files we don't want from experiment branches:
  - benchmark/  (we have our own)
  - docs/plans/ (researcher logs, not for production merge)
  - output/    (artifacts)
  - .gitignore (each branch has its own)

Writes to a log so the run is auditable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()

# Patterns to NEVER bring in from experiment branches
SKIP_PREFIXES = (
    "benchmark/",
    "docs/plans/",
    "output/",
    ".gitignore",
    ".claude/",
)
SKIP_FILES = {
    "MERGE_DECISIONS.md", "PRD_EXPERIMENT_MERGES.md",
    "REMAINING_EXPERIMENTS_PLAN.md", "ASSIGNMENTS.md",
    "schedule.html", "SEGMENTATION_EXPERIMENTS_PROMPT.md",
}


def git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr}")
    return r.stdout


def file_exists_on_main(path: str) -> bool:
    r = subprocess.run(
        ["git", "cat-file", "-e", f"main:{path}"],
        cwd=REPO_ROOT, capture_output=True,
    )
    return r.returncode == 0


def files_changed_on_branch(branch: str) -> list[str]:
    """All files different between main and the branch."""
    out = git(["diff", "--name-only", f"main...origin/{branch}"])
    return [f.strip() for f in out.split("\n") if f.strip()]


def should_keep(path: str) -> bool:
    """Filter: keep this file when bringing it in?"""
    if path in SKIP_FILES:
        return False
    for p in SKIP_PREFIXES:
        if path.startswith(p):
            return False
    if "__pycache__" in path or path.endswith(".pyc"):
        return False
    return True


def merge_branch(branch: str, verdict: str = "MERGE-AS-OPTION") -> dict:
    """Bring new files from branch into current working tree, commit."""
    report = {"branch": branch, "verdict": verdict, "status": "pending",
              "files_added": [], "files_modified_skipped": [], "error": ""}

    try:
        all_files = files_changed_on_branch(branch)
        new_files = []
        modified_existing = []
        for f in all_files:
            if not should_keep(f):
                continue
            if file_exists_on_main(f):
                # Modified existing file — skip for safety
                modified_existing.append(f)
            else:
                new_files.append(f)

        if not new_files:
            report["status"] = "no_new_files"
            report["files_modified_skipped"] = modified_existing
            return report

        # Check if any of the "new" files already exist on our current branch
        # (i.e., another branch beat us to adding them)
        truly_new = []
        already_exist = []
        for f in new_files:
            if (REPO_ROOT / f).exists():
                already_exist.append(f)
            else:
                truly_new.append(f)

        report["files_already_exist"] = already_exist

        if not truly_new:
            report["status"] = "all_files_already_present"
            return report

        # Bring in files
        # Use one git checkout call with all files
        cmd = ["git", "checkout", f"origin/{branch}", "--"] + truly_new
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            report["status"] = "checkout_failed"
            report["error"] = r.stderr.strip()[:200]
            return report

        report["files_added"] = truly_new

        # Stage and commit
        subprocess.run(["git", "add"] + truly_new, cwd=REPO_ROOT, capture_output=True)

        # Check if anything actually staged
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not r.stdout.strip():
            report["status"] = "nothing_to_commit"
            return report

        msg = f"merge {branch} ({verdict})\n\n"
        msg += f"Source: {branch}\n"
        msg += f"Verdict: {verdict}\n"
        msg += f"Adds {len(truly_new)} new file(s).\n\n"
        if modified_existing:
            msg += f"Skipped {len(modified_existing)} modified existing file(s) "
            msg += "to avoid silent overwrites:\n"
            for f in modified_existing[:10]:
                msg += f"  - {f}\n"
            if len(modified_existing) > 10:
                msg += f"  ... and {len(modified_existing) - 10} more\n"
        msg += "\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

        commit_proc = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if commit_proc.returncode != 0:
            report["status"] = "commit_failed"
            report["error"] = commit_proc.stderr.strip()[:200]
            return report

        report["status"] = "ok"

    except Exception as e:
        report["status"] = "exception"
        report["error"] = f"{type(e).__name__}: {e}"[:300]

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--types", default="eval-only,schema-additive",
                        help="Comma-separated branch types to merge")
    parser.add_argument("--branch", action="append",
                        help="Specific branch(es) only")
    parser.add_argument("--log", type=Path,
                        default=Path("benchmark/automerge.log.json"))
    args = parser.parse_args()

    classification = json.loads(args.classification.read_text())
    types_to_merge = set(args.types.split(","))

    if args.branch:
        targets = [d for d in classification if d["branch"] in args.branch]
    else:
        targets = [d for d in classification if d["type"] in types_to_merge]

    print(f"Auto-merging {len(targets)} branches")
    print(f"Types: {types_to_merge}")
    print()

    reports = []
    counts: dict[str, int] = {}
    for d in sorted(targets, key=lambda x: x["branch"]):
        verdict_label = "MERGE-AS-OPTION (eval-only)" if d["type"] == "eval-only" \
            else "MERGE-AS-OPTION (schema-additive)"
        r = merge_branch(d["branch"], verdict=verdict_label)
        reports.append(r)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        n_added = len(r.get("files_added", []))
        n_skipped = len(r.get("files_modified_skipped", []))
        n_already = len(r.get("files_already_exist", []))
        flag = {
            "ok": "[OK]",
            "no_new_files": "[NONE]",
            "all_files_already_present": "[DUPE]",
            "nothing_to_commit": "[EMPTY]",
            "commit_failed": "[CFAIL]",
            "checkout_failed": "[CHKFAIL]",
            "exception": "[EXC]",
        }.get(r["status"], "[?]")
        print(f"  {flag:<8} {d['branch']:<48} new={n_added:>2} skip={n_skipped:>2} dup={n_already:>2}")
        if r.get("error"):
            print(f"           -> {r['error'][:100]}")

    print()
    print("Status counts:")
    for s, n in sorted(counts.items()):
        print(f"  {s}: {n}")

    args.log.write_text(json.dumps(reports, indent=2))
    print(f"\nLog: {args.log}")


if __name__ == "__main__":
    main()
