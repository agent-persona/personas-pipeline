"""Classify each exp-* branch as eval-only, schema-additive, or runtime-changing.

Runtime-changing = modifies any of these files:
  synthesis/synthesis/engine/prompt_builder.py
  synthesis/synthesis/engine/synthesizer.py
  synthesis/synthesis/engine/model_backend.py
  synthesis/synthesis/engine/groundedness.py
  synthesis/synthesis/models/persona.py
  synthesis/synthesis/models/evidence.py
  synthesis/synthesis/models/cluster.py
  twin/twin/chat.py
  evaluation/evaluation/judges.py
  segmentation/segmentation/engine/*.py
  segmentation/segmentation/pipeline.py
  crawler/crawler/pipeline.py
  crawler/crawler/connectors/__init__.py  (if DEFAULT_CONNECTORS changed)

Schema-additive = only adds new files to synthesis/synthesis/models/ or
new modules (no changes to existing imported files).

Eval-only = only changes files under evals/ or benchmark/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNTIME_FILES = {
    "synthesis/synthesis/engine/prompt_builder.py",
    "synthesis/synthesis/engine/synthesizer.py",
    "synthesis/synthesis/engine/model_backend.py",
    "synthesis/synthesis/engine/groundedness.py",
    "synthesis/synthesis/models/persona.py",
    "synthesis/synthesis/models/evidence.py",
    "synthesis/synthesis/models/cluster.py",
    "synthesis/synthesis/config.py",
    "twin/twin/chat.py",
    "twin/twin/__init__.py",
    "evaluation/evaluation/judges.py",
    "evaluation/evaluation/metrics.py",
    "segmentation/segmentation/pipeline.py",
    "segmentation/segmentation/engine/clusterer.py",
    "segmentation/segmentation/engine/featurizer.py",
    "segmentation/segmentation/engine/summarizer.py",
    "segmentation/segmentation/models/record.py",
    "segmentation/segmentation/models/features.py",
    "crawler/crawler/pipeline.py",
    "crawler/crawler/connectors/__init__.py",
    "crawler/crawler/base.py",
    "crawler/crawler/models.py",
}


def classify_branch(branch: str) -> dict:
    """Determine what type of branch this is by looking at diff vs main."""
    # Get list of changed files vs main
    result = subprocess.run(
        ["git", "diff", "--name-only", f"main...origin/{branch}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"branch": branch, "type": "error", "error": result.stderr.strip()}

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    if not files:
        return {"branch": branch, "type": "empty", "files": []}

    # Filter out noise
    real_files = [f for f in files if not f.endswith(".pyc") and "__pycache__" not in f]

    runtime_changed = [f for f in real_files if f in RUNTIME_FILES]

    # Anything else is a "new file" if it didn't exist on main
    new_files = []
    non_runtime = []
    for f in real_files:
        if f in RUNTIME_FILES:
            continue
        # Check if file exists on main
        r = subprocess.run(
            ["git", "cat-file", "-e", f"main:{f}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            new_files.append(f)
        else:
            non_runtime.append(f)

    # Classify
    if runtime_changed:
        bucket = "runtime-changing"
    elif non_runtime:
        # Modified a non-runtime existing file (e.g., README, config)
        # treat as schema-additive if new files exist, else eval-only
        bucket = "schema-additive" if new_files else "eval-only"
    elif new_files:
        # Only new files
        eval_only = all(
            f.startswith("evals/") or f.startswith("benchmark/")
            or f.endswith(".md") or f.endswith(".html")
            for f in new_files
        )
        bucket = "eval-only" if eval_only else "schema-additive"
    else:
        bucket = "empty"

    return {
        "branch": branch,
        "type": bucket,
        "runtime_files_changed": runtime_changed,
        "new_files": new_files,
        "other_modified": non_runtime,
        "total_files": len(real_files),
    }


def main() -> None:
    # Get list of exp-* branches from remote
    result = subprocess.run(
        ["git", "branch", "-r"], capture_output=True, text=True,
    )
    branches = []
    for line in result.stdout.strip().split("\n"):
        b = line.strip()
        if b.startswith("origin/exp-") and "HEAD" not in b:
            branches.append(b.replace("origin/", ""))

    print(f"Classifying {len(branches)} branches...", file=sys.stderr)

    by_type: dict[str, list] = {
        "runtime-changing": [], "schema-additive": [], "eval-only": [],
        "empty": [], "error": [],
    }
    details = []
    for b in branches:
        c = classify_branch(b)
        details.append(c)
        by_type.setdefault(c["type"], []).append(b)

    # Output
    print(f"\n=== CLASSIFICATION SUMMARY ===\n")
    for t in ["runtime-changing", "schema-additive", "eval-only", "empty", "error"]:
        bs = by_type.get(t, [])
        print(f"\n{t.upper()} ({len(bs)}):")
        for b in sorted(bs):
            d = next(x for x in details if x["branch"] == b)
            extras = []
            if d.get("runtime_files_changed"):
                extras.append(f"rt={len(d['runtime_files_changed'])}")
            if d.get("new_files"):
                extras.append(f"new={len(d['new_files'])}")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            print(f"  {b}{extra_str}")

    # Save JSON
    out = Path("benchmark/branch_classification.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(details, indent=2))
    print(f"\nDetails saved to {out}")


if __name__ == "__main__":
    main()
