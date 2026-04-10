"""cache_analysis.py — Experiment 2.18: measure cacheable token fraction.

Reads cluster JSON files, builds cached messages via build_cached_messages(),
and reports the static vs dynamic token split without making any API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the worktree root without installing the package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.engine.prompt_builder import build_cached_messages  # noqa: E402

CLUSTER_DIR = REPO_ROOT / "output" / "clusters"
CLUSTER_FILES = ["cluster_00.json", "cluster_01.json"]


def analyse_cluster(path: Path) -> dict:
    with path.open() as fh:
        raw = json.load(fh)
    cluster = ClusterData.model_validate(raw)
    _, token_counts = build_cached_messages(cluster)
    return {
        "cluster": path.stem,
        **token_counts,
    }


def main() -> None:
    results = []
    for name in CLUSTER_FILES:
        p = CLUSTER_DIR / name
        if not p.exists():
            print(f"WARNING: {p} not found — skipping", file=sys.stderr)
            continue
        results.append(analyse_cluster(p))

    if not results:
        print(json.dumps({"error": "no cluster files found"}))
        sys.exit(1)

    avg_fraction = sum(r["cacheable_fraction"] for r in results) / len(results)
    output = {
        "clusters": results,
        "average_cacheable_fraction": round(avg_fraction, 4),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
