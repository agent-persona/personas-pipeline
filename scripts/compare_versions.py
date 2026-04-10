"""Compare two baseline JSON files and produce a version-comparison report.

Usage:
    python scripts/compare_versions.py \\
        --version-a evaluation/baselines/p1_baseline.json \\
        --version-b evaluation/baselines/p2_baseline.json \\
        [--output evaluation/baselines/comparison_p1_vs_p2.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from evaluation.judges import LLMJudge, build_judge_backend_from_settings  # noqa: E402
from synthesis.config import settings  # noqa: E402


def _comparison_metric_map(baseline: dict) -> tuple[dict, str]:
    comparison: dict = {}
    source_parts: list[str] = []

    summary = baseline.get("aggregate_summary")
    if isinstance(summary, dict):
        means = summary.get("means")
        if isinstance(means, dict) and means:
            comparison.update(means)
            source_parts.append("aggregate_summary.means")

    aggregate = baseline.get("aggregate", {})
    if isinstance(aggregate, dict):
        for key, value in aggregate.items():
            comparison.setdefault(key, value)
        source_parts.append("aggregate")

    if not comparison:
        return {}, "missing"
    return comparison, "+".join(source_parts)

def _load_baseline(path: Path) -> dict:
    """Load and validate a baseline JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")
    data = json.loads(path.read_text())
    for key in ("version", "per_persona", "aggregate"):
        if key not in data:
            raise ValueError(f"Baseline {path} missing required key: {key}")
    return data


def _match_personas(baseline_a: dict, baseline_b: dict, strict: bool = True) -> list[tuple[dict, dict]]:
    """Match personas from two baselines using stable keys.

    Priority: cluster_id (stable across versions) > name > index.
    Warns on any unmatched personas (or raises in strict mode).

    Args:
        baseline_a: First baseline dict
        baseline_b: Second baseline dict
        strict: If True, raise ValueError when personas cannot be fully matched.
               If False, warn and return partial matches.
    """
    personas_a = baseline_a["per_persona"]
    personas_b = baseline_b["per_persona"]

    # 1. Try cluster_id matching (most stable — same source = same cluster_id)
    ids_b = {p["cluster_id"]: p for p in personas_b if "cluster_id" in p}
    if ids_b:
        matched: list[tuple[dict, dict]] = []
        unmatched_a: list[str] = []
        for pa in personas_a:
            cid = pa.get("cluster_id")
            if cid and cid in ids_b:
                matched.append((pa, ids_b[cid]))
            else:
                unmatched_a.append(pa.get("name", "?"))
        matched_cids = {pa.get("cluster_id") for pa, _ in matched}
        unmatched_b = [p.get("name", "?") for p in personas_b if p.get("cluster_id") not in matched_cids]
        if unmatched_a or unmatched_b:
            msg = ""
            if unmatched_a:
                msg += f"{len(unmatched_a)} personas in A not matched by cluster_id: {unmatched_a}"
            if unmatched_b:
                if msg:
                    msg += "; "
                msg += f"{len(unmatched_b)} personas in B not matched by cluster_id: {unmatched_b}"
            if strict:
                raise ValueError(f"Personas failed to match (cluster_id matching): {msg}")
            else:
                print(f"  WARNING: {msg}")
        if matched:
            return matched

    # 2. Fall back to name matching
    names_b = {p["name"]: p for p in personas_b}
    matched = []
    for pa in personas_a:
        if pa["name"] in names_b:
            matched.append((pa, names_b[pa["name"]]))
    if matched:
        unmatched = len(personas_a) - len(matched)
        if unmatched:
            msg = f"{unmatched} personas unmatched by name (name fallback)"
            if strict:
                raise ValueError(f"Personas failed to match (name matching): {msg}")
            else:
                print(f"  WARNING: {msg}")
        return matched

    # 3. Last resort: index matching — never safe for decision-grade comparisons
    if strict:
        raise ValueError(
            "Personas failed to match: no cluster_id or name matches found. "
            "Index fallback is not allowed in strict mode because it can pair "
            "wrong personas and produce false winners. Use --allow-partial for exploratory runs."
        )
    n = min(len(personas_a), len(personas_b))
    if len(personas_a) != len(personas_b):
        print(f"  WARNING: index fallback with unequal counts ({len(personas_a)} vs {len(personas_b)}), using first {n}")
    else:
        print(f"  WARNING: index fallback — no stable keys matched, pairing by position ({n} pairs)")
    return [(personas_a[i], personas_b[i]) for i in range(n)]


def _safe_delta(a: float | None, b: float | None) -> float | None:
    """Compute b - a, returning None if either is None or NaN."""
    if a is None or b is None:
        return None
    if math.isnan(a) or math.isnan(b):
        return None
    return b - a


def _winner_from_delta(delta: float | None) -> str:
    if delta is None:
        return "N/A"
    if delta > 0.005:
        return "b"
    if delta < -0.005:
        return "a"
    return "tie"


def compute_metric_deltas(baseline_a: dict, baseline_b: dict) -> list[dict]:
    """Compute per-metric deltas between aggregate scores."""
    agg_a, _ = _comparison_metric_map(baseline_a)
    agg_b, _ = _comparison_metric_map(baseline_b)

    all_metrics = sorted(set(agg_a.keys()) | set(agg_b.keys()))
    deltas = []

    for metric in all_metrics:
        score_a = agg_a.get(metric)
        score_b = agg_b.get(metric)
        delta = _safe_delta(score_a, score_b)
        deltas.append({
            "metric": metric,
            "score_a": score_a,
            "score_b": score_b,
            "delta": delta,
            "winner": _winner_from_delta(delta),
        })

    return deltas


async def run_pairwise(pairs: list[tuple[dict, dict]]) -> dict:
    """Run pairwise LLM judge comparisons on matched persona pairs."""
    settings.validate_runtime_settings()
    backend = build_judge_backend_from_settings(model=settings.resolved_judge_model)
    judge = LLMJudge(backend=backend)
    wins_a = 0
    wins_b = 0
    ties = 0

    for pa, pb in pairs:
        # Bug fix #3: send full persona JSON, not the metrics dict
        persona_a = pa.get("persona", pa)
        persona_b = pb.get("persona", pb)
        # Bug fix #2: pairwise() returns PairwiseResult, not a tuple
        result = await judge.pairwise(persona_a, persona_b)
        if result.winner == "a":
            wins_a += 1
        elif result.winner == "b":
            wins_b += 1
        else:
            ties += 1

    total = len(pairs)
    if total == 0:
        return {"rate_a": 0.0, "rate_b": 0.0, "tie_rate": 0.0}

    return {
        "rate_a": round(wins_a / total, 4),
        "rate_b": round(wins_b / total, 4),
        "tie_rate": round(ties / total, 4),
    }


def determine_overall_winner(metric_deltas: list[dict], pairwise: dict) -> str:
    """Determine overall winner from metric deltas and pairwise results."""
    # Count metric wins (excluding N/A)
    a_wins = sum(1 for d in metric_deltas if d["winner"] == "a")
    b_wins = sum(1 for d in metric_deltas if d["winner"] == "b")

    # Weight pairwise heavily
    if pairwise["rate_b"] > pairwise["rate_a"] + 0.1:
        b_wins += 2
    elif pairwise["rate_a"] > pairwise["rate_b"] + 0.1:
        a_wins += 2

    if b_wins > a_wins:
        return "b"
    if a_wins > b_wins:
        return "a"
    return "tie"


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if math.isnan(v):
            return "N/A"
        return f"{v:.4f}"
    return str(v)


async def main(path_a: Path, path_b: Path, output_path: Path | None, strict: bool = True) -> None:
    print("=" * 72)
    print("VERSION COMPARISON")
    print("=" * 72)

    baseline_a = _load_baseline(path_a)
    baseline_b = _load_baseline(path_b)

    ver_a = baseline_a["version"]
    ver_b = baseline_b["version"]
    print(f"  Version A: {ver_a} ({path_a.name})")
    print(f"  Version B: {ver_b} ({path_b.name})")

    # Match personas
    pairs = _match_personas(baseline_a, baseline_b, strict=strict)
    print(f"  Matched pairs: {len(pairs)}")

    # Metric deltas
    print("\n" + "=" * 72)
    print("METRIC DELTAS (B - A)")
    print("=" * 72)
    _, source_a = _comparison_metric_map(baseline_a)
    _, source_b = _comparison_metric_map(baseline_b)
    print(f"  Metric source A: {source_a}")
    print(f"  Metric source B: {source_b}")
    metric_deltas = compute_metric_deltas(baseline_a, baseline_b)

    print(f"\n  {'Metric':<28s} {'A':>10s} {'B':>10s} {'Delta':>10s} {'Winner':>8s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for d in metric_deltas:
        print(
            f"  {d['metric']:<28s} "
            f"{_fmt(d['score_a']):>10s} "
            f"{_fmt(d['score_b']):>10s} "
            f"{_fmt(d['delta']):>10s} "
            f"{d['winner']:>8s}"
        )

    # Pairwise judging
    print("\n" + "=" * 72)
    print("PAIRWISE JUDGING")
    print("=" * 72)
    pairwise = await run_pairwise(pairs)
    print(f"  A win rate:   {pairwise['rate_a']:.2%}")
    print(f"  B win rate:   {pairwise['rate_b']:.2%}")
    print(f"  Tie rate:     {pairwise['tie_rate']:.2%}")

    # Overall winner
    overall = determine_overall_winner(metric_deltas, pairwise)

    print("\n" + "=" * 72)
    winner_label = {"a": ver_a, "b": ver_b, "tie": "TIE"}.get(overall, overall)
    print(f"OVERALL WINNER: {winner_label}")
    print("=" * 72)

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version_a": ver_a,
        "version_b": ver_b,
        "file_a": str(path_a),
        "file_b": str(path_b),
        "num_matched_pairs": len(pairs),
        "comparison_metric_source": {
            "a": source_a,
            "b": source_b,
        },
        "per_metric": metric_deltas,
        "pairwise": pairwise,
        "overall_winner": overall,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two persona baselines")
    parser.add_argument(
        "--version-a",
        type=Path,
        required=True,
        help="Path to baseline A JSON",
    )
    parser.add_argument(
        "--version-b",
        type=Path,
        required=True,
        help="Path to baseline B JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save comparison report JSON",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=False,
        help="Allow partial persona matches with warnings (for exploratory runs). Default: strict mode (fails on unmatched personas)",
    )
    args = parser.parse_args()
    strict = not args.allow_partial
    asyncio.run(main(args.version_a, args.version_b, args.output, strict))
