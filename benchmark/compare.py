"""Compare benchmark results across branches.

Usage:
  # Compare one branch vs main baseline
  python benchmark/compare.py --baseline benchmark/results/main --branch benchmark/results/exp-3.06

  # Full sweep summary — all branches vs main
  python benchmark/compare.py --baseline benchmark/results/main --all benchmark/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(result_dir: Path) -> dict:
    """Load all tenant results from a directory."""
    if not result_dir.exists():
        return {"tenants": {}, "summary": {}}
    tenants = {}
    for tenant_file in result_dir.glob("bench_*.json"):
        tenant_name = tenant_file.stem
        tenants[tenant_name] = json.loads(tenant_file.read_text())
    summary = {}
    summary_file = result_dir / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
    return {"tenants": tenants, "summary": summary}


def aggregate(results: dict) -> dict:
    """Compute aggregate metrics across all tenants."""
    tenants = results["tenants"]
    if not tenants:
        return {}

    all_groundedness = []
    total_cost = 0.0
    total_personas = 0
    total_ok = 0
    all_attempts = []
    total_latency = 0.0

    for t in tenants.values():
        for c in t.get("clusters", []):
            if not c.get("failed"):
                all_groundedness.append(c.get("groundedness", 0))
                all_attempts.append(c.get("attempts", 0))
                total_ok += 1
            total_personas += 1
            total_cost += c.get("cost_usd", 0)
        total_latency += t.get("total_latency_s", 0)

    return {
        "n_tenants": len(tenants),
        "n_personas": total_personas,
        "n_personas_ok": total_ok,
        "success_rate": total_ok / total_personas if total_personas else 0,
        "mean_groundedness": sum(all_groundedness) / len(all_groundedness) if all_groundedness else 0,
        "mean_attempts": sum(all_attempts) / len(all_attempts) if all_attempts else 0,
        "total_cost_usd": total_cost,
        "total_latency_s": total_latency,
    }


def judge_aggregate(results: dict) -> dict:
    """Compute aggregate judge scores if they're in the results."""
    tenants = results["tenants"]
    all_scores = []
    for t in tenants.values():
        for c in t.get("clusters", []):
            if c.get("failed"):
                continue
            js = c.get("judge_score")
            if js and not isinstance(js, str):
                overall = js.get("overall")
                if overall is not None:
                    all_scores.append(float(overall))
    if not all_scores:
        return {"n_scored": 0}
    return {
        "n_scored": len(all_scores),
        "mean_judge": sum(all_scores) / len(all_scores),
        "min_judge": min(all_scores),
        "max_judge": max(all_scores),
    }


def compare_one(baseline: dict, branch: dict, name: str) -> dict:
    """Compute deltas between baseline and a branch."""
    b_agg = aggregate(baseline)
    br_agg = aggregate(branch)
    b_judge = judge_aggregate(baseline)
    br_judge = judge_aggregate(branch)

    def delta(a: float, b: float) -> float:
        return b - a

    def pct_delta(a: float, b: float) -> float:
        return ((b - a) / a * 100) if a else 0.0

    d = {
        "branch": name,
        "baseline": b_agg,
        "branch_metrics": br_agg,
        "deltas": {
            "groundedness": delta(b_agg.get("mean_groundedness", 0), br_agg.get("mean_groundedness", 0)),
            "cost_pct": pct_delta(b_agg.get("total_cost_usd", 0), br_agg.get("total_cost_usd", 0)),
            "success_rate": delta(b_agg.get("success_rate", 0), br_agg.get("success_rate", 0)),
            "attempts": delta(b_agg.get("mean_attempts", 0), br_agg.get("mean_attempts", 0)),
            "latency_pct": pct_delta(b_agg.get("total_latency_s", 0), br_agg.get("total_latency_s", 0)),
        },
    }
    if b_judge.get("n_scored") and br_judge.get("n_scored"):
        d["deltas"]["judge"] = delta(
            b_judge.get("mean_judge", 0), br_judge.get("mean_judge", 0)
        )
        d["baseline"]["mean_judge"] = b_judge.get("mean_judge")
        d["branch_metrics"]["mean_judge"] = br_judge.get("mean_judge")

    # Verdict
    dj = d["deltas"].get("judge", 0)
    dg = d["deltas"]["groundedness"]
    dc = d["deltas"]["cost_pct"]
    ds = d["deltas"]["success_rate"]

    if dj < -0.1 or dg < -0.05 or dc > 50 or ds < -0.05:
        verdict = "REJECT"
    elif dj >= 0.1 or (dg >= 0.05 and ds >= 0):
        if dc < 20:
            verdict = "MERGE"
        else:
            verdict = "MERGE-WITH-CAVEAT"
    elif abs(dj) <= 0.02 and abs(dg) <= 0.02 and abs(dc) <= 5 and abs(ds) <= 0.02:
        verdict = "NEUTRAL"
    else:
        verdict = "REVIEW"
    d["verdict"] = verdict
    return d


def print_compare(d: dict) -> None:
    deltas = d["deltas"]
    dj = deltas.get("judge", float("nan"))
    print(f"\n=== {d['branch']} — {d['verdict']} ===")
    print(f"  Δgroundedness: {deltas['groundedness']:+.3f}")
    print(f"  Δcost:         {deltas['cost_pct']:+.1f}%")
    print(f"  Δsuccess_rate: {deltas['success_rate']:+.3f}")
    print(f"  Δattempts:     {deltas['attempts']:+.2f}")
    if "judge" in deltas:
        print(f"  Δjudge:        {deltas['judge']:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--branch", type=Path, help="Single branch result dir")
    parser.add_argument("--all", type=Path, help="Parent dir with per-branch subdirs")
    parser.add_argument("--output", type=Path, help="Write full JSON report")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    if not baseline["tenants"]:
        print(f"ERROR: no results in {args.baseline}", file=sys.stderr)
        sys.exit(1)

    reports = []
    if args.branch:
        br = load_results(args.branch)
        d = compare_one(baseline, br, args.branch.name)
        print_compare(d)
        reports.append(d)
    elif args.all:
        for sub in sorted(args.all.iterdir()):
            if not sub.is_dir() or sub.name == args.baseline.name:
                continue
            br = load_results(sub)
            if not br["tenants"]:
                continue
            d = compare_one(baseline, br, sub.name)
            print_compare(d)
            reports.append(d)

    # Sort by verdict priority
    verdict_order = {"MERGE": 0, "MERGE-WITH-CAVEAT": 1, "REVIEW": 2, "NEUTRAL": 3, "REJECT": 4}
    reports.sort(key=lambda r: (verdict_order.get(r["verdict"], 99), -r["deltas"].get("judge", 0)))

    print("\n\n=== SUMMARY ===")
    print(f"{'Branch':<40} {'Verdict':<18} {'ΔG':>6} {'ΔC%':>6} {'ΔJ':>6}")
    print("-" * 85)
    for r in reports:
        d = r["deltas"]
        dj = f"{d['judge']:+.2f}" if "judge" in d else "—"
        print(f"{r['branch']:<40} {r['verdict']:<18} {d['groundedness']:>+6.3f} {d['cost_pct']:>+6.1f} {dj:>6}")

    if args.output:
        args.output.write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nFull report saved to {args.output}")
