"""Produce MERGE_DECISIONS.md from benchmark results + classification.

Merges three inputs:
  1. benchmark/branch_classification.json (what each branch changes)
  2. benchmark/results/main/ (baseline results)
  3. benchmark/results/<branch>/ (per-branch results)
  4. Judge scores inside each result file (populated by judge_all.py)

Output: MERGE_DECISIONS.md ranked by verdict + delta

Usage:
  python benchmark/decide.py --classification benchmark/branch_classification.json \
       --baseline benchmark/results/main \
       --results-root benchmark/results \
       --output MERGE_DECISIONS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_dir(d: Path) -> dict:
    tenants = {}
    if not d.exists():
        return {"tenants": {}}
    for f in d.glob("bench_*.json"):
        try:
            tenants[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return {"tenants": tenants}


def aggregate(results: dict) -> dict:
    tenants = results["tenants"]
    if not tenants:
        return {}
    all_g, all_j, all_att = [], [], []
    total_cost = 0.0
    total_personas = 0
    total_ok = 0
    for t in tenants.values():
        for c in t.get("clusters", []):
            if not c.get("failed"):
                all_g.append(c.get("groundedness", 0))
                all_att.append(c.get("attempts", 0))
                js = c.get("judge_score")
                if js and isinstance(js, dict) and "overall" in js:
                    all_j.append(float(js["overall"]))
                total_ok += 1
            total_personas += 1
            total_cost += c.get("cost_usd", 0)
    return {
        "n_personas": total_personas,
        "n_ok": total_ok,
        "success_rate": total_ok / total_personas if total_personas else 0,
        "mean_groundedness": sum(all_g) / len(all_g) if all_g else 0,
        "mean_judge": sum(all_j) / len(all_j) if all_j else None,
        "mean_attempts": sum(all_att) / len(all_att) if all_att else 0,
        "total_cost": total_cost,
    }


def verdict_for(deltas: dict, base: dict, branch: dict) -> str:
    dj = deltas.get("judge")
    dg = deltas["groundedness"]
    dc = deltas["cost_pct"]
    ds = deltas["success_rate"]

    # Hard fail if branch broke
    if branch.get("n_ok", 0) == 0:
        return "REJECT-BROKEN"
    # Regression thresholds
    if (dj is not None and dj < -0.15) or dg < -0.05 or ds < -0.10 or dc > 80:
        return "REJECT"
    if ds < -0.05:
        return "REJECT-RELIABILITY"
    # Merge conditions
    if dj is not None and dj >= 0.15 and dc < 30:
        return "MERGE"
    if dg >= 0.05 and ds >= 0 and dc < 30:
        return "MERGE"
    if dj is not None and dj >= 0.05 and ds >= 0 and dc < 15:
        return "MERGE"
    # Neutral window
    if (abs(dj or 0) <= 0.05 and abs(dg) <= 0.02 and abs(dc) <= 10
            and abs(ds) <= 0.05):
        return "NEUTRAL"
    # Everything else for human eye
    return "REVIEW"


def compute(baseline_dir: Path, branch_dir: Path, name: str) -> dict:
    b = aggregate(load_dir(baseline_dir))
    br = aggregate(load_dir(branch_dir))
    if not br:
        return {"branch": name, "verdict": "NO-RESULTS", "deltas": {}, "branch_metrics": {}, "baseline": b}

    def d(a, bv): return bv - a
    def pct(a, bv): return ((bv - a) / a * 100) if a else 0.0

    deltas = {
        "groundedness": d(b.get("mean_groundedness", 0), br.get("mean_groundedness", 0)),
        "cost_pct": pct(b.get("total_cost", 0), br.get("total_cost", 0)),
        "success_rate": d(b.get("success_rate", 0), br.get("success_rate", 0)),
        "attempts": d(b.get("mean_attempts", 0), br.get("mean_attempts", 0)),
    }
    if b.get("mean_judge") is not None and br.get("mean_judge") is not None:
        deltas["judge"] = d(b["mean_judge"], br["mean_judge"])

    verdict = verdict_for(deltas, b, br)
    return {
        "branch": name,
        "verdict": verdict,
        "deltas": deltas,
        "baseline": b,
        "branch_metrics": br,
    }


def render_md(reports: list[dict], classification: dict) -> str:
    class_by_branch = {d["branch"]: d for d in classification}
    # Sort by verdict priority then by judge delta desc
    priority = {
        "MERGE": 0, "REVIEW": 1, "NEUTRAL": 2,
        "REJECT-RELIABILITY": 3, "REJECT": 4, "REJECT-BROKEN": 5, "NO-RESULTS": 6,
    }
    reports.sort(key=lambda r: (
        priority.get(r["verdict"], 99),
        -(r["deltas"].get("judge") or 0),
        -r["deltas"].get("groundedness", 0),
    ))

    # Count verdicts
    counts: dict[str, int] = {}
    for r in reports:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    md = []
    md.append("# Merge Decisions — Experiment Branch Evaluation")
    md.append("")
    md.append(f"Evaluated {len(reports)} runtime-changing branches against main baseline.")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Verdict | Count | Meaning |")
    md.append("|---------|-------|---------|")
    md.append(f"| MERGE | {counts.get('MERGE', 0)} | Clear improvement, safe to merge |")
    md.append(f"| REVIEW | {counts.get('REVIEW', 0)} | Mixed signals, human read-through needed |")
    md.append(f"| NEUTRAL | {counts.get('NEUTRAL', 0)} | No meaningful change; safe to merge if orthogonal |")
    md.append(f"| REJECT | {counts.get('REJECT', 0)} | Quality regression |")
    md.append(f"| REJECT-RELIABILITY | {counts.get('REJECT-RELIABILITY', 0)} | Reduced success rate |")
    md.append(f"| REJECT-BROKEN | {counts.get('REJECT-BROKEN', 0)} | Zero successful personas — branch is broken |")
    md.append(f"| NO-RESULTS | {counts.get('NO-RESULTS', 0)} | Benchmark didn't run — infra issue |")
    md.append("")
    md.append("## Legend")
    md.append("- **ΔG**: change in mean groundedness vs main (higher is better)")
    md.append("- **ΔC%**: % change in total cost vs main (negative is better)")
    md.append("- **ΔS**: change in success rate vs main (higher is better)")
    md.append("- **ΔJ**: change in mean judge score vs main (higher is better)")
    md.append("")
    md.append("## Ranked verdicts")
    md.append("")
    md.append("| Branch | Verdict | ΔJ | ΔG | ΔC% | ΔS | Notes |")
    md.append("|--------|---------|----|----|-----|----|-------|")
    for r in reports:
        d = r["deltas"]
        dj = f"{d['judge']:+.2f}" if d.get("judge") is not None else "—"
        dg = f"{d.get('groundedness', 0):+.3f}"
        dc = f"{d.get('cost_pct', 0):+.1f}%" if d.get("cost_pct") is not None else "—"
        ds = f"{d.get('success_rate', 0):+.2f}"
        c = class_by_branch.get(r["branch"], {})
        rt = len(c.get("runtime_files_changed", []))
        notes = f"rt-files={rt}"
        md.append(f"| `{r['branch']}` | **{r['verdict']}** | {dj} | {dg} | {dc} | {ds} | {notes} |")

    md.append("")
    md.append("## Raw metrics per branch")
    md.append("")
    for r in reports:
        br = r.get("branch_metrics", {})
        b = r.get("baseline", {})
        md.append(f"### `{r['branch']}` — {r['verdict']}")
        md.append("")
        md.append(f"- Personas: {br.get('n_ok', 0)}/{br.get('n_personas', 0)} (main: {b.get('n_ok', 0)}/{b.get('n_personas', 0)})")
        md.append(f"- Groundedness: {br.get('mean_groundedness', 0):.3f} (main: {b.get('mean_groundedness', 0):.3f})")
        if br.get("mean_judge") is not None:
            md.append(f"- Judge score: {br.get('mean_judge', 0):.2f} (main: {b.get('mean_judge') or 0:.2f})")
        md.append(f"- Mean attempts: {br.get('mean_attempts', 0):.2f} (main: {b.get('mean_attempts', 0):.2f})")
        md.append(f"- Total cost: ${br.get('total_cost', 0):.4f} (main: ${b.get('total_cost', 0):.4f})")
        md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classification = json.loads(args.classification.read_text())
    runtime = [d for d in classification if d["type"] == "runtime-changing"]

    reports = []
    for d in runtime:
        branch_dir = args.results_root / d["branch"]
        reports.append(compute(args.baseline, branch_dir, d["branch"]))

    md = render_md(reports, classification)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Processed {len(reports)} branches")
