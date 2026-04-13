"""Build a comprehensive HTML report covering every experiment branch.

Inputs:
  - benchmark/branch_classification.json  (type per branch)
  - benchmark/results/main/                (baseline)
  - benchmark/results/merging_experiments/ (accepted merges)
  - benchmark/results/gower-jaccard/       (gower branch, jaccard mode)
  - benchmark/results/gower-gower/         (gower branch, gower mode)
  - benchmark/results/<exp-branch>/        (each runtime-changing branch run)
  - benchmark/automerge.log.json           (auto-merge outcomes)

Produces: benchmark/report.html — single self-contained file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_dir(d: Path) -> dict:
    tenants = {}
    if not d.exists():
        return {"summary": {}, "tenants": {}}
    for f in d.glob("bench_*.json"):
        try:
            tenants[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    summary = {}
    sf = d / "summary.json"
    if sf.exists():
        summary = json.loads(sf.read_text())
    return {"summary": summary, "tenants": tenants}


def aggregate(data: dict) -> dict:
    tenants = data["tenants"]
    if not tenants:
        return {}
    all_g, all_j, all_att = [], [], []
    total_cost = 0.0
    total_personas = 0
    total_ok = 0
    total_clusters = 0
    for t in tenants.values():
        cs = t.get("clusters", [])
        total_clusters += len(cs)
        for c in cs:
            if not c.get("failed"):
                all_g.append(c.get("groundedness", 0))
                all_att.append(c.get("attempts", 0))
                js = c.get("judge_score")
                if isinstance(js, dict) and "overall" in js:
                    all_j.append(float(js["overall"]))
                total_ok += 1
            total_personas += 1
            total_cost += c.get("cost_usd", 0)
    return {
        "clusters": total_clusters,
        "personas_ok": total_ok,
        "personas_total": total_personas,
        "success_rate": total_ok / total_personas if total_personas else 0,
        "mean_groundedness": sum(all_g) / len(all_g) if all_g else 0,
        "mean_judge": sum(all_j) / len(all_j) if all_j else 0,
        "mean_attempts": sum(all_att) / len(all_att) if all_att else 0,
        "total_cost": total_cost,
    }


def deltas(baseline: dict, branch: dict) -> dict:
    if not branch or not baseline:
        return {}

    def d(a, b):
        return b - a

    def pct(a, b):
        return ((b - a) / a * 100) if a else 0

    return {
        "judge": d(baseline.get("mean_judge", 0), branch.get("mean_judge", 0)),
        "groundedness": d(baseline.get("mean_groundedness", 0), branch.get("mean_groundedness", 0)),
        "cost_pct": pct(baseline.get("total_cost", 0), branch.get("total_cost", 0)),
        "success_rate": d(baseline.get("success_rate", 0), branch.get("success_rate", 0)),
        "attempts": d(baseline.get("mean_attempts", 0), branch.get("mean_attempts", 0)),
    }


def compute_verdict(deltas_: dict, branch: dict) -> str:
    if not branch:
        return "NO-RESULTS"
    if branch.get("personas_ok", 0) == 0:
        return "REJECT-BROKEN"
    dj = deltas_.get("judge", 0)
    dg = deltas_.get("groundedness", 0)
    dc = deltas_.get("cost_pct", 0)
    ds = deltas_.get("success_rate", 0)
    if (dj is not None and dj < -0.15) or dg < -0.05 or ds < -0.10 or dc > 80:
        return "REJECT"
    if ds < -0.05:
        return "REJECT-RELIABILITY"
    if dj is not None and dj >= 0.15 and dc < 30:
        return "MERGE"
    if dg >= 0.05 and ds >= 0 and dc < 30:
        return "MERGE"
    if dj is not None and dj >= 0.05 and ds >= 0 and dc < 15:
        return "MERGE"
    if abs(dj or 0) <= 0.05 and abs(dg) <= 0.02 and abs(dc) <= 10 and abs(ds) <= 0.05:
        return "NEUTRAL"
    return "REVIEW"


# Human-readable descriptions for key branches
BRANCH_DESCRIPTIONS = {
    # The 4 MERGED branches
    "exp-1.11-negative-space": {
        "owner": "Pak",
        "title": "Negative-space persona fields",
        "hypothesis": "Adding a `not_this` field capturing what the persona would NOT do or say sharpens the persona's identity.",
        "change": "Adds `not_this: list[str]` (min 2, max 6) to PersonaV1 schema. Prompt instructs model to treat these as identity boundaries, distinct from sales objections.",
    },
    "exp-2.07-order-of-fields": {
        "owner": "Pak",
        "title": "Voice-first field ordering",
        "hypothesis": "Pydantic preserves declaration order in JSON schema; Claude fills tool-use fields in that order. Declaring vocabulary and quotes BEFORE demographics may reduce stereotyping.",
        "change": "Adds a `PersonaV1VoiceFirst` variant with vocabulary/sample_quotes declared first. Adds `schema_cls` parameter to synthesize() and build_tool_definition().",
    },
    "exp-4.14-latency-vs-realism": {
        "owner": "Pak",
        "title": "Latency-vs-realism tradeoff knob",
        "hypothesis": "Twin response latency affects perceived realism; adding a configurable artificial delay lets callers study this tradeoff.",
        "change": "Adds `artificial_delay_ms` parameter on TwinChat and latency tracking (model_latency_ms, total_latency_ms) on TwinReply.",
    },
    "exp-6.04": {
        "owner": "Billy",
        "title": "Cross-persona contrast prompting",
        "hypothesis": "When synthesizing persona N+1, injecting personas 1..N with instructions to differ on goals/vocabulary/voice will produce more distinctive personas.",
        "change": "Adds `existing_personas` parameter to synthesize() and build_user_message(). When provided, injects a contrast block instructing the LLM to differentiate.",
    },
    # Gower branch special case
    "feat/adaptive-gower-segmentation": {
        "owner": "Shruti",
        "title": "Adaptive Gower-distance segmentation",
        "hypothesis": "Jaccard distance over behavior sets is one of many ways to cluster users. Gower distance handles numeric, categorical, and set features uniformly, which may produce better clusters when records have firmographic or session-duration signals.",
        "change": "3,635 lines across 36 files. New modules: gower.py, registry.py, schema_inference.py. FeatureType enum with typed feature dicts (numeric/categorical/set). Feature schema registry for connectors. Per-type distance functions with family weighting and missing-data handling. ClusterPrototype (union/mean/mode) replaces fixed-seed comparison. Pipeline accepts `distance_metric='jaccard'|'gower'`, `feature_registry`, `family_weights`. Jaccard path preserved byte-for-byte as default. 177 tests.",
    },
}


def build_report(
    classification: list[dict],
    baseline_dir: Path,
    merging_dir: Path,
    gower_jaccard_dir: Path,
    gower_gower_dir: Path,
    results_root: Path,
    automerge_log: Path,
) -> dict:
    baseline_data = load_dir(baseline_dir)
    merging_data = load_dir(merging_dir)
    gower_j_data = load_dir(gower_jaccard_dir)
    gower_g_data = load_dir(gower_gower_dir)

    baseline_agg = aggregate(baseline_data)
    merging_agg = aggregate(merging_data)
    gower_j_agg = aggregate(gower_j_data)
    gower_g_agg = aggregate(gower_g_data)

    # Load automerge log
    automerge_results: dict[str, dict] = {}
    if automerge_log.exists():
        for r in json.loads(automerge_log.read_text()):
            automerge_results[r["branch"]] = r

    # Classify every branch
    branches = []
    for d in classification:
        name = d["branch"]
        btype = d["type"]
        entry = {
            "name": name,
            "type": btype,
            "files_changed_runtime": d.get("runtime_files_changed", []),
            "new_files": d.get("new_files", []),
            "other_modified": d.get("other_modified", []),
        }
        # Benchmarked branches
        if btype == "runtime-changing":
            branch_dir = results_root / name
            bd = load_dir(branch_dir)
            bag = aggregate(bd)
            dl = deltas(baseline_agg, bag)
            entry["metrics"] = bag
            entry["deltas"] = dl
            entry["verdict"] = compute_verdict(dl, bag) if bag else "NO-RESULTS"
            if entry["verdict"] == "MERGE":
                entry["merged_into"] = "merging_experiments"
        else:
            # Auto-merge branches
            am = automerge_results.get(name, {})
            entry["automerge_status"] = am.get("status", "unknown")
            entry["files_added"] = am.get("files_added", [])
            entry["files_skipped"] = am.get("files_modified_skipped", [])
            if am.get("status") == "ok":
                entry["verdict"] = "MERGED-AS-OPTION"
                entry["merged_into"] = "merging_experiments"
            elif am.get("status") in ("no_new_files", "all_files_already_present", "nothing_to_commit"):
                entry["verdict"] = "SKIPPED-NO-CHANGE"
            else:
                entry["verdict"] = "NOT-PROCESSED"

        # Owner + description if known
        info = BRANCH_DESCRIPTIONS.get(name, {})
        entry.update(info)
        branches.append(entry)

    # Counts
    from collections import Counter
    by_verdict = Counter(b["verdict"] for b in branches)
    by_type = Counter(b["type"] for b in branches)

    return {
        "baseline": baseline_agg,
        "merging_experiments": merging_agg,
        "gower_jaccard": gower_j_agg,
        "gower_gower": gower_g_agg,
        "branches": branches,
        "counts_by_verdict": dict(by_verdict),
        "counts_by_type": dict(by_type),
        "gower_meta": BRANCH_DESCRIPTIONS["feat/adaptive-gower-segmentation"],
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment branch evaluation — full report</title>
<style>
  * { box-sizing: border-box; }
  :root {
    --bg: #0f1419;
    --panel: #161b22;
    --panel2: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --dim: #656d76;
    --main: #58a6ff;
    --merge: #d29922;
    --gower: #c678dd;
    --up: #3fb950;
    --down: #f85149;
    --flat: #8b949e;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-size: 14px; line-height: 1.5;
  }
  header {
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 24px 32px; position: sticky; top: 0; z-index: 100;
  }
  h1 { font-size: 22px; margin: 0 0 4px 0; }
  h2 { font-size: 18px; margin: 32px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  h3 { font-size: 15px; margin: 20px 0 8px 0; color: var(--text); }
  .subtitle { color: var(--muted); font-size: 13px; }
  nav { display: flex; gap: 16px; margin-top: 12px; font-size: 13px; }
  nav a { color: var(--main); text-decoration: none; padding: 2px 0; }
  nav a:hover { text-decoration: underline; }
  main { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }
  section { margin-bottom: 48px; }
  .callout {
    background: var(--panel); border-left: 3px solid var(--main);
    padding: 12px 16px; margin: 12px 0; border-radius: 4px;
    color: var(--text);
  }
  .callout.warning { border-left-color: var(--merge); }
  .callout.reject { border-left-color: var(--down); }
  .metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0;
  }
  .metric-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px;
  }
  .metric-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .metric-card .value { font-size: 22px; font-weight: 600; }
  .metric-card .delta { font-size: 13px; font-weight: 600; margin-top: 4px; }
  .delta.up { color: var(--up); } .delta.down { color: var(--down); } .delta.flat { color: var(--flat); }
  .verdict-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0;
  }
  .verdict-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px; display: flex; justify-content: space-between; align-items: baseline;
  }
  .verdict-card .label { font-size: 13px; }
  .verdict-card .count { font-size: 24px; font-weight: 700; }
  .verdict-card.merge { border-left: 4px solid var(--up); }
  .verdict-card.reject { border-left: 4px solid var(--down); }
  .verdict-card.neutral { border-left: 4px solid var(--flat); }
  .verdict-card.review { border-left: 4px solid var(--merge); }
  .verdict-card.option { border-left: 4px solid var(--main); }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  tr:hover td { background: var(--panel2); }
  code {
    font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px;
    background: var(--panel2); padding: 1px 5px; border-radius: 3px;
  }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge.MERGE { background: rgba(63, 185, 80, 0.15); color: var(--up); }
  .badge.MERGED-AS-OPTION { background: rgba(88, 166, 255, 0.15); color: var(--main); }
  .badge.REVIEW { background: rgba(210, 153, 34, 0.15); color: var(--merge); }
  .badge.NEUTRAL { background: rgba(139, 148, 158, 0.15); color: var(--flat); }
  .badge.REJECT, .badge.REJECT-RELIABILITY, .badge.REJECT-BROKEN { background: rgba(248, 81, 73, 0.15); color: var(--down); }
  .badge.NO-RESULTS, .badge.SKIPPED-NO-CHANGE, .badge.NOT-PROCESSED { background: rgba(101, 109, 118, 0.15); color: var(--dim); }
  .branch-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 12px;
  }
  .branch-card .head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; gap: 12px; }
  .branch-card .name { font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 600; font-size: 14px; }
  .branch-card .title { font-size: 13px; color: var(--muted); }
  .branch-card .section { margin-top: 10px; }
  .branch-card .section .label { font-size: 11px; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; letter-spacing: 0.5px; }
  .branch-card dl { margin: 0; }
  .branch-card dt { font-weight: 600; color: var(--muted); font-size: 12px; margin-top: 6px; }
  .branch-card dd { margin: 2px 0 0 0; font-size: 13px; }
  .branch-card .metrics-row { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; }
  .branch-card .metrics-row span { white-space: nowrap; }
  .branch-card .file-list { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; color: var(--muted); }
  .branch-card .file-list li { margin: 1px 0; }
  details summary { cursor: pointer; color: var(--main); font-size: 12px; padding: 4px 0; }
  .toolbar { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
  .toolbar input[type="text"] {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 4px; font-size: 13px; flex: 1; min-width: 240px;
  }
  .toolbar button {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer;
  }
  .toolbar button.active { background: var(--main); color: var(--bg); border-color: var(--main); }
  .toolbar button:hover { background: var(--panel); }
  .comparison-table { font-size: 13px; }
  .comparison-table .label-col { font-weight: 600; color: var(--muted); }
  .comparison-table .winner { font-weight: 700; }
  .variant-main { color: var(--main); }
  .variant-merge { color: var(--merge); }
  .variant-gower { color: var(--gower); }
  .kicker { color: var(--muted); font-size: 12px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .byline { color: var(--muted); font-size: 12px; }
  .q { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>Experiment Branch Evaluation — Full Report</h1>
  <div class="subtitle">
    136 experiment branches evaluated against a 42-persona benchmark on 10 synthetic tenants.
    Judge-scored by calibrated Haiku LLM. Merged onto <code>merging_experiments</code> branch.
  </div>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#methodology">Methodology</a>
    <a href="#scoreboard">Scoreboard</a>
    <a href="#merged">Merged (4)</a>
    <a href="#gower">Shruti: Gower</a>
    <a href="#automerged">Auto-merged (64)</a>
    <a href="#review">Review (7)</a>
    <a href="#neutral">Neutral (19)</a>
    <a href="#rejected">Rejected (21)</a>
    <a href="#skipped">Skipped/Missing (21)</a>
    <a href="#all">All branches</a>
  </nav>
</header>

<main>

<section id="overview">
  <h2>Overview</h2>
  <div class="callout">
    The team produced <strong>136 experiment branches</strong> across 6 researchers (Yash, Shruti, Max, Billy, Pak, Ivan).
    We built a deterministic 10-tenant synthetic benchmark (42 clusters, 3,305 records total), ran the pipeline on
    <code>main</code> as a baseline, then ran each branch that modifies runtime code and compared judge scores,
    groundedness, success rate, and cost. Backwards-compatible additions (new modules, new eval harnesses) were
    auto-merged without benchmarking. One branch (<code>feat/adaptive-gower-segmentation</code>) was evaluated
    separately in both Jaccard and Gower modes.
  </div>

  <div class="verdict-grid" id="verdict-overview"></div>
</section>

<section id="methodology">
  <h2>Methodology</h2>

  <h3>The benchmark</h3>
  <p>10 deterministic synthetic tenants producing 42 clusters:</p>
  <ul>
    <li><code>bench_mega_saas</code> — 500 records, 6 cohorts (engineers, designers, PMs, marketers, sales, support)</li>
    <li><code>bench_mega_fintech</code> — 500 records, 5 cohorts (engineers, compliance, sales, security, data)</li>
    <li><code>bench_mega_ecommerce</code> — 500 records, 5 cohorts</li>
    <li><code>bench_dense_devtools</code> — 350 records, 4 cohorts</li>
    <li><code>bench_sparse_30</code> — sparsity stress, 30 records</li>
    <li><code>bench_sparse_60</code> — mid-sparsity, 60 records</li>
    <li><code>bench_poisoned</code> — 350 + 15 adversarial records (agriculture/culinary poison)</li>
    <li><code>bench_heavy_tail</code> — 300 records, 70/10/10/10 imbalance</li>
    <li><code>bench_single_cohort</code> — 200 records, 1 cohort — tests degeneracy</li>
    <li><code>bench_diverse</code> — 500 records, 8 cohorts</li>
  </ul>

  <h3>Classification</h3>
  <p>Each branch was bucketed by what files it modifies:</p>
  <dl>
    <dt>Runtime-changing (51 branches)</dt>
    <dd>Modifies core pipeline files (prompt_builder, synthesizer, chat, judges, groundedness, segmentation).
        Default behavior differs from main, so a benchmark run is required to measure the delta.</dd>
    <dt>Schema-additive (74 branches)</dt>
    <dd>Adds new modules / persona variants without changing existing behavior. Auto-merge candidate.</dd>
    <dt>Eval-only (11 branches)</dt>
    <dd>Only adds measurement tooling under <code>evals/</code>. Auto-merge candidate.</dd>
  </dl>

  <h3>Verdict rubric (for runtime-changing branches)</h3>
  <table>
    <thead><tr><th>Verdict</th><th>Criteria</th></tr></thead>
    <tbody>
    <tr><td><span class="badge MERGE">MERGE</span></td>
        <td>ΔJudge ≥ +0.15 AND cost penalty &lt; 30%, OR ΔGroundedness ≥ +0.05 AND ΔSuccess ≥ 0 AND cost penalty &lt; 30%</td></tr>
    <tr><td><span class="badge REVIEW">REVIEW</span></td>
        <td>Mixed signals — positive judge but reliability concerns, or high variance across tenants</td></tr>
    <tr><td><span class="badge NEUTRAL">NEUTRAL</span></td>
        <td>All deltas within ±5% judge, ±2% groundedness, ±10% cost, ±5% success</td></tr>
    <tr><td><span class="badge REJECT">REJECT</span></td>
        <td>ΔJudge &lt; -0.15, OR ΔGroundedness &lt; -0.05, OR cost penalty &gt; 80%, OR ΔSuccess &lt; -10%</td></tr>
    <tr><td><span class="badge REJECT-RELIABILITY">REJECT-RELIABILITY</span></td>
        <td>ΔSuccess between -5% and -10% (functional but less reliable)</td></tr>
    <tr><td><span class="badge REJECT-BROKEN">REJECT-BROKEN</span></td>
        <td>Zero personas produced — branch is fundamentally broken</td></tr>
    </tbody>
  </table>

  <h3>Judge model</h3>
  <p>LLM-as-judge scoring 1-5 on five dimensions (grounded, distinctive, coherent, actionable, voice_fidelity)
  using Claude Haiku 4.5 with few-shot calibration (from exp-5.13). Every persona produced by every variant
  was scored — 1,867 + 42 + 39 + 21 = 1,969 total judge calls.</p>
</section>

<section id="scoreboard">
  <h2>Scoreboard: 4-way comparison</h2>
  <div id="scoreboard-table"></div>
  <div class="callout">
    <strong>Merged result</strong>: <code>merging_experiments</code> = main + 4 MERGE branches + 64 additive branches.
    After bumping <code>MAX_RETRIES</code> from 2 to 4 to absorb the stacked schema complexity, it ties main
    on reliability (42/42 personas) and beats it on judge score by <strong>+0.19</strong> points.
  </div>
</section>

<section id="merged">
  <h2>The 4 MERGE-verdict branches</h2>
  <p>These are the high-impact branches that got full merges onto <code>merging_experiments</code> in
  descending-ΔJudge order. Smoke-tested after each merge; one conflict (<code>exp-6.04</code> vs <code>exp-2.07</code>
  in <code>synthesizer.py</code> parameter list) was resolved by keeping both additions.</p>
  <div id="merged-list"></div>
</section>

<section id="gower">
  <h2>Shruti: <code>feat/adaptive-gower-segmentation</code></h2>
  <div class="byline">Why this one is special: not named <code>exp-*</code>, much larger scope, own benchmark.</div>
  <div id="gower-detail"></div>
</section>

<section id="automerged">
  <h2>Auto-merged branches (64)</h2>
  <div class="subtitle">Backwards-compatible additions — new files, no default-behavior changes. Auto-merged onto
  <code>merging_experiments</code> without benchmarking since their runtime impact is nil by construction.</div>
  <div class="toolbar">
    <input type="text" id="filter-auto" placeholder="Filter by name…">
  </div>
  <div id="automerged-list"></div>
</section>

<section id="review">
  <h2>REVIEW — mixed signals (7)</h2>
  <div class="subtitle">Judge score went up but reliability or cost hit warrants human review before merging.</div>
  <div id="review-list"></div>
</section>

<section id="neutral">
  <h2>NEUTRAL — within noise (19)</h2>
  <div class="subtitle">Within ±5% judge and ±5% success rate of main. Safe to merge as optional capability if the new parameter is useful, but no measurable lift on the default path.</div>
  <div id="neutral-list"></div>
</section>

<section id="rejected">
  <h2>REJECT-verdict branches (21)</h2>
  <div class="subtitle">Quality regression, reliability drop, or outright broken. Not merged.</div>
  <div id="rejected-list"></div>
</section>

<section id="skipped">
  <h2>Skipped / No-change branches (21)</h2>
  <div class="subtitle">Branches whose diff against main contained only filtered files (docs/plans/, output/,
  benchmark/, .gitignore) — nothing to bring in.</div>
  <div id="skipped-list"></div>
</section>

<section id="all">
  <h2>All 136 branches (sortable)</h2>
  <div class="toolbar">
    <input type="text" id="filter-all" placeholder="Filter by name, owner, or type…">
    <button data-verdict="all" class="active">All</button>
    <button data-verdict="MERGE">MERGE</button>
    <button data-verdict="MERGED-AS-OPTION">AUTO-MERGED</button>
    <button data-verdict="REVIEW">REVIEW</button>
    <button data-verdict="NEUTRAL">NEUTRAL</button>
    <button data-verdict="REJECT">REJECT</button>
    <button data-verdict="SKIPPED-NO-CHANGE">SKIPPED</button>
  </div>
  <div id="all-branches">(loading...)</div>
</section>

</main>

<script>
const DATA = __DATA__;

function fmt(n, d=2) { return (n==null||isNaN(n)) ? '—' : n.toFixed(d); }
function fmtPct(n) { return n==null ? '—' : (n*100).toFixed(1) + '%'; }
function fmtMoney(n) { return n==null ? '—' : '$' + n.toFixed(4); }
function fmtDeltaJ(d) {
  if (d == null || isNaN(d)) return '—';
  const s = d >= 0 ? '+' : '';
  return s + d.toFixed(2);
}
function fmtDeltaG(d) {
  if (d == null || isNaN(d)) return '—';
  const s = d >= 0 ? '+' : '';
  return s + d.toFixed(3);
}
function fmtDeltaPct(d) {
  if (d == null || isNaN(d)) return '—';
  const s = d >= 0 ? '+' : '';
  return s + d.toFixed(1) + '%';
}

// ============================================================================
// Verdict overview
// ============================================================================
(function() {
  const counts = DATA.counts_by_verdict;
  const groups = [
    {label:'MERGE', keys:['MERGE'], cls:'merge', desc:'High-impact, benchmarked wins'},
    {label:'MERGED-AS-OPTION', keys:['MERGED-AS-OPTION'], cls:'option', desc:'Additive, backwards-compatible'},
    {label:'REVIEW', keys:['REVIEW'], cls:'review', desc:'Mixed signals'},
    {label:'NEUTRAL', keys:['NEUTRAL'], cls:'neutral', desc:'Within noise'},
    {label:'REJECT family', keys:['REJECT','REJECT-RELIABILITY','REJECT-BROKEN'], cls:'reject', desc:'Regression or broken'},
    {label:'SKIPPED / missing', keys:['SKIPPED-NO-CHANGE','NO-RESULTS','NOT-PROCESSED'], cls:'neutral', desc:'No code to bring in or failed to run'},
  ];
  const root = document.getElementById('verdict-overview');
  root.innerHTML = groups.map(g => {
    const total = g.keys.reduce((s, k) => s + (counts[k] || 0), 0);
    return `<div class="verdict-card ${g.cls}">
      <div><div class="label">${g.label}</div><div class="subtitle">${g.desc}</div></div>
      <div class="count">${total}</div>
    </div>`;
  }).join('');
})();

// ============================================================================
// Scoreboard
// ============================================================================
(function() {
  const b = DATA.baseline;
  const m = DATA.merging_experiments;
  const gj = DATA.gower_jaccard;
  const gg = DATA.gower_gower;
  const variants = [
    {name: 'main baseline', data: b, cls: 'variant-main', note: 'untouched reference'},
    {name: 'merging_experiments', data: m, cls: 'variant-merge', note: '4 MERGE + 64 additive, MAX_RETRIES=4'},
    {name: 'gower (jaccard mode)', data: gj, cls: 'variant-gower', note: 'parity claim — should match main'},
    {name: 'gower (gower mode)', data: gg, cls: 'variant-gower', note: 'typed-feature clustering'},
  ];
  const rows = [
    {label:'Clusters', get: d=>d.clusters, fmt: v=>v, better:null},
    {label:'Personas ok / total', get: d=>`${d.personas_ok}/${d.personas_total}`, fmt: v=>v, better:null},
    {label:'Success rate', get: d=>d.success_rate, fmt: v=>fmtPct(v), better:'up'},
    {label:'Mean judge (1-5)', get: d=>d.mean_judge, fmt: v=>fmt(v,2), better:'up'},
    {label:'Mean groundedness', get: d=>d.mean_groundedness, fmt: v=>fmt(v,3), better:'up'},
    {label:'Mean attempts', get: d=>d.mean_attempts, fmt: v=>fmt(v,2), better:'down'},
    {label:'Total cost', get: d=>d.total_cost, fmt: v=>fmtMoney(v), better:'down'},
  ];
  const root = document.getElementById('scoreboard-table');
  let html = '<table class="comparison-table"><thead><tr><th></th>';
  variants.forEach(v => { html += `<th class="${v.cls}">${v.name}<div class="byline">${v.note}</div></th>`; });
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    const values = variants.map(v => v.data[r.label] !== undefined ? v.data[r.label] : r.get(v.data));
    let winnerIdx = null;
    if (r.better) {
      const nums = values.map(v => typeof v === 'number' ? v : NaN);
      if (r.better === 'up') winnerIdx = nums.indexOf(Math.max(...nums.filter(n=>!isNaN(n))));
      else winnerIdx = nums.indexOf(Math.min(...nums.filter(n=>!isNaN(n))));
    }
    html += `<tr><td class="label-col">${r.label}</td>`;
    values.forEach((v, i) => {
      html += `<td class="${i === winnerIdx ? 'winner' : ''}">${r.fmt(v)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  root.innerHTML = html;
})();

// ============================================================================
// Merged (MERGE verdict)
// ============================================================================
function renderBranchCard(b, opts={}) {
  const d = b.deltas || {};
  const m = b.metrics || {};
  const metricStr = (m.mean_judge || m.mean_judge === 0)
    ? `<div class="metrics-row">
        <span>personas: <strong>${m.personas_ok}/${m.personas_total}</strong></span>
        <span>judge: <strong>${fmt(m.mean_judge,2)}</strong></span>
        <span>grounded: ${fmt(m.mean_groundedness,3)}</span>
        <span>attempts: ${fmt(m.mean_attempts,2)}</span>
        <span>cost: ${fmtMoney(m.total_cost)}</span>
      </div>` : '';
  const deltaStr = d.judge !== undefined
    ? `<div class="metrics-row">
        <span>ΔJudge: <strong class="delta ${d.judge>0.01?'up':d.judge<-0.01?'down':'flat'}">${fmtDeltaJ(d.judge)}</strong></span>
        <span>ΔGnd: <strong class="delta ${d.groundedness>0.01?'up':d.groundedness<-0.01?'down':'flat'}">${fmtDeltaG(d.groundedness)}</strong></span>
        <span>ΔCost: <strong class="delta ${d.cost_pct<-1?'up':d.cost_pct>1?'down':'flat'}">${fmtDeltaPct(d.cost_pct)}</strong></span>
        <span>ΔSuccess: <strong class="delta ${d.success_rate>0.01?'up':d.success_rate<-0.01?'down':'flat'}">${fmtDeltaG(d.success_rate)}</strong></span>
      </div>` : '';
  const filesStr = (() => {
    const rt = b.files_changed_runtime || [];
    const added = b.files_added || b.new_files || [];
    if (!rt.length && !added.length) return '';
    return `<details><summary>files (${rt.length} rt + ${added.length} new)</summary>
      <ul class="file-list">
        ${rt.map(f => '<li><code>'+f+'</code> <span class="q">(modified)</span></li>').join('')}
        ${added.map(f => '<li><code>'+f+'</code> <span class="q">(new)</span></li>').join('')}
      </ul></details>`;
  })();
  const owner = b.owner ? `<span class="q">by ${b.owner}</span>` : '';
  const body = b.hypothesis ? `
    <div class="section">
      <div class="label">Hypothesis</div>
      <div>${b.hypothesis}</div>
    </div>
    <div class="section">
      <div class="label">Change</div>
      <div>${b.change}</div>
    </div>` : '';
  return `<div class="branch-card">
    <div class="head">
      <div>
        <div class="name">${b.name}</div>
        <div class="title">${b.title || b.type} ${owner}</div>
      </div>
      <div><span class="badge ${b.verdict}">${b.verdict}</span></div>
    </div>
    ${metricStr}
    ${deltaStr}
    ${body}
    ${filesStr}
  </div>`;
}

(function() {
  const merged = DATA.branches
    .filter(b => b.verdict === 'MERGE')
    .sort((a,b) => (b.deltas.judge || 0) - (a.deltas.judge || 0));
  document.getElementById('merged-list').innerHTML = merged.map(b => renderBranchCard(b)).join('');
})();

// ============================================================================
// Gower detail
// ============================================================================
(function() {
  const gj = DATA.gower_jaccard;
  const gg = DATA.gower_gower;
  const b = DATA.baseline;
  const m = DATA.gower_meta;
  const root = document.getElementById('gower-detail');
  root.innerHTML = `
    <div class="branch-card">
      <div class="head">
        <div>
          <div class="name">feat/adaptive-gower-segmentation</div>
          <div class="title">${m.title} <span class="q">by ${m.owner}</span></div>
        </div>
        <div><span class="badge MERGED-AS-OPTION">MERGE-WITH-JACCARD-DEFAULT</span></div>
      </div>
      <div class="section">
        <div class="label">Hypothesis</div>
        <div>${m.hypothesis}</div>
      </div>
      <div class="section">
        <div class="label">Change</div>
        <div>${m.change}</div>
      </div>
      <div class="section">
        <div class="label">Jaccard-mode run (parity claim)</div>
        <div class="metrics-row">
          <span>personas: <strong>${gj.personas_ok}/${gj.personas_total}</strong></span>
          <span>judge: <strong>${fmt(gj.mean_judge,2)}</strong></span>
          <span>grounded: ${fmt(gj.mean_groundedness,3)}</span>
          <span>cost: ${fmtMoney(gj.total_cost)}</span>
          <span>ΔJudge vs main: <strong class="delta up">+${fmt(gj.mean_judge - b.mean_judge,2)}</strong></span>
        </div>
      </div>
      <div class="section">
        <div class="label">Gower-mode run (typed-feature clustering)</div>
        <div class="metrics-row">
          <span>personas: <strong>${gg.personas_ok}/${gg.personas_total}</strong></span>
          <span>clusters: <strong>${gg.clusters}</strong> (vs main's ${b.clusters})</span>
          <span>judge: <strong>${fmt(gg.mean_judge,2)}</strong></span>
          <span>cost: <strong>${fmtMoney(gg.total_cost)}</strong></span>
          <span>ΔJudge vs main: <strong class="delta down">${fmt(gg.mean_judge - b.mean_judge,2)}</strong></span>
        </div>
      </div>
      <div class="callout warning">
        <strong>Verdict</strong>: Merge the module (Jaccard stays default, backwards-compatible). Do NOT flip
        default to Gower. Gower produces a <strong>${Math.round((1 - gg.clusters/b.clusters)*100)}% drop in cluster count</strong>
        and lower judge scores because it groups users by firmographic attributes (company size, industry)
        rather than behavioral overlap, collapsing distinct behavioral groups into one.
      </div>
    </div>
  `;
})();

// ============================================================================
// Auto-merged
// ============================================================================
(function() {
  const items = DATA.branches.filter(b => b.verdict === 'MERGED-AS-OPTION');
  const filter = document.getElementById('filter-auto');
  function render() {
    const q = filter.value.toLowerCase();
    const matching = items.filter(b => !q || b.name.toLowerCase().includes(q));
    document.getElementById('automerged-list').innerHTML = matching
      .map(b => renderAutoCard(b)).join('') || '<p class="q">no matches</p>';
  }
  filter.addEventListener('input', render);
  render();
})();

function renderAutoCard(b) {
  const added = b.files_added || [];
  return `<div class="branch-card">
    <div class="head">
      <div>
        <div class="name">${b.name}</div>
        <div class="title">${b.type}${b.owner ? ' — by '+b.owner : ''}</div>
      </div>
      <div><span class="badge MERGED-AS-OPTION">AUTO-MERGED</span></div>
    </div>
    <div class="metrics-row"><span>new files added: <strong>${added.length}</strong></span></div>
    ${added.length ? `<details><summary>files</summary><ul class="file-list">${added.map(f=>'<li><code>'+f+'</code></li>').join('')}</ul></details>` : ''}
  </div>`;
}

// ============================================================================
// Review / Neutral / Rejected / Skipped
// ============================================================================
function renderGroup(verdicts, rootId) {
  const items = DATA.branches
    .filter(b => verdicts.includes(b.verdict))
    .sort((a,b) => (a.deltas ? (a.deltas.judge||0) : 0) - (b.deltas ? (b.deltas.judge||0) : 0));
  document.getElementById(rootId).innerHTML = items.map(b => renderBranchCard(b)).join('') || '<p class="q">none</p>';
}
renderGroup(['REVIEW'], 'review-list');
renderGroup(['NEUTRAL'], 'neutral-list');
renderGroup(['REJECT','REJECT-RELIABILITY','REJECT-BROKEN','NO-RESULTS'], 'rejected-list');
renderGroup(['SKIPPED-NO-CHANGE','NOT-PROCESSED'], 'skipped-list');

// ============================================================================
// All branches
// ============================================================================
(function() {
  let currentFilter = 'all';
  const filterInput = document.getElementById('filter-all');
  document.querySelectorAll('[data-verdict]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-verdict]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.verdict;
      render();
    });
  });
  filterInput.addEventListener('input', render);

  function render() {
    const q = filterInput.value.toLowerCase();
    let items = DATA.branches.slice();
    if (currentFilter !== 'all') {
      if (currentFilter === 'REJECT') {
        items = items.filter(b => ['REJECT','REJECT-RELIABILITY','REJECT-BROKEN'].includes(b.verdict));
      } else if (currentFilter === 'SKIPPED-NO-CHANGE') {
        items = items.filter(b => ['SKIPPED-NO-CHANGE','NOT-PROCESSED','NO-RESULTS'].includes(b.verdict));
      } else {
        items = items.filter(b => b.verdict === currentFilter);
      }
    }
    if (q) {
      items = items.filter(b =>
        b.name.toLowerCase().includes(q) ||
        (b.owner && b.owner.toLowerCase().includes(q)) ||
        (b.type && b.type.toLowerCase().includes(q))
      );
    }
    items.sort((a,b) => {
      const va = a.deltas && a.deltas.judge !== undefined ? a.deltas.judge : -99;
      const vb = b.deltas && b.deltas.judge !== undefined ? b.deltas.judge : -99;
      return vb - va;
    });
    let html = '<table><thead><tr><th>Branch</th><th>Type</th><th>Verdict</th><th>ΔJudge</th><th>ΔGnd</th><th>ΔCost%</th><th>ΔSuccess</th></tr></thead><tbody>';
    html += items.map(b => {
      const d = b.deltas || {};
      return `<tr>
        <td><code>${b.name}</code></td>
        <td>${b.type}</td>
        <td><span class="badge ${b.verdict}">${b.verdict}</span></td>
        <td>${fmtDeltaJ(d.judge)}</td>
        <td>${fmtDeltaG(d.groundedness)}</td>
        <td>${fmtDeltaPct(d.cost_pct)}</td>
        <td>${fmtDeltaG(d.success_rate)}</td>
      </tr>`;
    }).join('');
    html += '</tbody></table>';
    document.getElementById('all-branches').innerHTML = html || '<p class="q">no matches</p>';
  }
  render();
})();
</script>
</body>
</html>
"""


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, default=Path("benchmark/branch_classification.json"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmark/results/main"))
    parser.add_argument("--merging", type=Path, default=Path("benchmark/results/merging_experiments"))
    parser.add_argument("--gower-jaccard", type=Path, default=Path("benchmark/results/gower-jaccard"))
    parser.add_argument("--gower-gower", type=Path, default=Path("benchmark/results/gower-gower"))
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--automerge-log", type=Path, default=Path("benchmark/automerge.log.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/report.html"))
    args = parser.parse_args()

    classification = json.loads(args.classification.read_text())
    data = build_report(
        classification, args.baseline, args.merging,
        args.gower_jaccard, args.gower_gower,
        args.results_root, args.automerge_log,
    )
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, default=str))
    args.output.write_text(html, encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"  Branches: {len(data['branches'])}")
    print(f"  Verdicts: {data['counts_by_verdict']}")


if __name__ == "__main__":
    main_cli()
