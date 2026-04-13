"""Build a self-contained HTML comparison page (main vs merging_experiments).

Output: benchmark/comparison.html — single file, no external deps,
embedded data + Chart.js from CDN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_dir(d: Path) -> dict:
    """Load all tenant results from a directory."""
    out = {"summary": {}, "tenants": {}}
    if not d.exists():
        return out
    for f in d.glob("bench_*.json"):
        out["tenants"][f.stem] = json.loads(f.read_text())
    summary_file = d / "summary.json"
    if summary_file.exists():
        out["summary"] = json.loads(summary_file.read_text())
    return out


def build_data(main_dir: Path, merge_dir: Path) -> dict:
    """Build the data structure passed to the HTML."""
    main_data = load_dir(main_dir)
    merge_data = load_dir(merge_dir)

    tenant_ids = sorted(set(main_data["tenants"]) | set(merge_data["tenants"]))

    def cluster_key(c: dict) -> str:
        # Use sorted record_ids as a stable join key — same input data, same key
        return "|".join(sorted(c.get("record_ids", [])))

    # Build per-tenant comparison
    tenants = []
    for tid in tenant_ids:
        m = main_data["tenants"].get(tid, {})
        x = merge_data["tenants"].get(tid, {})

        m_clusters = m.get("clusters", [])
        x_clusters = x.get("clusters", [])

        # Index by cluster_key (record content)
        m_by_key = {cluster_key(c): c for c in m_clusters}
        x_by_key = {cluster_key(c): c for c in x_clusters}
        all_keys = sorted(set(m_by_key) | set(x_by_key))

        clusters = []
        for k in all_keys:
            mc = m_by_key.get(k, {})
            xc = x_by_key.get(k, {})
            sample_ids = (mc.get("record_ids") or xc.get("record_ids") or [])[:5]
            clusters.append({
                "record_id_sample": sample_ids,
                "n_records": mc.get("n_records") or xc.get("n_records", 0),
                "main": _persona_summary(mc),
                "merge": _persona_summary(xc),
            })

        def tenant_aggs(tdata: dict) -> dict:
            cs = tdata.get("clusters", [])
            ok = [c for c in cs if not c.get("failed")]
            return {
                "n_clusters": len(cs),
                "n_ok": len(ok),
                "success_rate": len(ok) / len(cs) if cs else 0,
                "mean_g": _mean([c.get("groundedness", 0) for c in ok]),
                "mean_judge": _mean([
                    c["judge_score"]["overall"] for c in ok
                    if isinstance(c.get("judge_score"), dict) and "overall" in c["judge_score"]
                ]),
                "total_cost": sum(c.get("cost_usd", 0) for c in cs),
                "total_latency": tdata.get("total_latency_s", 0),
                "mean_attempts": _mean([c.get("attempts", 0) for c in ok]),
            }

        tenants.append({
            "tenant_id": tid,
            "industry": m.get("industry") or x.get("industry", ""),
            "product": m.get("product") or x.get("product", ""),
            "n_records": m.get("n_records") or x.get("n_records", 0),
            "main_aggs": tenant_aggs(m),
            "merge_aggs": tenant_aggs(x),
            "clusters": clusters,
        })

    # Top-level aggregates
    def overall(data: dict) -> dict:
        all_g, all_j, all_att = [], [], []
        total_cost = 0.0
        total_personas = 0
        total_ok = 0
        total_latency = 0.0
        for t in data["tenants"].values():
            for c in t.get("clusters", []):
                if not c.get("failed"):
                    all_g.append(c.get("groundedness", 0))
                    all_att.append(c.get("attempts", 0))
                    js = c.get("judge_score")
                    if isinstance(js, dict) and "overall" in js:
                        all_j.append(float(js["overall"]))
                    total_ok += 1
                total_personas += 1
                total_cost += c.get("cost_usd", 0)
            total_latency += t.get("total_latency_s", 0)
        return {
            "n_tenants": len(data["tenants"]),
            "n_personas": total_personas,
            "n_ok": total_ok,
            "success_rate": total_ok / total_personas if total_personas else 0,
            "mean_g": _mean(all_g),
            "mean_judge": _mean(all_j),
            "mean_attempts": _mean(all_att),
            "total_cost": total_cost,
            "total_latency": total_latency,
        }

    return {
        "main": overall(main_data),
        "merge": overall(merge_data),
        "tenants": tenants,
    }


def _persona_summary(c: dict) -> dict:
    if not c:
        return {}
    pj = c.get("persona_json", {})
    js = c.get("judge_score") if isinstance(c.get("judge_score"), dict) else {}
    return {
        "name": c.get("persona_name", ""),
        "groundedness": c.get("groundedness", 0),
        "cost": c.get("cost_usd", 0),
        "attempts": c.get("attempts", 0),
        "latency": c.get("latency_s", 0),
        "failed": c.get("failed", False),
        "error": c.get("error", ""),
        "judge_overall": js.get("overall"),
        "judge_grounded": js.get("grounded"),
        "judge_distinctive": js.get("distinctive"),
        "judge_coherent": js.get("coherent"),
        "judge_actionable": js.get("actionable"),
        "judge_voice_fidelity": js.get("voice_fidelity"),
        "summary": pj.get("summary", ""),
        "goals": pj.get("goals", []),
        "pains": pj.get("pains", []),
        "vocabulary": pj.get("vocabulary", []),
        "sample_quotes": pj.get("sample_quotes", []),
    }


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>main vs merging_experiments — benchmark comparison</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 0; background: #0f1419; color: #e6edf3;
  }
  header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 24px 32px;
  }
  h1 { font-size: 22px; margin: 0 0 4px 0; }
  .subtitle { color: #8b949e; font-size: 14px; }
  main { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }
  h2 { font-size: 18px; margin: 32px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid #30363d; }
  .summary-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;
  }
  .metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px;
  }
  .metric-card .label { font-size: 11px; text-transform: uppercase; color: #8b949e; margin-bottom: 6px; letter-spacing: 0.5px; }
  .metric-row { display: flex; justify-content: space-between; align-items: baseline; }
  .metric-side { display: flex; flex-direction: column; }
  .metric-side .name { font-size: 10px; color: #8b949e; }
  .metric-side .value { font-size: 18px; font-weight: 600; }
  .delta { font-size: 13px; font-weight: 600; padding: 2px 6px; border-radius: 4px; }
  .delta.up { background: rgba(46, 160, 67, 0.15); color: #3fb950; }
  .delta.down { background: rgba(248, 81, 73, 0.15); color: #f85149; }
  .delta.flat { background: rgba(139, 148, 158, 0.15); color: #8b949e; }
  .tenant-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }
  .tenant-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .tenant-name { font-size: 16px; font-weight: 600; }
  .tenant-meta { font-size: 12px; color: #8b949e; }
  .tenant-stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 12px; }
  .stat { background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px; font-size: 12px; }
  .stat .label { color: #8b949e; font-size: 10px; text-transform: uppercase; }
  .stat .row { display: flex; justify-content: space-between; }
  .stat .row .v { font-weight: 600; }
  .stat .row.main .v { color: #58a6ff; }
  .stat .row.merge .v { color: #d29922; }
  .clusters { margin-top: 8px; }
  .cluster {
    border-top: 1px dashed #30363d; padding: 12px 0;
  }
  .cluster:first-child { border-top: none; padding-top: 0; }
  .cluster-header { display: flex; justify-content: space-between; align-items: baseline; }
  .cluster-id { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; color: #8b949e; }
  .persona-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
  .persona {
    background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 10px;
  }
  .persona.main { border-left: 3px solid #58a6ff; }
  .persona.merge { border-left: 3px solid #d29922; }
  .persona .name { font-weight: 600; margin-bottom: 4px; font-size: 13px; }
  .persona .meta { font-size: 11px; color: #8b949e; margin-bottom: 6px; }
  .persona .judge { font-size: 11px; margin-bottom: 6px; }
  .persona .judge span { display: inline-block; padding: 1px 5px; margin-right: 4px; border-radius: 3px; background: rgba(139, 148, 158, 0.15); }
  .persona details { margin-top: 6px; font-size: 11px; }
  .persona summary { cursor: pointer; color: #58a6ff; }
  .persona ul { margin: 4px 0 4px 18px; padding: 0; font-size: 11px; }
  .persona li { margin-bottom: 2px; }
  .persona.failed { opacity: 0.5; }
  .persona.failed .name { color: #f85149; }
  .scoreboard { display: flex; gap: 12px; align-items: center; }
  .scoreboard .badge {
    padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
  }
  .badge.main { background: rgba(88, 166, 255, 0.15); color: #58a6ff; }
  .badge.merge { background: rgba(210, 153, 34, 0.15); color: #d29922; }
  .toolbar { display: flex; gap: 8px; align-items: center; margin: 8px 0 16px 0; }
  .toolbar input {
    background: #0d1117; border: 1px solid #30363d; color: #e6edf3;
    padding: 6px 10px; border-radius: 4px; font-size: 13px; flex: 1; max-width: 320px;
  }
</style>
</head>
<body>
<header>
  <h1>main vs merging_experiments — 42-persona benchmark</h1>
  <div class="subtitle">
    <span class="badge main">main</span> baseline,
    <span class="badge merge">merge</span> = main + 4 MERGE branches + 64 schema-additive/eval-only.
    Lower is better for cost, attempts, latency. Higher is better for groundedness, judge, success rate.
  </div>
</header>
<main>

<h2>Overall</h2>
<div class="summary-grid" id="summary-grid"></div>

<h2>Per-tenant comparison</h2>
<div class="toolbar">
  <input type="text" id="filter" placeholder="Filter by tenant or persona name…">
</div>
<div id="tenant-list"></div>

</main>

<script>
const DATA = __DATA__;

function fmt(n, d=2) { return (n == null || isNaN(n)) ? '—' : n.toFixed(d); }
function fmtPct(n, d=1) { return (n == null || isNaN(n)) ? '—' : (n*100).toFixed(d) + '%'; }
function fmtMoney(n) { return n == null ? '—' : '$' + n.toFixed(4); }
function deltaClass(d, higherBetter=true) {
  if (Math.abs(d) < 0.01) return 'flat';
  return ((d > 0) === higherBetter) ? 'up' : 'down';
}
function deltaText(d, fmtFn=fmt, sign=true) {
  if (d == null || isNaN(d)) return '';
  const s = sign && d > 0 ? '+' : '';
  return s + fmtFn(d);
}

// Summary grid
function renderSummary() {
  const root = document.getElementById('summary-grid');
  const m = DATA.main, x = DATA.merge;
  const cards = [
    {label:'Mean judge score (1-5)', main: m.mean_judge, merge: x.mean_judge, fmt: v=>fmt(v,2), better: 'up'},
    {label:'Mean groundedness', main: m.mean_g, merge: x.mean_g, fmt: v=>fmt(v,3), better: 'up'},
    {label:'Success rate', main: m.success_rate, merge: x.success_rate, fmt: v=>fmtPct(v,1), better: 'up'},
    {label:'Mean attempts', main: m.mean_attempts, merge: x.mean_attempts, fmt: v=>fmt(v,2), better: 'down'},
    {label:'Personas (ok/total)', main: m.n_ok+'/'+m.n_personas, merge: x.n_ok+'/'+x.n_personas, fmt: v=>v, better: null},
    {label:'Total cost', main: m.total_cost, merge: x.total_cost, fmt: v=>fmtMoney(v), better: 'down'},
    {label:'Total latency (s)', main: m.total_latency, merge: x.total_latency, fmt: v=>fmt(v,1), better: 'down'},
    {label:'Tenants', main: m.n_tenants, merge: x.n_tenants, fmt: v=>v, better: null},
  ];
  root.innerHTML = cards.map(c => {
    const d = (typeof c.main === 'number' && typeof c.merge === 'number') ? c.merge - c.main : null;
    const dCls = c.better ? deltaClass(d, c.better === 'up') : 'flat';
    const dStr = c.better && d !== null ? `<span class="delta ${dCls}">${deltaText(d, c.fmt)}</span>` : '';
    return `<div class="metric-card">
      <div class="label">${c.label}</div>
      <div class="metric-row">
        <div class="metric-side"><div class="name">main</div><div class="value">${c.fmt(c.main)}</div></div>
        <div class="metric-side"><div class="name">merge</div><div class="value">${c.fmt(c.merge)}</div></div>
      </div>
      <div style="text-align:right; margin-top:4px;">${dStr}</div>
    </div>`;
  }).join('');
}

function renderTenants(filter='') {
  const root = document.getElementById('tenant-list');
  const lower = filter.toLowerCase();
  const html = DATA.tenants.map(t => {
    const matches = !filter
      || t.tenant_id.toLowerCase().includes(lower)
      || t.clusters.some(c =>
          (c.main.name||'').toLowerCase().includes(lower) ||
          (c.merge.name||'').toLowerCase().includes(lower));
    if (!matches) return '';

    const ma = t.main_aggs, xa = t.merge_aggs;

    function statRow(label, fmtFn, mainVal, mergeVal, better) {
      const d = (typeof mainVal === 'number' && typeof mergeVal === 'number') ? mergeVal - mainVal : null;
      const dCls = better ? deltaClass(d, better === 'up') : 'flat';
      const dStr = better && d !== null ? `<span class="delta ${dCls}">${deltaText(d, fmtFn)}</span>` : '';
      return `<div class="stat">
        <div class="label">${label}</div>
        <div class="row main"><span>main</span><span class="v">${fmtFn(mainVal)}</span></div>
        <div class="row merge"><span>merge</span><span class="v">${fmtFn(mergeVal)}</span></div>
        <div style="text-align:right; margin-top:2px;">${dStr}</div>
      </div>`;
    }

    const stats = [
      statRow('Judge', v=>fmt(v,2), ma.mean_judge, xa.mean_judge, 'up'),
      statRow('Groundedness', v=>fmt(v,3), ma.mean_g, xa.mean_g, 'up'),
      statRow('Success', v=>fmtPct(v,0), ma.success_rate, xa.success_rate, 'up'),
      statRow('Attempts', v=>fmt(v,2), ma.mean_attempts, xa.mean_attempts, 'down'),
      statRow('Cost', v=>fmtMoney(v), ma.total_cost, xa.total_cost, 'down'),
    ];

    const clusters = t.clusters.map(c => {
      function pCard(p, side) {
        if (!p || !p.name) return `<div class="persona ${side}"><div class="name">—</div></div>`;
        const cls = (side==='main'?'main':'merge') + (p.failed ? ' failed' : '');
        const judge = p.judge_overall != null
          ? `<div class="judge">judge=${fmt(p.judge_overall,1)}
              <span>g=${p.judge_grounded}</span>
              <span>d=${p.judge_distinctive}</span>
              <span>c=${p.judge_coherent}</span>
              <span>a=${p.judge_actionable}</span>
              <span>v=${p.judge_voice_fidelity}</span>
            </div>` : '';
        const lists = `
          <details><summary>summary</summary><div>${p.summary||''}</div></details>
          <details><summary>goals (${(p.goals||[]).length})</summary><ul>${(p.goals||[]).map(g=>'<li>'+g+'</li>').join('')}</ul></details>
          <details><summary>pains (${(p.pains||[]).length})</summary><ul>${(p.pains||[]).map(g=>'<li>'+g+'</li>').join('')}</ul></details>
          <details><summary>vocabulary (${(p.vocabulary||[]).length})</summary><div>${(p.vocabulary||[]).join(', ')}</div></details>
          <details><summary>sample quotes (${(p.sample_quotes||[]).length})</summary><ul>${(p.sample_quotes||[]).map(q=>'<li>"'+q+'"</li>').join('')}</ul></details>
        `;
        return `<div class="persona ${cls}">
          <div class="name">${p.name||'(failed)'}</div>
          <div class="meta">g=${fmt(p.groundedness,2)} · att=${p.attempts} · ${fmtMoney(p.cost)} · ${fmt(p.latency,1)}s${p.failed?' · FAILED':''}</div>
          ${judge}
          ${lists}
        </div>`;
      }
      const records = (c.record_id_sample || []).join(', ');
      return `<div class="cluster">
        <div class="cluster-header">
          <span><strong>${c.n_records} records</strong></span>
          <span class="cluster-id">${records}…</span>
        </div>
        <div class="persona-pair">${pCard(c.main, 'main')}${pCard(c.merge, 'merge')}</div>
      </div>`;
    }).join('');

    return `<div class="tenant-card">
      <div class="tenant-header">
        <div>
          <div class="tenant-name">${t.tenant_id}</div>
          <div class="tenant-meta">${t.industry} — ${t.product} — ${t.n_records} records</div>
        </div>
      </div>
      <div class="tenant-stats">${stats.join('')}</div>
      <div class="clusters">${clusters}</div>
    </div>`;
  }).join('');
  root.innerHTML = html;
}

renderSummary();
renderTenants();
document.getElementById('filter').addEventListener('input', e => renderTenants(e.target.value));
</script>
</body>
</html>
"""


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--merge-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = build_data(args.main_dir, args.merge_dir)
    payload = json.dumps(data, default=str)
    html = HTML_TEMPLATE.replace("__DATA__", payload)
    args.output.write_text(html, encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"  Tenants: {len(data['tenants'])}")
    print(f"  main:  {data['main']['n_ok']}/{data['main']['n_personas']} personas, "
          f"judge={data['main']['mean_judge']:.2f}, ${data['main']['total_cost']:.4f}")
    print(f"  merge: {data['merge']['n_ok']}/{data['merge']['n_personas']} personas, "
          f"judge={data['merge']['mean_judge']:.2f}, ${data['merge']['total_cost']:.4f}")


if __name__ == "__main__":
    main_cli()
