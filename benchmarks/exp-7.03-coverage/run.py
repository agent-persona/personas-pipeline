"""exp-7.03 coverage: ours vs persona-hub over tenant_acme_corp records.

No LLM calls. Uses sentence-transformer embeddings (all-MiniLM-L6-v2)
to score how well each persona set "covers" the tenant's records.

For every record, we compute max cosine similarity to any persona in a
given persona-set. Coverage@t = fraction of records where max_sim >= t.

Compared:
  - ours         : our shipped output/persona_{00,01}.json (flattened
                   to summary+goals+pains+quotes)
  - persona-hub-100 : 100 personas sampled from the public persona.jsonl
                   at https://huggingface.co/datasets/proj-persona/PersonaHub
                   (one-sentence archetype strings).

Outputs:
  output/experiments/exp-7.03-oss-bench-coverage/results.json
  output/experiments/exp-7.03-oss-bench-coverage/FINDINGS.md
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from crawler import fetch_all

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "experiments" / "exp-7.03-oss-bench-coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TENANT = "tenant_acme_corp"
PERSONA_HUB_URL = "https://huggingface.co/datasets/proj-persona/PersonaHub/resolve/main/persona.jsonl"
PERSONA_HUB_SAMPLE = 100
COVERAGE_THRESHOLDS = [0.20, 0.30, 0.40, 0.50]
MAIN_REPORT_THRESHOLD = 0.30

EMB_MODEL = "all-MiniLM-L6-v2"


def _record_text(r) -> str:
    parts = [f"source={r.source}"]
    if r.behaviors:
        parts.append("behaviors=" + ", ".join(r.behaviors))
    if r.pages:
        parts.append("pages=" + ", ".join(r.pages))
    payload_keys = [k for k in (r.payload or {}) if k != "session_duration"]
    if payload_keys:
        parts.append("payload_keys=" + ", ".join(payload_keys))
    return ". ".join(parts)


def _load_our_personas() -> list[dict]:
    """Read output/persona_*.json and flatten to name + summary + goals + pains + quotes."""
    out = []
    for p in sorted((ROOT / "output").glob("persona_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        persona = data.get("persona") or {}
        name = persona.get("name") or "(unnamed)"
        summary = persona.get("summary") or ""
        goals = persona.get("goals") or []
        pains = persona.get("pains") or []
        quotes = persona.get("sample_quotes") or []
        text = (
            f"Name: {name}. "
            f"Summary: {summary} "
            f"Goals: {'; '.join(goals[:5])}. "
            f"Pains: {'; '.join(pains[:5])}. "
            f"Sample quotes: {'; '.join(quotes[:3])}."
        )
        out.append({"id": p.stem, "name": name, "text": text})
    return out


def _load_persona_hub(n: int) -> list[dict]:
    """Fetch ~n personas from persona-hub via HTTP range request.

    Short circuit: if the fetch fails, write a clear error to findings
    rather than silently using a hand-rolled fallback.
    """
    # Fetch enough bytes — each persona line is ~100 bytes avg, so 32KB
    # comfortably yields hundreds.
    req = urllib.request.Request(
        PERSONA_HUB_URL, headers={"Range": f"bytes=0-{n * 200}"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read().decode("utf-8", errors="replace")
    out = []
    for line in data.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        persona_text = obj.get("persona")
        if not persona_text:
            continue
        out.append({
            "id": f"phub_{len(out):04d}",
            "name": "",
            "text": persona_text,
        })
        if len(out) >= n:
            break
    return out


@dataclass
class SetResult:
    set_name: str
    n_personas: int
    mean_max_sim: float
    median_max_sim: float
    coverage_at_threshold: dict[str, float]  # stringified threshold -> %
    per_record_max_sim: list[float]


def _score(embs_personas: np.ndarray, embs_records: np.ndarray) -> np.ndarray:
    """Return per-record max cosine similarity to any persona in the set."""
    # Both matrices are L2-normalized by the encoder (we enforce), so dot product = cos sim.
    sims = embs_records @ embs_personas.T  # (N_records, N_personas)
    return sims.max(axis=1)


def run() -> None:
    print(f"[exp-7.03] loading records for tenant={TENANT}")
    records = fetch_all(TENANT)
    record_texts = [_record_text(r) for r in records]
    print(f"[exp-7.03]   {len(records)} records")

    print(f"[exp-7.03] loading our personas from output/")
    ours = _load_our_personas()
    print(f"[exp-7.03]   {len(ours)} of ours")

    print(f"[exp-7.03] fetching {PERSONA_HUB_SAMPLE} persona-hub entries from HF")
    try:
        phub = _load_persona_hub(PERSONA_HUB_SAMPLE)
        phub_err = None
    except Exception as e:
        phub = []
        phub_err = f"{type(e).__name__}: {e}"
    print(f"[exp-7.03]   got {len(phub)} persona-hub samples (err={phub_err})")

    print(f"[exp-7.03] encoding with {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL)
    record_embs = model.encode(record_texts, normalize_embeddings=True, show_progress_bar=False)
    our_embs = model.encode([p["text"] for p in ours], normalize_embeddings=True, show_progress_bar=False)
    phub_embs = (
        model.encode([p["text"] for p in phub], normalize_embeddings=True, show_progress_bar=False)
        if phub
        else np.zeros((0, record_embs.shape[1]))
    )

    sets = [("ours", our_embs, ours), ("persona-hub-100", phub_embs, phub)]
    results: list[SetResult] = []
    for name, emb, src in sets:
        if emb.shape[0] == 0:
            continue
        maxs = _score(emb, record_embs).tolist()
        cov = {}
        for t in COVERAGE_THRESHOLDS:
            cov[f"{t:.2f}"] = float(sum(1 for m in maxs if m >= t) / len(maxs))
        results.append(SetResult(
            set_name=name,
            n_personas=emb.shape[0],
            mean_max_sim=float(np.mean(maxs)),
            median_max_sim=float(np.median(maxs)),
            coverage_at_threshold=cov,
            per_record_max_sim=maxs,
        ))
        print(
            f"[exp-7.03] {name}: n={emb.shape[0]} mean_max={np.mean(maxs):.3f} "
            f"cov@{MAIN_REPORT_THRESHOLD:.2f}={cov[f'{MAIN_REPORT_THRESHOLD:.2f}']*100:.1f}%"
        )

    # Per-record paired comparison: for how many records is ours' max-sim
    # strictly higher than persona-hub's?
    win_rate = None
    if len(results) == 2:
        a = results[0].per_record_max_sim
        b = results[1].per_record_max_sim
        if len(a) == len(b) and a:
            wins = sum(1 for x, y in zip(a, b) if x > y)
            ties = sum(1 for x, y in zip(a, b) if x == y)
            win_rate = {
                "ours_wins": wins,
                "ties": ties,
                "phub_wins": len(a) - wins - ties,
                "n": len(a),
                "mean_gap": float(np.mean([x - y for x, y in zip(a, b)])),
            }

    out = {
        "experiment": "exp-7.03-oss-bench-coverage",
        "tenant": TENANT,
        "embedding_model": EMB_MODEL,
        "n_records": len(records),
        "persona_hub_url": PERSONA_HUB_URL,
        "persona_hub_error": phub_err,
        "coverage_thresholds": COVERAGE_THRESHOLDS,
        "sets": [asdict(r) for r in results],
        "paired_win_rate": win_rate,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_findings(out)
    print(f"[exp-7.03] wrote {OUT_DIR / 'results.json'}")


def _write_findings(out: dict) -> None:
    md = [
        "# exp-7.03 — coverage: ours vs persona-hub",
        "",
        f"**Tenant:** `{out['tenant']}` ({out['n_records']} records)  ",
        f"**Embedding model:** `{out['embedding_model']}` (local, no LLM spend)  ",
        f"**persona-hub source:** {out['persona_hub_url']}  ",
        "",
        "## Coverage of tenant records",
        "",
        "For each record we compute the max cosine similarity to any persona "
        "in the set. Coverage@t = fraction of records with max_sim ≥ t.",
        "",
        "| Set | #personas | mean max-sim | median max-sim | cov@0.20 | cov@0.30 | cov@0.40 | cov@0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out["sets"]:
        cov = r["coverage_at_threshold"]
        md.append(
            f"| {r['set_name']} | {r['n_personas']} | {r['mean_max_sim']:.3f} | "
            f"{r['median_max_sim']:.3f} | "
            f"{cov['0.20']*100:.1f}% | {cov['0.30']*100:.1f}% | "
            f"{cov['0.40']*100:.1f}% | {cov['0.50']*100:.1f}% |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "- `ours` personas are flattened from `output/persona_*.json` "
        "(name + summary + top-5 goals/pains + 3 sample quotes). They were "
        "synthesized from this tenant's records, so a coverage win is "
        "expected — the interesting numbers are the *magnitude* of the gap."
    )
    md.append(
        "- `persona-hub-100` entries are one-sentence generic archetypes "
        "sampled from a 1B-persona pool designed for synthetic data "
        "generation, not audience segmentation. Low coverage here isn't "
        "a flaw of persona-hub — it's confirmation that its personas aren't "
        "designed to index a specific tenant's behavior."
    )
    md.append(
        "- The gap in mean max-sim is the load-bearing finding: it's the "
        "measurable value of grounded synthesis over sampling from a generic "
        "pool."
    )
    wr = out.get("paired_win_rate")
    if wr:
        md.append("")
        md.append("## Paired per-record win-rate (ours vs persona-hub-100)")
        md.append("")
        md.append(
            f"- ours wins: **{wr['ours_wins']}/{wr['n']}** "
            f"({wr['ours_wins']/wr['n']*100:.0f}%)"
        )
        md.append(f"- ties: {wr['ties']}/{wr['n']}")
        md.append(f"- persona-hub wins: {wr['phub_wins']}/{wr['n']}")
        md.append(f"- mean gap (ours − phub): {wr['mean_gap']:+.3f}")
    if out.get("persona_hub_error"):
        md.append(f"\n**Note:** persona-hub fetch failed ({out['persona_hub_error']}) — only ours was scored.")
    (OUT_DIR / "FINDINGS.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    run()
