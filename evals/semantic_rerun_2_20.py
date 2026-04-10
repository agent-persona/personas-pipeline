"""
Track B2: Re-run exp-2.20 with semantic groundedness proxy.

Exp-2.20 tested transcript-first persona synthesis. Both baseline and
transcript-first personas scored 1.0 on structural groundedness (ceiling).
This script applies the semantic groundedness proxy to detect richer grounding.

Usage:
    python evals/semantic_rerun_2_20.py
"""

import json
import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Semantic groundedness proxy — inline, no external imports
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "for", "to", "with", "that",
    "their", "our", "is", "are", "was", "were", "be", "been", "have", "has",
    "this", "it", "its", "at", "by", "from", "on", "as", "not", "but", "so",
    "they", "we", "you", "he", "she", "also", "which", "what", "when", "how",
    "can", "will", "would", "could", "should", "may", "might", "do", "did",
    "use", "used", "using", "get", "got", "need", "needs", "work", "works",
}


def _claim_tokens(text):
    return {
        w.lower().strip(".,;:'\"()")
        for w in text.split()
        if len(w) > 3 and w.lower().strip(".,;:'\"()") not in _STOPWORDS
    }


def _payload_tokens(text):
    parts = re.split(r"[^a-zA-Z0-9]+", text)
    return {p.lower() for p in parts if len(p) > 3 and p.lower() not in _STOPWORDS}


def semantic_groundedness_proxy(persona, cluster):
    records_by_id = {r["record_id"]: r for r in cluster.get("sample_records", [])}
    evidence_map = {e["field_path"]: e["record_ids"] for e in persona.get("source_evidence", [])}
    FIELDS = ["goals", "pains", "motivations", "objections"]
    overlaps, weak, covered = [], [], 0
    for field in FIELDS:
        for i, item in enumerate(persona.get(field, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            field_path = f"{field}.{i}"
            record_ids = evidence_map.get(field_path, [])
            if not record_ids:
                continue
            covered += 1
            claim_tokens = _claim_tokens(text)
            record_tokens = set()
            for rid in record_ids:
                rec = records_by_id.get(rid, {})
                for v in rec.get("payload", {}).values():
                    record_tokens |= _payload_tokens(str(v))
            overlap = len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
            overlaps.append(overlap)
            if overlap < 0.1:
                weak.append({
                    "field_path": field_path,
                    "claim": text[:100],
                    "overlap": round(overlap, 4),
                })
    total_claims = sum(len(persona.get(f, [])) for f in FIELDS)
    score = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return {
        "semantic_score": round(score, 4),
        "weak_pairs": weak,
        "claim_count": len(overlaps),
        "weak_count": len(weak),
        "coverage": round(covered / max(total_claims, 1), 4),
    }


# ---------------------------------------------------------------------------
# Qualitative token analysis
# ---------------------------------------------------------------------------

def token_overlap_detail(persona, cluster):
    """Per-claim token overlap breakdown for qualitative inspection."""
    records_by_id = {r["record_id"]: r for r in cluster.get("sample_records", [])}
    evidence_map = {e["field_path"]: e["record_ids"] for e in persona.get("source_evidence", [])}
    FIELDS = ["goals", "pains", "motivations", "objections"]
    detail = []
    for field in FIELDS:
        for i, item in enumerate(persona.get(field, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            field_path = f"{field}.{i}"
            record_ids = evidence_map.get(field_path, [])
            if not record_ids:
                continue
            claim_tokens = _claim_tokens(text)
            record_tokens = set()
            for rid in record_ids:
                rec = records_by_id.get(rid, {})
                for v in rec.get("payload", {}).values():
                    record_tokens |= _payload_tokens(str(v))
            shared = sorted(claim_tokens & record_tokens)
            overlap = len(shared) / max(len(claim_tokens), 1)
            detail.append({
                "field_path": field_path,
                "claim": text[:120],
                "claim_tokens": sorted(claim_tokens),
                "shared_tokens": shared,
                "overlap": round(overlap, 4),
            })
    return detail


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

WORKTREE = Path(__file__).parent.parent
CLUSTER_DIR = WORKTREE / "output" / "clusters"

# Persona source paths on the remote branch
_REMOTE_BRANCH = "origin/exp-2.20-reverse-engineered-persona"
_REMOTE_PERSONA_PATHS = {
    0: {
        "baseline": "output/baseline_persona_00.json",
        "transcript": "output/transcript_first_persona_00.json",
    },
    1: {
        "baseline": "output/baseline_persona_01.json",
        "transcript": "output/transcript_first_persona_01.json",
    },
}


def fetch_personas():
    """
    Fetch exp-2.20 personas from the remote branch into output/personas/.
    Re-runs are safe — files are overwritten each time so results stay current.
    """
    persona_dir = WORKTREE / "output" / "personas"
    persona_dir.mkdir(parents=True, exist_ok=True)
    local_paths = {}
    for idx, paths in _REMOTE_PERSONA_PATHS.items():
        local_paths[idx] = {}
        for kind, remote_path in paths.items():
            local_file = persona_dir / f"{kind}_persona_{idx:02d}.json"
            result = subprocess.run(
                ["git", "show", f"{_REMOTE_BRANCH}:{remote_path}"],
                capture_output=True,
                text=True,
                cwd=WORKTREE,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to fetch {_REMOTE_BRANCH}:{remote_path}\n{result.stderr}"
                )
            local_file.write_text(result.stdout)
            local_paths[idx][kind] = str(local_file)
    return local_paths


def load_persona(path):
    with open(path) as f:
        data = json.load(f)
    # persona may be nested under "persona" key
    if "persona" in data and isinstance(data["persona"], dict):
        return data["persona"]
    return data


def load_cluster(idx):
    path = CLUSTER_DIR / f"cluster_{idx:02d}.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(persona_files=None):
    if persona_files is None:
        persona_files = fetch_personas()

    results = {"clusters": []}
    all_baseline_scores = []
    all_transcript_scores = []

    for idx in [0, 1]:
        cluster = load_cluster(idx)
        baseline_persona = load_persona(persona_files[idx]["baseline"])
        transcript_persona = load_persona(persona_files[idx]["transcript"])

        baseline_result = semantic_groundedness_proxy(baseline_persona, cluster)
        transcript_result = semantic_groundedness_proxy(transcript_persona, cluster)

        baseline_detail = token_overlap_detail(baseline_persona, cluster)
        transcript_detail = token_overlap_detail(transcript_persona, cluster)

        delta = round(transcript_result["semantic_score"] - baseline_result["semantic_score"], 4)

        cluster_result = {
            "cluster_id": cluster["cluster_id"],
            "baseline": {
                **baseline_result,
                "token_detail": baseline_detail,
            },
            "transcript_first": {
                **transcript_result,
                "token_detail": transcript_detail,
            },
            "semantic_score_delta": delta,
        }
        results["clusters"].append(cluster_result)
        all_baseline_scores.append(baseline_result["semantic_score"])
        all_transcript_scores.append(transcript_result["semantic_score"])

        print(f"\n=== Cluster {idx:02d} ({cluster['cluster_id']}) ===")
        print(f"  Baseline       semantic_score={baseline_result['semantic_score']:.4f}  "
              f"claim_count={baseline_result['claim_count']}  "
              f"weak_count={baseline_result['weak_count']}  "
              f"coverage={baseline_result['coverage']:.4f}")
        print(f"  Transcript-1st semantic_score={transcript_result['semantic_score']:.4f}  "
              f"claim_count={transcript_result['claim_count']}  "
              f"weak_count={transcript_result['weak_count']}  "
              f"coverage={transcript_result['coverage']:.4f}")
        print(f"  Delta          {delta:+.4f}")
        if transcript_result["weak_pairs"]:
            print(f"  Transcript weak pairs:")
            for wp in transcript_result["weak_pairs"]:
                print(f"    [{wp['field_path']}] overlap={wp['overlap']:.4f} — {wp['claim']}")
        if baseline_result["weak_pairs"]:
            print(f"  Baseline weak pairs:")
            for wp in baseline_result["weak_pairs"]:
                print(f"    [{wp['field_path']}] overlap={wp['overlap']:.4f} — {wp['claim']}")

    # Aggregate
    avg_baseline = round(sum(all_baseline_scores) / len(all_baseline_scores), 4)
    avg_transcript = round(sum(all_transcript_scores) / len(all_transcript_scores), 4)
    avg_delta = round(avg_transcript - avg_baseline, 4)

    both_clusters_positive = all(
        c["semantic_score_delta"] >= 0.05 for c in results["clusters"]
    )
    signal = "STRONG" if both_clusters_positive else (
        "WEAK" if avg_delta > 0 else "NOISE"
    )
    recommendation = "ADOPT" if signal == "STRONG" else ("DEFER" if signal == "WEAK" else "REJECT")

    results["summary"] = {
        "avg_baseline_semantic_score": avg_baseline,
        "avg_transcript_first_semantic_score": avg_transcript,
        "avg_semantic_score_delta": avg_delta,
        "signal": signal,
        "recommendation": recommendation,
    }

    print(f"\n=== Summary ===")
    print(f"  avg_baseline_semantic_score       : {avg_baseline:.4f}")
    print(f"  avg_transcript_first_semantic_score: {avg_transcript:.4f}")
    print(f"  avg_semantic_score_delta           : {avg_delta:+.4f}")
    print(f"  signal                             : {signal}")
    print(f"  recommendation                     : {recommendation}")

    # Write results.json
    out_dir = WORKTREE / "output" / "experiments" / "exp-b2-rerun-2.20-semantic"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")

    return results


if __name__ == "__main__":
    results = run()  # fetches personas from remote branch automatically

    # Write FINDINGS_v2.md after we have the numbers
    clusters = results["clusters"]
    summary = results["summary"]
    c0 = clusters[0]
    c1 = clusters[1]

    findings = f"""# Findings: Track B2 — Exp-2.20 Semantic Groundedness Re-Run

## Context

Experiment 2.20 tested **transcript-first persona synthesis** (interview-style):
the pipeline first synthesizes a simulated transcript from cluster records, then
derives the persona from that transcript. Both the baseline and transcript-first
personas scored **1.0 on structural groundedness** — the metric was at ceiling
and produced zero signal (`NOISE`, delta=0.0).

This re-run applies the **semantic groundedness proxy** to measure token-level
overlap between persona claims and the cited source records.

---

## Original Exp-2.20 Result (Structural Groundedness)

| Method | Structural Score | Delta |
|---|---|---|
| Baseline | 1.0 | — |
| Transcript-first | 1.0 | **0.0 (NOISE)** |

The structural metric only checks whether `source_evidence` entries are present
and cite valid record IDs. Both personas cite all fields, so the score is always
1.0 regardless of claim quality.

---

## New Semantic Scores

### Cluster 00 — `{c0["cluster_id"]}`

| Method | semantic_score | claim_count | weak_count | coverage |
|---|---|---|---|---|
| Baseline | {c0["baseline"]["semantic_score"]} | {c0["baseline"]["claim_count"]} | {c0["baseline"]["weak_count"]} | {c0["baseline"]["coverage"]} |
| Transcript-first | {c0["transcript_first"]["semantic_score"]} | {c0["transcript_first"]["claim_count"]} | {c0["transcript_first"]["weak_count"]} | {c0["transcript_first"]["coverage"]} |

**semantic_score_delta: {c0["semantic_score_delta"]:+.4f}**

### Cluster 01 — `{c1["cluster_id"]}`

| Method | semantic_score | claim_count | weak_count | coverage |
|---|---|---|---|---|
| Baseline | {c1["baseline"]["semantic_score"]} | {c1["baseline"]["claim_count"]} | {c1["baseline"]["weak_count"]} | {c1["baseline"]["coverage"]} |
| Transcript-first | {c1["transcript_first"]["semantic_score"]} | {c1["transcript_first"]["claim_count"]} | {c1["transcript_first"]["weak_count"]} | {c1["transcript_first"]["coverage"]} |

**semantic_score_delta: {c1["semantic_score_delta"]:+.4f}**

---

## Aggregate Summary

| Metric | Value |
|---|---|
| avg_baseline_semantic_score | {summary["avg_baseline_semantic_score"]} |
| avg_transcript_first_semantic_score | {summary["avg_transcript_first_semantic_score"]} |
| avg_semantic_score_delta | {summary["avg_semantic_score_delta"]:+.4f} |
| Signal | **{summary["signal"]}** |
| Recommendation | **{summary["recommendation"]}** |

---

## Weak-Pair Analysis

Weak pairs are claim-evidence pairs with token overlap < 0.10 (claim vocabulary
almost entirely absent from cited records).

**Cluster 00 — Baseline weak pairs:** {len(c0["baseline"]["weak_pairs"])}
**Cluster 00 — Transcript-first weak pairs:** {len(c0["transcript_first"]["weak_pairs"])}

**Cluster 01 — Baseline weak pairs:** {len(c1["baseline"]["weak_pairs"])}
**Cluster 01 — Transcript-first weak pairs:** {len(c1["transcript_first"]["weak_pairs"])}

---

## Qualitative Observation

The semantic proxy measures whether the words used in persona claims appear in
the payload of the cited source records. A higher score means claims stay closer
to the vocabulary of the underlying data rather than drifting into paraphrase or
hallucination.

Transcript-first synthesis routes all information through a simulated interview
transcript before persona generation. This introduces an additional abstraction
layer: the model first rewrites record payloads as natural interview speech, then
generates claims from that speech. The effect on token overlap depends on whether
the transcript preserves or transforms the raw vocabulary.

- If transcript-first **preserves** source vocabulary → higher semantic overlap
- If transcript-first **paraphrases** heavily → lower overlap even when factually grounded

The delta observed here ({summary["avg_semantic_score_delta"]:+.4f} average) reflects this tradeoff.

---

## Conclusion

**Signal: {summary["signal"]}**

{"Both clusters show semantic_score_delta >= 0.05, confirming that transcript-first produces claims with meaningfully higher token overlap against cited records. The transcript-first method stays closer to source vocabulary — a strong signal that the additional synthesis step does not hallucinate away from evidence." if summary["signal"] == "STRONG" else "The semantic delta does not reach the STRONG threshold (>= 0.05 for both clusters). Transcript-first does not consistently produce higher token overlap, suggesting the intermediate transcript step may paraphrase rather than preserve source vocabulary." if summary["signal"] == "WEAK" else "No meaningful semantic advantage detected for transcript-first synthesis. The intermediate transcript step appears to transform vocabulary enough to reduce or neutralize token overlap with cited records."}

**Recommendation: {summary["recommendation"]}**
"""

    findings_path = (
        WORKTREE / "output" / "experiments" / "exp-b2-rerun-2.20-semantic" / "FINDINGS_v2.md"
    )
    with open(findings_path, "w") as f:
        f.write(findings)
    print(f"Wrote {findings_path}")
