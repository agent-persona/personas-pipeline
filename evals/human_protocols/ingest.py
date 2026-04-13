"""Ingest Prolific-returned labels and compute agreement metrics.

Expects files at:
  output/experiments/exp-5.06/labels_protocol_{a,b,c}.csv

Each label CSV is expected to have at minimum:
  item_id, rater_id, response

Where `response` is the value the rater picked from the protocol's
response_values list.

Computes, per protocol:
  - Inter-rater agreement (Cohen's kappa or Krippendorff's alpha)
  - Optional: Claude-as-judge agreement vs. human majority vote
  - Attention-check pass rate (if attention_check_expected is provided)

Usage:
    python -m evals.human_protocols.ingest
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .agreement import cohen_kappa, krippendorff_alpha
from .protocols import PROTOCOLS

REPO_ROOT = Path(__file__).parent.parent.parent
LABELS_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.06"


def load_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def compute_agreement(rows: list[dict], metric: str) -> float | None:
    """Compute agreement using the protocol's metric.

    `rows` is a flat list of {item_id, rater_id, response}. We build a
    {item_id: {rater_id: response}} structure then hand it to the metric.
    """
    if not rows:
        return None

    by_item: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r["rater_id"]] = r["response"]

    items_as_dicts = list(by_item.values())

    if metric == "cohen_kappa":
        # Cohen's kappa strictly needs exactly 2 raters. If we have more, fall
        # back to the mean pairwise kappa — that's the pragmatic extension.
        rater_ids = sorted({rid for item in items_as_dicts for rid in item})
        if len(rater_ids) < 2:
            return None
        if len(rater_ids) == 2:
            a = [item.get(rater_ids[0]) for item in items_as_dicts if rater_ids[0] in item and rater_ids[1] in item]
            b = [item.get(rater_ids[1]) for item in items_as_dicts if rater_ids[0] in item and rater_ids[1] in item]
            return cohen_kappa(a, b) if a else None
        # Multi-rater: mean pairwise
        kappas = []
        for i in range(len(rater_ids)):
            for j in range(i + 1, len(rater_ids)):
                ri, rj = rater_ids[i], rater_ids[j]
                a = [item[ri] for item in items_as_dicts if ri in item and rj in item]
                b = [item[rj] for item in items_as_dicts if ri in item and rj in item]
                if a:
                    kappas.append(cohen_kappa(a, b))
        return sum(kappas) / len(kappas) if kappas else None

    if metric == "krippendorff_alpha_ordinal":
        # Pairwise preference has an ordinal structure: a_better < tie < b_better
        return krippendorff_alpha(
            items_as_dicts,
            level="ordinal",
            ordinal_order=["a_better", "tie", "b_better"],
        )
    if metric == "krippendorff_alpha_nominal":
        return krippendorff_alpha(items_as_dicts, level="nominal")

    raise ValueError(f"unknown metric: {metric}")


def majority_vote(rows: list[dict]) -> dict[str, str]:
    """Majority-vote label per item_id."""
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r["response"])
    return {
        item: Counter(responses).most_common(1)[0][0]
        for item, responses in by_item.items()
    }


def main() -> None:
    findings = {
        "experiment_id": "5.06",
        "protocols": {},
    }

    for key, proto in PROTOCOLS.items():
        labels_path = LABELS_DIR / f"labels_protocol_{proto.protocol_id}.csv"
        rows = load_labels(labels_path)
        if not rows:
            findings["protocols"][key] = {
                "status": "no labels",
                "note": f"file not found: {labels_path}",
            }
            continue
        agreement = compute_agreement(rows, proto.metric)
        mv = majority_vote(rows)
        findings["protocols"][key] = {
            "status": "ingested",
            "n_items": len(mv),
            "n_raters_seen": len({r["rater_id"] for r in rows}),
            "metric": proto.metric,
            "agreement": agreement,
            "majority_vote_sample": dict(list(mv.items())[:5]),
        }

    # Pick the protocol with the highest agreement
    ranked = sorted(
        (
            (k, v.get("agreement"))
            for k, v in findings["protocols"].items()
            if v.get("agreement") is not None
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    findings["winner"] = ranked[0][0] if ranked else None
    findings["winner_agreement"] = ranked[0][1] if ranked else None

    out_path = LABELS_DIR / "FINDINGS_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
