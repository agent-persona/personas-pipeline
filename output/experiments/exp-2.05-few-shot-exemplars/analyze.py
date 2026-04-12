"""Post-hoc analyzer for exp-2.05 few-shot-exemplars sweep.

Read-only over `output/persona_*.json`. Computes groundedness, schema
validity, cost-per-persona, Jaccard overlap with injected exemplars, tracer
tokens, quote paraphrase overlap, and EXAMPLE_rec_* leakage per cluster × N.

Run from repo root:
    python output/experiments/exp-2.05-few-shot-exemplars/analyze.py
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evaluation.metrics import (  # noqa: E402
    cost_per_persona,
    groundedness_rate,
    schema_validity,
)
from segmentation.engine.clusterer import jaccard_similarity  # noqa: E402
from synthesis.engine.prompt_builder import _load_golden_examples  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / "exp-2.05-few-shot-exemplars"
LOG_PATH = EXPERIMENT_DIR / "structural_comparison.log"

# Distinctive vocabulary tokens from each exemplar vertical. Case-insensitive
# substring search on the full persona text — any hit at N>0 that is absent at
# N=0 is a smoking-gun cloning signal.
TRACER_TOKENS = [
    # clinical (exemplar 0)
    "IRB", "CRF", "monitor visit", "REDCap", "21 CFR",
    # tax (exemplar 1)
    "1099", "EBITDA", "S-corp", "Drake",
    # teacher (exemplar 2)
    "lesson plan", "IEP", "DIBELS", "Title I",
    # restaurant (exemplar 3)
    "food cost variance", "mise en place",
    # planner (exemplar 4)
    "zoning variance", "comp plan", "CEQA",
]

EXAMPLE_REC_PATTERN = re.compile(r"EXAMPLE_rec_\w+")


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", s.lower()) if len(t) > 2}


def _vocab_set(p: dict) -> set[str]:
    return {v.lower().strip() for v in p.get("vocabulary", [])}


def _goals_tokens(p: dict) -> set[str]:
    toks: set[str] = set()
    for g in p.get("goals", []):
        toks |= _tokens(g)
    return toks


def _persona_text(p: dict) -> str:
    """Flatten a persona dict to a single searchable string."""
    return json.dumps(p, default=str)


def _count_tracer_hits(text: str) -> dict[str, int]:
    lower = text.lower()
    return {tok: lower.count(tok.lower()) for tok in TRACER_TOKENS if tok.lower() in lower}


def _quote_lcs_ratio(synth_quotes: list[str], exemplar_quotes: list[str]) -> float:
    """Max longest-common-substring ratio across any (synth, exemplar) quote pair."""
    best = 0.0
    for a in synth_quotes:
        for b in exemplar_quotes:
            m = difflib.SequenceMatcher(None, a.lower(), b.lower()).find_longest_match(
                0, len(a), 0, len(b)
            )
            ratio = m.size / max(len(a), len(b), 1)
            if ratio > best:
                best = ratio
    return best


def analyze() -> str:
    exemplars = _load_golden_examples()
    exemplar_dicts = [e.model_dump(mode="json") for e in exemplars]

    persona_files = sorted(OUTPUT_DIR.glob("persona_*.json"))
    if not persona_files:
        raise SystemExit("No persona_*.json in output/ — run pipeline first.")

    out: list[str] = []
    out.append("# exp-2.05 few-shot-exemplars — structural comparison")
    out.append("")
    out.append(f"Analyzed {len(persona_files)} cluster artifacts against "
               f"{len(exemplars)} golden exemplars.")
    out.append("")

    # Global aggregates by N
    all_by_n: dict[int, list[dict]] = {}
    cost_by_n: dict[int, float] = {}
    ground_by_n: dict[int, list[float]] = {}

    for pf in persona_files:
        entry = json.loads(pf.read_text())
        cid = entry["cluster_id"]
        out.append(f"## {cid}")
        out.append("")
        out.append(
            "| N | schema_ok | groundedness | cost $ | max Jaccard vocab | "
            "max Jaccard goals | tracer hits | quote LCS | EXAMPLE_rec_ leaks |"
        )
        out.append(
            "|---|---|---|---|---|---|---|---|---|"
        )
        for s in entry["exemplar_sweep"]:
            n = s["n_exemplars"]
            p = s["persona"]
            cost_by_n[n] = cost_by_n.get(n, 0.0) + s["cost_usd"]
            ground_by_n.setdefault(n, []).append(s["groundedness"])

            if p is None:
                out.append(
                    f"| {n} | FAIL | {s['groundedness']:.2f} | "
                    f"{s['cost_usd']:.4f} | — | — | SYNTHESIS_FAILED | — | — |"
                )
                continue

            all_by_n.setdefault(n, []).append(p)

            # Schema check
            try:
                PersonaV1.model_validate(p)
                schema_ok = "OK"
            except Exception:
                schema_ok = "FAIL"

            # Jaccard max vs injected exemplars (first-N nesting)
            injected = exemplar_dicts[:n] if n > 0 else []
            synth_vocab = _vocab_set(p)
            synth_goals = _goals_tokens(p)
            max_jacc_vocab = 0.0
            max_jacc_goals = 0.0
            for ex in injected:
                max_jacc_vocab = max(
                    max_jacc_vocab, jaccard_similarity(synth_vocab, _vocab_set(ex))
                )
                max_jacc_goals = max(
                    max_jacc_goals, jaccard_similarity(synth_goals, _goals_tokens(ex))
                )

            # Tracer hits — count appearances of distinctive exemplar tokens
            text = _persona_text(p)
            tracer_hits = _count_tracer_hits(text)
            tracer_str = (
                ", ".join(f"{k}={v}" for k, v in tracer_hits.items())
                if tracer_hits else "none"
            )

            # Quote LCS vs exemplar quotes (use all 5 exemplars as pool —
            # we want to catch copying at N>0 regardless of exact exemplar)
            all_exemplar_quotes: list[str] = []
            for ex in exemplar_dicts:
                all_exemplar_quotes.extend(ex.get("sample_quotes", []))
            lcs = _quote_lcs_ratio(p.get("sample_quotes", []), all_exemplar_quotes)

            # EXAMPLE_rec_* leakage — both in persona dict and in violations
            leaks = len(EXAMPLE_REC_PATTERN.findall(text))
            for v in s.get("groundedness_violations", []) or []:
                leaks += len(EXAMPLE_REC_PATTERN.findall(v))

            out.append(
                f"| {n} | {schema_ok} | {s['groundedness']:.2f} | "
                f"{s['cost_usd']:.4f} | {max_jacc_vocab:.2f} | "
                f"{max_jacc_goals:.2f} | {tracer_str} | {lcs:.2f} | {leaks} |"
            )
        out.append("")

    # Global summary
    out.append("## Global summary by N")
    out.append("")
    out.append("| N | schema_validity | groundedness_rate | cost_per_persona $ |")
    out.append("|---|---|---|---|")
    for n in sorted(all_by_n):
        sv = schema_validity(all_by_n[n], PersonaV1)
        gr = groundedness_rate(
            [SimpleNamespace(score=s) for s in ground_by_n[n]]
        )
        cpp = cost_per_persona(cost_by_n[n], len(all_by_n[n]))
        out.append(f"| {n} | {sv:.2f} | {gr:.2f} | {cpp:.4f} |")

    return "\n".join(out) + "\n"


def main() -> None:
    report = analyze()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report)
    print(report)
    print(f"\nWrote {LOG_PATH}")


if __name__ == "__main__":
    main()
