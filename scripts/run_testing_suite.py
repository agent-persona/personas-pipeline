"""Run the tiered persona scorer suite against persona JSON files.

Usage:
    # single persona
    python scripts/run_testing_suite.py output/persona_00.json

    # every persona in a directory (PersonaV1 dumps or pre-adapted shape)
    python scripts/run_testing_suite.py output/

    # restrict to a tier
    python scripts/run_testing_suite.py output/persona_00.json --tier 1
    python scripts/run_testing_suite.py output/persona_00.json --tier 2

    # emit machine-readable output
    python scripts/run_testing_suite.py output/persona_00.json --json-out results.json

The script accepts two persona shapes:
  1. ``{"persona": {...}, ...}`` — pipeline output format (``output/persona_*.json``)
     The nested ``persona`` dict is expected to be a PersonaV1 dump and is
     passed through the adapter.
  2. ``{...}`` — a persona_eval-native dict with top-level ``id``/``name``/etc.
     (as produced into ``output/eval_personas/``).

Scorers that require set-level context or a populated conversation transcript
are reported but skipped when inputs don't satisfy their contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from evaluation.testing import (  # noqa: E402
    Persona,
    SourceContext,
    SuiteRunner,
    persona_v1_to_testing,
)
from evaluation.testing.scorers.all import get_all_scorers  # noqa: E402


def load_persona(path: Path) -> Persona:
    """Load a persona from either pipeline-output shape or pre-adapted shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "persona" in data and isinstance(data["persona"], dict):
        # Pipeline-output shape: { cluster_id, persona: <PersonaV1>, ... }
        return persona_v1_to_testing(data["persona"])
    # Pre-adapted shape (e.g. output/eval_personas/*.json)
    return Persona(**data)


def iter_persona_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.glob("*.json") if p.is_file())
    raise SystemExit(f"not a file or directory: {target}")


def run(
    persona_files: list[Path],
    *,
    tier_filter: int | None,
) -> dict:
    scorers = get_all_scorers()
    if tier_filter is not None:
        scorers = [s for s in scorers if s.tier == tier_filter]
    runner = SuiteRunner(scorers)

    report: dict = {"tier_filter": tier_filter, "personas": []}
    per_tier_pass: dict[int, Counter] = defaultdict(Counter)

    for path in persona_files:
        persona = load_persona(path)
        ctx = SourceContext(id=persona.id, text="")
        results = runner.run(persona, ctx, tier_filter=tier_filter)

        persona_entry = {
            "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "id": persona.id,
            "name": persona.name,
            "results": [r.model_dump() for r in results],
        }
        report["personas"].append(persona_entry)

        print(f"\n=== {persona.name}  ({path.name}) ===")
        current_tier = None
        for r in results:
            scorer = next((s for s in scorers if s.dimension_id == r.dimension_id), None)
            tier = scorer.tier if scorer else -1
            if tier != current_tier:
                current_tier = tier
                print(f"  -- tier {tier} --")
            if r.details.get("skipped"):
                status = "skip"
            elif r.passed:
                status = "PASS"
            else:
                status = "FAIL"
            line = f"    [{status}] {r.dimension_id:<6} {r.dimension_name:<40} score={r.score:.2f}"
            if r.errors:
                line += f"  errors={r.errors[:3]}"
            print(line)
            if not r.details.get("skipped"):
                per_tier_pass[tier]["pass" if r.passed else "fail"] += 1

    print("\n=== Tier summary ===")
    for tier in sorted(per_tier_pass):
        c = per_tier_pass[tier]
        total = c["pass"] + c["fail"]
        print(f"  tier {tier}: {c['pass']}/{total} passed ({c['fail']} failed)")

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="persona .json file OR directory of .json files")
    ap.add_argument("--tier", type=int, default=None, help="only run scorers of this tier (1..8)")
    ap.add_argument("--json-out", type=Path, default=None, help="write full report to this path")
    args = ap.parse_args()

    files = iter_persona_files(args.target)
    if not files:
        print(f"no persona files found under {args.target}", file=sys.stderr)
        return 2

    report = run(files, tier_filter=args.tier)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nreport -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
