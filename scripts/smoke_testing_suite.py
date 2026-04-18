"""Smoke test for the ported testing suite.

Runs Tier 1 scorers (no LLM/embedder needed) against:
  - a real pipeline persona via the PersonaV1 -> testing.Persona adapter
  - a pre-adapted persona from output/eval_personas/

Exits non-zero if any scorer raises an unexpected exception. Expected
test-level failures (e.g. missing optional fields) are reported but do not
fail the smoke run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make evaluation/ importable without installing.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

from evaluation.testing import Persona, SourceContext, SuiteRunner, persona_v1_to_testing  # noqa: E402
from evaluation.testing.scorers.structural.schema_compliance import SchemaComplianceScorer  # noqa: E402
from evaluation.testing.scorers.structural.completeness import CompletenessScorer  # noqa: E402
from evaluation.testing.scorers.structural.consistency import ConsistencyScorer  # noqa: E402


def run_tier1(persona: Persona, ctx: SourceContext, label: str) -> int:
    print(f"\n=== Tier 1 suite against {label} ({persona.id}) ===")
    runner = SuiteRunner([SchemaComplianceScorer(), CompletenessScorer(), ConsistencyScorer()])
    results = runner.run(persona, ctx)
    errors = 0
    for r in results:
        status = "PASS" if r.passed else "fail"
        print(f"  [{status}] {r.dimension_id} {r.dimension_name}: score={r.score:.2f}"
              + (f"  errors={r.errors}" if r.errors and r.passed else ""))
        if r.errors and not r.passed:
            print(f"         errors: {r.errors}")
    return errors


def main() -> int:
    ctx = SourceContext(id="smoke", text="No real source — smoke test.")

    # 1. Adapter path: real PersonaV1 from output/persona_00.json
    p1_raw = json.loads((REPO / "output" / "persona_00.json").read_text())["persona"]
    p1 = persona_v1_to_testing(p1_raw)
    run_tier1(p1, ctx, "output/persona_00.json (adapted from PersonaV1)")

    # 2. Native path: pre-adapted persona from output/eval_personas/
    eval_dir = REPO / "output" / "eval_personas"
    eval_files = sorted(eval_dir.glob("*.json"))
    if eval_files:
        p2_raw = json.loads(eval_files[0].read_text())
        p2 = Persona(**p2_raw)
        run_tier1(p2, ctx, f"output/eval_personas/{eval_files[0].name}")

    print("\nSmoke OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
