"""Experiment 6.12: Power-user heuristic.

Hypothesis: Every tenant's persona set should contain a power-user archetype
(the heaviest-engagement segment). If missing, it's a coverage gap.

Setup:
  1. Synthesize personas from the golden tenant.
  2. Score each persona for power-user signals.
  3. Report whether a power-user was found and how strong the signal is.

Metrics:
  - Power-user inclusion rate (1.0 = found, 0.0 = missing)
  - Per-persona power-user score
  - Signal breakdown (goals, vocab, role matches)

Usage:
    python scripts/experiment_6_12.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from power_user_check import (  # noqa: E402
    PowerUserReport,
    check_power_user,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


def get_clusters() -> list[ClusterData]:
    crawler_records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(c) for c in cluster_dicts]


def print_results(report: PowerUserReport, synth_cost: float) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 6.12 — POWER-USER HEURISTIC — RESULTS")
    p("=" * 100)
    p(f"\n  Personas: {report.n_personas}")

    found_str = "YES" if report.power_user_found else "WARNING: NOT FOUND"
    p(f"  Power-user found: {found_str}")
    p(f"  Best match: {report.best_match} (score={report.best_score:.3f})")

    p(f"\n-- PER-PERSONA SCORES --")
    header = f"  {'Persona':<45}{'Goals':>8}{'Vocab':>8}{'Role':>8}{'Score':>8}{'Status':>12}"
    p(header)
    p("  " + "-" * 89)

    for s in report.scores:
        status = "POWER USER" if s.is_power_user else "regular"
        p(f"  {s.persona_name[:43]:<45}{s.goal_matches:>8}{s.vocab_matches:>8}"
          f"{s.role_matches:>8}{s.total_score:>8.3f}{status:>12}")

    p(f"\n-- SIGNAL ASSESSMENT --")
    if report.power_user_found and report.best_score > 0.5:
        strength = "STRONG FINDING"
        detail = (f"Clear power-user persona detected ({report.best_match}, "
                  f"score={report.best_score:.3f}). The pipeline reliably "
                  f"identifies the heaviest-engagement segment.")
    elif report.power_user_found:
        strength = "MODERATE FINDING"
        detail = (f"Power-user persona present but signal is moderate "
                  f"(score={report.best_score:.3f}). The persona has some "
                  f"power-user traits but could be more distinctive.")
    else:
        strength = "STRONG FINDING (gap)"
        detail = ("No power-user persona detected. The persona set is missing "
                  "the heaviest-engagement segment — a critical coverage gap.")

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")
    p(f"\n  Synthesis cost: ${synth_cost:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


async def main():
    print("=" * 72)
    print("EXPERIMENT 6.12: Power-user heuristic")
    print("Hypothesis: Every tenant should have a power-user persona")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/3] Synthesizing personas...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    personas: list[dict] = []
    synth_cost = 0.0
    for cluster in clusters:
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            personas.append(pd)
            synth_cost += result.total_cost_usd
            print(f"      {result.persona.name}")
        except Exception as e:
            print(f"      FAILED: {e}")

    if not personas:
        print("ERROR: No personas synthesized")
        sys.exit(1)

    print(f"\n[2/3] Checking for power-user archetype...")
    report = check_power_user(personas)

    print(f"\n[3/3] Generating report...")
    report_text = print_results(report, synth_cost)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "6.12",
        "title": "Power-user heuristic",
        "hypothesis": "Every tenant should have a power-user persona",
        "model": settings.default_model,
        "n_personas": report.n_personas,
        "power_user_found": report.power_user_found,
        "best_match": report.best_match,
        "best_score": report.best_score,
        "inclusion_rate": report.inclusion_rate,
        "scores": [
            {
                "persona_name": s.persona_name,
                "goal_matches": s.goal_matches,
                "vocab_matches": s.vocab_matches,
                "role_matches": s.role_matches,
                "total_score": s.total_score,
                "is_power_user": s.is_power_user,
            }
            for s in report.scores
        ],
        "synthesis_cost_usd": synth_cost,
    }

    (output_dir / "exp_6_12_results.json").write_text(json.dumps(results_data, indent=2))
    (output_dir / "exp_6_12_report.txt").write_text(report_text)
    print(f"\nResults saved to: {output_dir / 'exp_6_12_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
