"""Experiment 1.07: Field interdependence.

Hypothesis: Some persona fields are load-bearing (removing them degrades
multiple scoring dimensions) while others are decorative (removing them
has minimal impact). Understanding field interdependence helps us decide
which fields to prioritize in synthesis and which can be simplified.

Approach:
  1. Generate a control persona via the standard pipeline on golden tenant
  2. For each ablatable field, set it to [] in a copy and re-score
  3. Build dependency matrix: rows = removed field, cols = dimension, cells = delta
  4. Classify fields as load-bearing vs decorative

Usage:
    python scripts/experiment_1_07.py
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
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))

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

from evals.field_interdependence import (  # noqa: E402
    format_dependency_matrix,
    report_to_dict,
    run_ablation_study,
)
from evals.judge_helper_1_07 import JudgeBackend, LLMJudge  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-1.07-field-interdependence"


# ── Pipeline helpers ──────────────────────────────────────────────────

def get_clusters() -> list[ClusterData]:
    """Run ingest + segmentation."""
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


async def generate_control_persona(
    cluster: ClusterData,
    backend: AnthropicBackend,
) -> dict:
    """Generate a persona and return it as a dict."""
    result = await synthesize(cluster, backend)
    persona_dict = result.persona.model_dump(mode="json")
    return persona_dict


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 1.07: Field Interdependence")
    print("Hypothesis: Some fields are load-bearing, others decorative.")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Judge uses sonnet for cost-efficiency (10 judge calls per persona)
    judge_backend = JudgeBackend(client=client, model="claude-sonnet-4-20250514")
    judge = LLMJudge(backend=judge_backend, model="claude-sonnet-4-20250514", calibration="few_shot")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Ingest + segment
    print("\n[1/4] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    if not clusters:
        print("ERROR: No clusters found")
        sys.exit(1)

    # Use first cluster for the experiment
    cluster = clusters[0]
    print(f"      Using cluster: {cluster.cluster_id}")

    # Step 2: Generate control persona
    print("\n[2/4] Generating control persona...")
    t0 = time.monotonic()
    control_persona = await generate_control_persona(cluster, backend)
    synth_time = time.monotonic() - t0
    print(f"      Control persona: {control_persona.get('name', 'unknown')}")
    print(f"      Synthesis time: {synth_time:.1f}s")

    # Save control persona
    control_path = OUTPUT_DIR / "control_persona.json"
    control_path.write_text(json.dumps(control_persona, indent=2, default=str))

    # Step 3: Run ablation study
    print("\n[3/4] Running ablation study (scoring control + 9 ablations)...")
    t0 = time.monotonic()
    report = await run_ablation_study(control_persona, judge)
    ablation_time = time.monotonic() - t0
    print(f"      Ablation study took {ablation_time:.1f}s")

    # Step 4: Report results
    print("\n[4/4] Results\n")
    matrix_str = format_dependency_matrix(report)
    print(matrix_str)

    # Save results
    results_data = {
        "experiment": "1.07",
        "title": "Field Interdependence",
        "hypothesis": "Some persona fields are load-bearing (removal degrades 2+ dimensions), others are decorative",
        "model_synthesis": settings.default_model,
        "model_judge": "claude-sonnet-4-20250514",
        "calibration": "few_shot",
        "tenant_id": TENANT_ID,
        "cluster_id": cluster.cluster_id,
        "synthesis_time_s": synth_time,
        "ablation_time_s": ablation_time,
        "report": report_to_dict(report),
        "matrix_text": matrix_str,
    }

    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"\nResults saved to: {results_path}")

    # Generate FINDINGS.md
    findings = generate_findings(report, results_data)
    findings_path = OUTPUT_DIR / "FINDINGS.md"
    findings_path.write_text(findings)
    print(f"Findings saved to: {findings_path}")


def generate_findings(report, results_data: dict) -> str:
    """Generate FINDINGS.md from results."""
    lines = [
        "# Experiment 1.07: Field Interdependence",
        "",
        "## Hypothesis",
        "Some persona fields are load-bearing (removing them degrades 2+ scoring",
        "dimensions significantly) while others are decorative (minimal impact).",
        "",
        "## Method",
        "1. Generated a control persona via the standard pipeline (tenant_acme_corp)",
        "2. For each of 9 ablatable fields, set it to [] in a copy",
        "3. Re-scored each ablated copy with the LLM judge (few-shot calibrated)",
        "4. Computed score deltas and classified fields",
        "",
        f"- Synthesis model: {results_data['model_synthesis']}",
        f"- Judge model: {results_data['model_judge']}",
        f"- Calibration: {results_data['calibration']}",
        "",
        "## Dependency Matrix",
        "",
        "```",
        results_data.get("matrix_text", "N/A"),
        "```",
        "",
        "## Classification",
        "",
        f"**Load-bearing fields** ({len(report.load_bearing_fields)}):",
    ]

    for f in report.load_bearing_fields:
        abl = next((a for a in report.ablations if a.field_name == f), None)
        if abl:
            drops = [f"{d}: {v:+.1f}" for d, v in abl.deltas.items() if v <= -0.5]
            lines.append(f"- `{f}` — drops: {', '.join(drops)}")

    lines.append("")
    lines.append(f"**Decorative fields** ({len(report.decorative_fields)}):")
    for f in report.decorative_fields:
        abl = next((a for a in report.ablations if a.field_name == f), None)
        if abl:
            lines.append(f"- `{f}` (overall delta: {abl.overall_delta:+.1f})")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Load-bearing fields are those whose removal degrades the persona across",
        "multiple quality dimensions. These fields provide structural coherence",
        "and should be prioritized in synthesis. Decorative fields contribute",
        "marginally and could potentially be simplified or made optional.",
        "",
        "## Decision",
        "",
        "TBD after reviewing results — adopt/reject/defer.",
    ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    asyncio.run(main())
