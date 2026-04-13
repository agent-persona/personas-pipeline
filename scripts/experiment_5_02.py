"""Experiment 5.02: Cross-judge agreement.

Hypothesis: Different frontier LLM judges (Opus, Sonnet, GPT-class, Gemini-class)
will disagree on subjective rubric dimensions (distinctive, voice_fidelity) more
than on objective ones (grounded, coherent). Dimensions with high inter-judge
variance are low-trust and should not be used as sole signals.

Setup:
  1. Generate personas from the golden tenant using the default pipeline.
  2. Score each persona with every available judge model.
  3. Compute per-dimension agreement (std dev, spread) and pairwise MAD.
  4. Flag disagreement hotspots.

Metrics:
  - Agreement matrix (per-dimension std dev and spread across judges)
  - Pairwise mean absolute difference between judge pairs
  - Disagreement hotspots (dimensions flagged as low/medium trust)

Usage:
    python scripts/experiment_5_02.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
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

from judge_harness import (  # noqa: E402
    DIMENSIONS,
    AgreementMatrix,
    DisagreementHotspot,
    JudgeResult,
    MultiJudgeHarness,
    compute_agreement_matrix,
    find_disagreement_hotspots,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


# ── Pipeline ──────────────────────────────────────────────────────────

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


async def generate_personas(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> list[dict]:
    """Synthesize one persona per cluster and return as dicts."""
    personas = []
    for cluster in clusters:
        try:
            result = await synthesize(cluster, backend)
            persona_dict = result.persona.model_dump(mode="json")
            persona_dict["_meta"] = {
                "cluster_id": cluster.cluster_id,
                "groundedness": result.groundedness.score,
                "cost_usd": result.total_cost_usd,
                "attempts": result.attempts,
            }
            personas.append(persona_dict)
            print(f"    Synthesized: {result.persona.name} "
                  f"(groundedness={result.groundedness.score:.2f}, "
                  f"cost=${result.total_cost_usd:.4f})")
        except Exception as e:
            print(f"    FAILED cluster {cluster.cluster_id}: {e}")
    return personas


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(
    all_judge_results: list[list[JudgeResult]],
    matrix: AgreementMatrix,
    hotspots: list[DisagreementHotspot],
    persona_names: list[str],
) -> str:
    """Print and return a formatted results report."""
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 5.02 — CROSS-JUDGE AGREEMENT — RESULTS")
    p("=" * 100)

    # ── Per-persona raw scores ──
    p("\n── RAW SCORES BY JUDGE ──")
    for i, (persona_results, name) in enumerate(zip(all_judge_results, persona_names)):
        p(f"\n  Persona {i+1}: {name}")
        valid = [r for r in persona_results if r.error is None]
        if not valid:
            p("    No valid judge results")
            continue

        # Header
        header = f"    {'Dimension':<20}"
        for r in valid:
            short = r.judge_model.split("-")[1] if "-" in r.judge_model else r.judge_model
            header += f"{short:>14}"
        p(header)
        p("    " + "-" * (20 + 14 * len(valid)))

        for dim in DIMENSIONS:
            row = f"    {dim:<20}"
            for r in valid:
                score = r.dimensions.get(dim, float("nan"))
                if math.isnan(score):
                    row += f"{'ERR':>14}"
                else:
                    row += f"{score:>14.3f}"
            p(row)

        row = f"    {'OVERALL':<20}"
        for r in valid:
            if math.isnan(r.overall):
                row += f"{'ERR':>14}"
            else:
                row += f"{r.overall:>14.3f}"
        p(row)

        # Costs
        row = f"    {'cost ($)':<20}"
        for r in valid:
            row += f"{r.cost_usd:>14.4f}"
        p(row)

        # Rationales
        for r in valid:
            short = r.judge_model.split("-")[1] if "-" in r.judge_model else r.judge_model
            p(f"    [{short}] {r.rationale[:120]}")

    # ── Agreement matrix ──
    p("\n── AGREEMENT MATRIX (per-dimension) ──")
    p(f"  Judges: {matrix.n_judges}, Personas: {matrix.n_personas}")

    header = f"  {'Dimension':<20}{'Mean':>10}{'Std Dev':>10}{'Min':>10}{'Max':>10}{'Spread':>10}"
    p(header)
    p("  " + "-" * 70)

    for cell in matrix.cells:
        row = f"  {cell.dimension:<20}"
        row += f"{cell.mean_score:>10.3f}" if not math.isnan(cell.mean_score) else f"{'N/A':>10}"
        row += f"{cell.std_dev:>10.3f}" if not math.isnan(cell.std_dev) else f"{'N/A':>10}"
        row += f"{cell.min_score:>10.3f}" if not math.isnan(cell.min_score) else f"{'N/A':>10}"
        row += f"{cell.max_score:>10.3f}" if not math.isnan(cell.max_score) else f"{'N/A':>10}"
        row += f"{cell.spread:>10.3f}"
        p(row)

    # ── Pairwise MAD ──
    if matrix.pairwise_mad:
        p("\n── PAIRWISE MEAN ABSOLUTE DIFFERENCE ──")
        for pair, mad in sorted(matrix.pairwise_mad.items(), key=lambda x: x[1], reverse=True):
            if math.isnan(mad):
                p(f"  {pair:<55} N/A")
            else:
                p(f"  {pair:<55} {mad:.4f}")

    # ── Disagreement hotspots ──
    p("\n── DISAGREEMENT HOTSPOTS ──")
    if not hotspots:
        p("  No hotspots detected (all dimensions show high agreement)")
    for h in hotspots:
        trust_marker = {"low": "!!!", "medium": "! ", "high": "  "}[h.trust_level]
        p(f"  {trust_marker} {h.dimension:<20} trust={h.trust_level:<8} "
          f"std={h.std_dev:.3f} spread={h.spread:.3f}")
        for judge, score in h.scores_by_judge.items():
            short = judge.split("-")[1] if "-" in judge else judge
            if math.isnan(score):
                p(f"       {short}: N/A")
            else:
                p(f"       {short}: {score:.3f}")

    # ── Signal assessment ──
    p("\n── SIGNAL ASSESSMENT ──")
    low_trust = [h for h in hotspots if h.trust_level == "low"]
    med_trust = [h for h in hotspots if h.trust_level == "medium"]
    high_trust = [h for h in hotspots if h.trust_level == "high"]

    p(f"  High-trust dimensions: {len(high_trust)} "
      f"({', '.join(h.dimension for h in high_trust) or 'none'})")
    p(f"  Medium-trust dimensions: {len(med_trust)} "
      f"({', '.join(h.dimension for h in med_trust) or 'none'})")
    p(f"  Low-trust dimensions: {len(low_trust)} "
      f"({', '.join(h.dimension for h in low_trust) or 'none'})")

    # Overall signal strength
    if low_trust:
        avg_spread = statistics.mean([h.spread for h in low_trust])
        if avg_spread > 0.3:
            strength = "STRONG FINDING"
            detail = ("Major disagreement detected — low-trust dimensions should NOT "
                      "be used as standalone metrics without human calibration.")
        else:
            strength = "MODERATE FINDING"
            detail = ("Some disagreement detected — low-trust dimensions should be "
                      "used cautiously and averaged across judges.")
    elif med_trust:
        strength = "WEAK FINDING"
        detail = ("Minor disagreement — most dimensions show reasonable agreement. "
                  "Medium-trust dimensions may benefit from multi-judge averaging.")
    else:
        strength = "NULL RESULT"
        detail = ("All dimensions show high agreement across judges. "
                  "The judge system appears consistent regardless of model.")

    p(f"\n  Overall signal: {strength}")
    p(f"  {detail}")

    # Cost summary
    total_cost = sum(
        r.cost_usd for persona_results in all_judge_results
        for r in persona_results if r.error is None
    )
    p(f"\n  Total judging cost: ${total_cost:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 5.02: Cross-judge agreement")
    print("Hypothesis: Subjective dims (distinctive, voice_fidelity) show")
    print("  higher inter-judge variance than objective dims (grounded, coherent)")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Report available backends
    backends = ["anthropic (Opus, Sonnet)"]
    if os.environ.get("OPENAI_API_KEY"):
        backends.append("openai (GPT-4o)")
    if os.environ.get("GOOGLE_API_KEY"):
        backends.append("google (Gemini)")
    print(f"\nAvailable judge backends: {', '.join(backends)}")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    # Step 1: Generate personas
    print("\n[1/4] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    print("\n[2/4] Synthesizing personas...")
    personas = await generate_personas(clusters, synth_backend)
    if not personas:
        print("ERROR: No personas generated")
        sys.exit(1)
    print(f"      Generated {len(personas)} personas")

    # Step 2: Score with all judges
    print("\n[3/4] Scoring with multiple judges...")
    harness = MultiJudgeHarness.from_env(anthropic_key=settings.anthropic_api_key)
    print(f"      Active judges: {', '.join(j.name for j in harness.judges)}")

    all_judge_results: list[list[JudgeResult]] = []
    persona_names: list[str] = []

    for i, persona in enumerate(personas):
        name = persona.get("name", f"Persona {i+1}")
        persona_names.append(name)
        print(f"\n  Scoring: {name}")

        # Remove internal _meta before sending to judges
        judge_input = {k: v for k, v in persona.items() if not k.startswith("_")}
        results = await harness.score_persona(judge_input)
        all_judge_results.append(results)

        for r in results:
            short = r.judge_model.split("-")[1] if "-" in r.judge_model else r.judge_model
            if r.error:
                print(f"    {short}: FAILED ({r.error[:80]})")
            else:
                print(f"    {short}: overall={r.overall:.3f} cost=${r.cost_usd:.4f}")

    # Step 3: Compute agreement
    print("\n[4/4] Computing agreement matrix...")
    matrix = compute_agreement_matrix(all_judge_results)
    hotspots = find_disagreement_hotspots(matrix)

    # Step 4: Report
    report = print_results(all_judge_results, matrix, hotspots, persona_names)

    # Save outputs
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "5.02",
        "title": "Cross-judge agreement",
        "hypothesis": (
            "Subjective dimensions show higher inter-judge variance than "
            "objective dimensions"
        ),
        "synthesis_model": settings.default_model,
        "judges": [j.name for j in harness.judges],
        "judge_models": [j.model for j in harness.judges],
        "n_personas": len(personas),
        "n_judges": len(harness.judges),
        "personas": [
            {
                "name": name,
                "cluster_id": p.get("_meta", {}).get("cluster_id", ""),
                "synthesis_cost": p.get("_meta", {}).get("cost_usd", 0),
                "groundedness": p.get("_meta", {}).get("groundedness", 0),
                "judge_scores": [
                    {
                        "judge": r.judge_model,
                        "backend": r.judge_backend,
                        "overall": r.overall if not math.isnan(r.overall) else None,
                        "dimensions": {
                            k: v if not math.isnan(v) else None
                            for k, v in r.dimensions.items()
                        },
                        "rationale": r.rationale,
                        "cost_usd": r.cost_usd,
                        "error": r.error,
                    }
                    for r in judge_results
                ],
            }
            for name, p, judge_results in zip(
                persona_names, personas, all_judge_results,
            )
        ],
        "agreement_matrix": {
            cell.dimension: {
                "mean": cell.mean_score if not math.isnan(cell.mean_score) else None,
                "std_dev": cell.std_dev if not math.isnan(cell.std_dev) else None,
                "min": cell.min_score if not math.isnan(cell.min_score) else None,
                "max": cell.max_score if not math.isnan(cell.max_score) else None,
                "spread": cell.spread,
                "scores_by_judge": {
                    k: v if not math.isnan(v) else None
                    for k, v in cell.scores_by_judge.items()
                },
            }
            for cell in matrix.cells
        },
        "pairwise_mad": {
            k: v if not math.isnan(v) else None
            for k, v in matrix.pairwise_mad.items()
        },
        "hotspots": [
            {
                "dimension": h.dimension,
                "trust_level": h.trust_level,
                "std_dev": h.std_dev,
                "spread": h.spread,
            }
            for h in hotspots
        ],
        "total_judging_cost_usd": sum(
            r.cost_usd for pr in all_judge_results
            for r in pr if r.error is None
        ),
    }

    results_path = output_dir / "exp_5_02_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_5_02_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
