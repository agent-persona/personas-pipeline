"""Experiment 5.03: Self-preference bias.

Hypothesis: LLM judges score outputs from their own model class higher than
outputs from other models, even when quality is comparable. This "diagonal
bias" in the synthesizer x judge matrix undermines trust in single-judge eval.

Setup:
  1. Synthesize personas using multiple models (Haiku, Sonnet, Opus).
  2. Score each persona with each judge model.
  3. Build the synthesizer x judge preference matrix.
  4. Compute preference deltas (diagonal vs off-diagonal).
  5. If bias exists, derive a debiasing prior.

Metrics:
  - Preference delta per judge (positive = self-preference)
  - Debiasing prior (correction offset)
  - Per-dimension breakdown of bias

Usage:
    python scripts/experiment_5_03.py
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
import time
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

from self_preference import (  # noqa: E402
    DIMENSIONS,
    DebiasingPrior,
    PreferenceDelta,
    PreferenceMatrix,
    ScoreResult,
    build_preference_matrix,
    compute_debiasing_prior,
    compute_preference_deltas,
    score_persona_with_judge,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

# Models used as both synthesizers AND judges
MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}


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


async def synthesize_with_model(
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
) -> dict | None:
    """Synthesize a persona using a specific model, return dict or None."""
    backend = AnthropicBackend(client=client, model=model)
    try:
        result = await synthesize(cluster, backend)
        persona_dict = result.persona.model_dump(mode="json")
        persona_dict["_meta"] = {
            "cluster_id": cluster.cluster_id,
            "synthesizer_model": model,
            "groundedness": result.groundedness.score,
            "cost_usd": result.total_cost_usd,
            "attempts": result.attempts,
        }
        return persona_dict
    except Exception as e:
        print(f"      FAILED: {e}")
        return None


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(
    all_scores: list[ScoreResult],
    matrix: PreferenceMatrix,
    deltas: list[PreferenceDelta],
    prior: DebiasingPrior,
    synthesis_costs: dict[str, float],
) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    synth_names = matrix.synthesizer_models
    judge_names = matrix.judge_models

    def short(model: str) -> str:
        if "opus" in model:
            return "opus"
        if "sonnet" in model:
            return "sonnet"
        if "haiku" in model:
            return "haiku"
        return model[:12]

    p("\n" + "=" * 100)
    p("EXPERIMENT 5.03 — SELF-PREFERENCE BIAS — RESULTS")
    p("=" * 100)

    # ── Preference matrix (overall) ──
    p("\n── PREFERENCE MATRIX (overall scores) ──")
    p(f"  Personas per synthesizer: {matrix.n_personas_per_synth}")
    col_label = "Synth / Judge"
    header = f"  {col_label:<20}"
    for j in judge_names:
        header += f"{short(j):>12}"
    p(header)
    p("  " + "-" * (20 + 12 * len(judge_names)))

    for s in synth_names:
        row = f"  {short(s):<20}"
        for j in judge_names:
            val = matrix.matrix[s][j]
            marker = ""
            if short(s) == short(j):
                marker = " *"  # diagonal
            if math.isnan(val):
                row += f"{'N/A':>12}"
            else:
                row += f"{val:>10.3f}{marker}"
        p(row)

    p("  (* = diagonal: judge scoring own model's output)")

    # ── Per-dimension matrix ──
    for dim in DIMENSIONS:
        p(f"\n── {dim.upper()} scores ──")
        header = f"  {col_label:<20}"
        for j in judge_names:
            header += f"{short(j):>12}"
        p(header)
        p("  " + "-" * (20 + 12 * len(judge_names)))

        for s in synth_names:
            row = f"  {short(s):<20}"
            for j in judge_names:
                val = matrix.per_dim[s][j].get(dim, float("nan"))
                marker = ""
                if short(s) == short(j):
                    marker = " *"
                if math.isnan(val):
                    row += f"{'N/A':>12}"
                else:
                    row += f"{val:>10.3f}{marker}"
            p(row)

    # ── Preference deltas ──
    p("\n── PREFERENCE DELTAS (positive = self-preference bias) ──")
    header = f"  {'Judge':<15}{'Diagonal':>12}{'Off-diag':>12}{'Delta':>12}{'Bias?':>10}"
    p(header)
    p("  " + "-" * 61)

    for d in deltas:
        bias_flag = "YES" if d.delta > 0.03 else ("maybe" if d.delta > 0.01 else "no")
        p(f"  {short(d.judge_model):<15}"
          f"{d.diagonal_score:>12.3f}"
          f"{d.off_diagonal_mean:>12.3f}"
          f"{d.delta:>+12.4f}"
          f"{bias_flag:>10}")

    # Per-dim deltas
    p("\n── PER-DIMENSION DELTAS ──")
    header = f"  {'Judge':<15}"
    for dim in DIMENSIONS:
        header += f"{dim[:10]:>12}"
    p(header)
    p("  " + "-" * (15 + 12 * len(DIMENSIONS)))

    for d in deltas:
        row = f"  {short(d.judge_model):<15}"
        for dim in DIMENSIONS:
            val = d.per_dim_delta.get(dim, float("nan"))
            if math.isnan(val):
                row += f"{'N/A':>12}"
            else:
                row += f"{val:>+12.4f}"
        p(row)

    # ── Debiasing prior ──
    p("\n── DEBIASING PRIOR ──")
    p(f"  Overall bias magnitude: {prior.overall_bias:+.4f}")
    p(f"  Bias detected (threshold > 0.03): {'YES' if prior.bias_detected else 'NO'}")
    if prior.judge_offsets:
        p("\n  Per-judge correction offsets (subtract from diagonal scores):")
        for judge, offset in prior.judge_offsets.items():
            direction = "inflate" if offset > 0 else "deflate"
            p(f"    {short(judge)}: {offset:+.4f} "
              f"({direction}s own output by {abs(offset):.3f})")

    # ── Signal assessment ──
    p("\n── SIGNAL ASSESSMENT ──")

    valid_deltas = [d.delta for d in deltas if not math.isnan(d.delta)]
    if not valid_deltas:
        strength = "INCONCLUSIVE"
        detail = "Not enough data to assess self-preference bias."
    else:
        pos_biased = sum(1 for d in valid_deltas if d > 0.03)
        any_biased = sum(1 for d in valid_deltas if d > 0.01)
        max_delta = max(abs(d) for d in valid_deltas)
        mean_delta = statistics.mean(valid_deltas)

        if pos_biased >= 2 and max_delta > 0.05:
            strength = "STRONG FINDING"
            detail = (
                f"Clear self-preference bias: {pos_biased}/{len(valid_deltas)} "
                f"judges inflate their own outputs (max delta={max_delta:.3f}). "
                f"Debiasing prior should be applied when using single-model judge."
            )
        elif any_biased >= 1 and mean_delta > 0.02:
            strength = "MODERATE FINDING"
            detail = (
                f"Some self-preference detected: mean delta={mean_delta:+.4f}. "
                f"Multi-judge averaging recommended over single-model scoring."
            )
        elif mean_delta > 0.005:
            strength = "WEAK FINDING"
            detail = (
                f"Marginal self-preference: mean delta={mean_delta:+.4f}. "
                f"Effect is small and may not be practically significant."
            )
        else:
            strength = "NULL RESULT"
            detail = (
                f"No self-preference detected: mean delta={mean_delta:+.4f}. "
                f"Judges appear model-agnostic in their scoring."
            )

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")

    # Cost summary
    total_synth = sum(synthesis_costs.values())
    total_judge = sum(s.cost_usd for s in all_scores if s.error is None)
    p(f"\n  Synthesis cost: ${total_synth:.4f}")
    p(f"  Judging cost:   ${total_judge:.4f}")
    p(f"  Total cost:     ${total_synth + total_judge:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 5.03: Self-preference bias")
    print("Hypothesis: Judges score their own model's outputs higher")
    print("  than outputs from other models (diagonal bias)")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Step 1: Ingest + segment
    print("\n[1/5] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    # Step 2: Synthesize with each model
    print("\n[2/5] Synthesizing personas with each model...")
    synth_models = list(MODELS.values())
    personas_by_synth: dict[str, list[dict]] = {m: [] for m in synth_models}
    synthesis_costs: dict[str, float] = {m: 0.0 for m in synth_models}

    for model_name, model_id in MODELS.items():
        print(f"\n  ── Synthesizer: {model_name} ({model_id}) ──")
        for cluster in clusters:
            print(f"    Cluster {cluster.cluster_id}...", end=" ", flush=True)
            persona = await synthesize_with_model(cluster, client, model_id)
            if persona:
                personas_by_synth[model_id].append(persona)
                cost = persona.get("_meta", {}).get("cost_usd", 0)
                synthesis_costs[model_id] += cost
                print(f"{persona.get('name', '?')} "
                      f"(cost=${cost:.4f})")
            else:
                print("FAILED")

    # Verify we got personas from each synthesizer
    for model_id, personas in personas_by_synth.items():
        if not personas:
            print(f"WARNING: No personas from {model_id}")

    # Step 3: Score each persona with each judge
    print("\n[3/5] Scoring all personas with all judges...")
    judge_models = list(MODELS.values())
    all_scores: list[ScoreResult] = []

    for synth_model, personas in personas_by_synth.items():
        synth_name = next(k for k, v in MODELS.items() if v == synth_model)
        for i, persona in enumerate(personas):
            name = persona.get("name", f"Persona {i+1}")
            print(f"\n  {synth_name}-synthesized: {name}")

            judge_input = {k: v for k, v in persona.items() if not k.startswith("_")}

            # Score with all judges concurrently
            tasks = [
                score_persona_with_judge(client, j, judge_input, synth_model)
                for j in judge_models
            ]
            results = await asyncio.gather(*tasks)

            for r in results:
                all_scores.append(r)
                j_name = next(k for k, v in MODELS.items() if v == r.judge_model)
                if r.error:
                    print(f"    judge={j_name}: FAILED ({r.error[:60]})")
                else:
                    print(f"    judge={j_name}: overall={r.overall:.3f} "
                          f"cost=${r.cost_usd:.4f}")

    # Step 4: Build matrix and compute deltas
    print("\n[4/5] Computing preference matrix...")
    matrix = build_preference_matrix(all_scores, synth_models, judge_models)
    deltas = compute_preference_deltas(matrix)
    prior = compute_debiasing_prior(deltas)

    # Step 5: Report
    print("\n[5/5] Generating report...")
    report = print_results(all_scores, matrix, deltas, prior, synthesis_costs)

    # Save outputs
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe_val(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    results_data = {
        "experiment": "5.03",
        "title": "Self-preference bias",
        "hypothesis": "Judges score their own model's outputs higher (diagonal bias)",
        "models": MODELS,
        "n_clusters": len(clusters),
        "personas_per_synth": {
            next(k for k, v in MODELS.items() if v == m): len(p)
            for m, p in personas_by_synth.items()
        },
        "synthesis_costs": {
            next(k for k, v in MODELS.items() if v == m): c
            for m, c in synthesis_costs.items()
        },
        "preference_matrix": {
            next(k for k, v in MODELS.items() if v == s): {
                next(k for k, v in MODELS.items() if v == j): safe_val(matrix.matrix[s][j])
                for j in judge_models
            }
            for s in synth_models
        },
        "per_dimension_matrix": {
            dim: {
                next(k for k, v in MODELS.items() if v == s): {
                    next(k for k, v in MODELS.items() if v == j): safe_val(
                        matrix.per_dim[s][j].get(dim, float("nan"))
                    )
                    for j in judge_models
                }
                for s in synth_models
            }
            for dim in DIMENSIONS
        },
        "preference_deltas": [
            {
                "judge": next(k for k, v in MODELS.items() if v == d.judge_model),
                "diagonal_score": safe_val(d.diagonal_score),
                "off_diagonal_mean": safe_val(d.off_diagonal_mean),
                "delta": safe_val(d.delta),
                "per_dim_delta": {
                    dim: safe_val(v) for dim, v in d.per_dim_delta.items()
                },
            }
            for d in deltas
        ],
        "debiasing_prior": {
            "overall_bias": safe_val(prior.overall_bias),
            "bias_detected": prior.bias_detected,
            "judge_offsets": {
                next(k for k, v in MODELS.items() if v == j): safe_val(o)
                for j, o in prior.judge_offsets.items()
            },
        },
        "all_scores": [
            {
                "synthesizer": next(
                    (k for k, v in MODELS.items() if v == s.synthesizer_model),
                    s.synthesizer_model,
                ),
                "judge": next(
                    (k for k, v in MODELS.items() if v == s.judge_model),
                    s.judge_model,
                ),
                "overall": safe_val(s.overall),
                "dimensions": {k: safe_val(v) for k, v in s.dimensions.items()},
                "rationale": s.rationale,
                "cost_usd": s.cost_usd,
                "error": s.error,
            }
            for s in all_scores
        ],
        "total_cost_usd": (
            sum(synthesis_costs.values())
            + sum(s.cost_usd for s in all_scores if s.error is None)
        ),
    }

    results_path = output_dir / "exp_5_03_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_5_03_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
