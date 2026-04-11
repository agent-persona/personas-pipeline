"""Experiment 6.16: Vocabulary uniqueness.

Hypothesis: The synthesis pipeline produces personas with linguistically
distinct vocabulary — measured by Jensen-Shannon divergence between every
pair of vocabulary[] lists in a tenant. Higher JSD means more diverse
language use across the persona set.

This experiment is purely evaluative: there is no code change to the pipeline.
We run the default pipeline, collect all personas, and measure whether their
vocabulary distributions are meaningfully different.

Metrics:
  - Mean pairwise JS divergence (primary)
  - Min / Max pairwise JSD
  - Vocabulary Jaccard overlap
  - Standard dimensions: groundedness, schema validity, cost

Usage:
    python scripts/experiment_6_16.py
"""

from __future__ import annotations

import asyncio
import json
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
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

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
from synthesis.models.persona import PersonaV1  # noqa: E402
from evaluation.metrics import schema_validity, groundedness_rate, cost_per_persona  # noqa: E402
from vocab_divergence import (  # noqa: E402
    mean_pairwise_js_divergence,
    pairwise_js_matrix,
    vocab_overlap_stats,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class PersonaResult:
    cluster_id: str
    persona_name: str
    persona_dict: dict
    vocabulary: list[str]
    groundedness_score: float
    schema_valid: bool
    cost_usd: float
    attempts: int
    duration_seconds: float


@dataclass
class ExperimentResults:
    # Primary metric
    mean_pairwise_jsd: float = 0.0
    min_pairwise_jsd: float = 0.0
    max_pairwise_jsd: float = 0.0
    # Supplementary vocab stats
    mean_jaccard_overlap: float = 0.0
    unique_ratio: float = 0.0
    total_unique_terms: int = 0
    terms_shared: int = 0
    terms_unique_to_one: int = 0
    # Per-pair breakdown
    pair_details: list[dict] = field(default_factory=list)
    # Standard dimensions
    mean_groundedness: float = 0.0
    schema_validity_rate: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_persona_usd: float = 0.0
    mean_vocab_size: float = 0.0
    # Per-persona results
    persona_results: list[PersonaResult] = field(default_factory=list)


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


async def synthesize_all(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> list[PersonaResult]:
    results = []
    for i, cluster in enumerate(clusters):
        print(f"  [{i + 1}/{len(clusters)}] synthesizing {cluster.cluster_id}...")
        t0 = time.monotonic()
        try:
            result = await synthesize(cluster, backend)
            persona_dict = result.persona.model_dump(mode="json")
            pr = PersonaResult(
                cluster_id=cluster.cluster_id,
                persona_name=result.persona.name,
                persona_dict=persona_dict,
                vocabulary=persona_dict.get("vocabulary", []),
                groundedness_score=result.groundedness.score,
                schema_valid=True,
                cost_usd=result.total_cost_usd,
                attempts=result.attempts,
                duration_seconds=time.monotonic() - t0,
            )
            results.append(pr)
            print(f"      [OK] {pr.persona_name} — vocab={len(pr.vocabulary)} terms, "
                  f"cost=${pr.cost_usd:.4f}, groundedness={pr.groundedness_score:.2f}")
        except Exception as e:
            print(f"      FAILED: {e}")
            results.append(PersonaResult(
                cluster_id=cluster.cluster_id,
                persona_name="FAILED",
                persona_dict={},
                vocabulary=[],
                groundedness_score=0.0,
                schema_valid=False,
                cost_usd=0.0,
                attempts=0,
                duration_seconds=time.monotonic() - t0,
            ))
    return results


# ── Analysis ──────────────────────────────────────────────────────────

def analyze_results(persona_results: list[PersonaResult]) -> ExperimentResults:
    valid = [pr for pr in persona_results if pr.schema_valid]
    er = ExperimentResults(persona_results=persona_results)

    if len(valid) < 2:
        print("  WARNING: Fewer than 2 valid personas — cannot compute pairwise metrics")
        return er

    vocabularies = [pr.vocabulary for pr in valid]
    names = [pr.persona_name for pr in valid]

    # Primary metric: Jensen-Shannon divergence
    matrix = pairwise_js_matrix(vocabularies, names)
    er.mean_pairwise_jsd = matrix["mean_jsd"]
    er.min_pairwise_jsd = matrix["min_jsd"]
    er.max_pairwise_jsd = matrix["max_jsd"]
    er.pair_details = matrix["pairs"]

    # Supplementary: overlap stats
    overlap = vocab_overlap_stats(vocabularies)
    er.mean_jaccard_overlap = overlap["mean_jaccard"]
    er.unique_ratio = overlap["unique_ratio"]
    er.total_unique_terms = overlap["total_unique_terms"]
    er.terms_shared = overlap["terms_shared_across_personas"]
    er.terms_unique_to_one = overlap["terms_unique_to_one_persona"]

    # Standard dimensions
    er.mean_groundedness = statistics.mean([pr.groundedness_score for pr in valid])
    er.schema_validity_rate = schema_validity(
        [pr.persona_dict for pr in persona_results], PersonaV1
    )
    er.total_cost_usd = sum(pr.cost_usd for pr in valid)
    er.cost_per_persona_usd = cost_per_persona(er.total_cost_usd, len(valid))
    er.mean_vocab_size = statistics.mean([len(pr.vocabulary) for pr in valid])

    return er


# ── Reporting ─────────────────────────────────────────────────────────

def print_report(er: ExperimentResults) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 80)
    p("EXPERIMENT 6.16 — VOCABULARY UNIQUENESS — RESULTS")
    p("=" * 80)

    p(f"\n{'Hypothesis:':<20} Personas have distinct word distributions (measured by JSD)")
    p(f"{'Metric:':<20} Mean pairwise Jensen-Shannon divergence")

    valid = [pr for pr in er.persona_results if pr.schema_valid]
    p(f"\n{'Personas generated:':<30} {len(valid)}")
    p(f"{'Personas failed:':<30} {len(er.persona_results) - len(valid)}")

    p("\n── PRIMARY METRIC ──")
    p(f"  {'Mean pairwise JSD:':<35} {er.mean_pairwise_jsd:.6f}")
    p(f"  {'Min pairwise JSD:':<35} {er.min_pairwise_jsd:.6f}")
    p(f"  {'Max pairwise JSD:':<35} {er.max_pairwise_jsd:.6f}")
    p(f"  {'JSD range:':<35} {er.max_pairwise_jsd - er.min_pairwise_jsd:.6f}")

    p("\n── VOCABULARY OVERLAP ──")
    p(f"  {'Mean Jaccard similarity:':<35} {er.mean_jaccard_overlap:.6f}")
    p(f"  {'Unique term ratio:':<35} {er.unique_ratio:.6f}")
    p(f"  {'Total unique terms:':<35} {er.total_unique_terms}")
    p(f"  {'Terms shared across personas:':<35} {er.terms_shared}")
    p(f"  {'Terms unique to one persona:':<35} {er.terms_unique_to_one}")
    p(f"  {'Mean vocab size per persona:':<35} {er.mean_vocab_size:.1f}")

    p("\n── STANDARD DIMENSIONS ──")
    p(f"  {'Mean groundedness:':<35} {er.mean_groundedness:.4f}")
    p(f"  {'Schema validity rate:':<35} {er.schema_validity_rate:.4f}")
    p(f"  {'Total synthesis cost (USD):':<35} ${er.total_cost_usd:.4f}")
    p(f"  {'Cost per persona (USD):':<35} ${er.cost_per_persona_usd:.4f}")

    # Per-pair breakdown
    if er.pair_details:
        p("\n── PAIRWISE JSD BREAKDOWN ──")
        p(f"  {'Persona A':<25} {'Persona B':<25} {'JSD':>10}")
        p(f"  {'-' * 25} {'-' * 25} {'-' * 10}")
        for pair in sorted(er.pair_details, key=lambda x: x["jsd"], reverse=True):
            p(f"  {pair['persona_a']:<25} {pair['persona_b']:<25} {pair['jsd']:>10.6f}")

    # Per-persona vocab details
    p("\n── PER-PERSONA VOCABULARY ──")
    for pr in valid:
        p(f"\n  {pr.persona_name} (cluster: {pr.cluster_id}):")
        p(f"    Vocab size: {len(pr.vocabulary)}")
        p(f"    Terms: {', '.join(pr.vocabulary[:15])}")
        if len(pr.vocabulary) > 15:
            p(f"    ... and {len(pr.vocabulary) - 15} more")

    # Signal assessment
    p("\n── SIGNAL ASSESSMENT ──")

    # JSD thresholds for signal strength
    # 0.0 = identical, 1.0 = completely different
    # For vocabulary lists of 3-15 terms, empirical ranges:
    #   < 0.3  = weak distinctiveness (too much overlap)
    #   0.3-0.6 = moderate distinctiveness
    #   0.6-0.8 = strong distinctiveness
    #   > 0.8  = very strong (nearly no overlap, possibly too divergent)
    jsd = er.mean_pairwise_jsd
    if jsd >= 0.7:
        strength = "VERY STRONG"
        interpretation = "Personas use highly distinct language. Risk: may be too divergent (check domain relevance)."
    elif jsd >= 0.5:
        strength = "STRONG"
        interpretation = "Personas show clear linguistic differentiation. Good diversity signal."
    elif jsd >= 0.3:
        strength = "MODERATE"
        interpretation = "Some vocabulary overlap exists but personas are distinguishable."
    elif jsd >= 0.15:
        strength = "WEAK"
        interpretation = "Significant vocabulary overlap. Personas may sound too similar."
    else:
        strength = "VERY WEAK"
        interpretation = "Minimal vocabulary differentiation. Personas are linguistically redundant."

    p(f"\n  Mean pairwise JSD: {jsd:.6f}")
    p(f"  Signal strength:   {strength}")
    p(f"  Interpretation:    {interpretation}")

    # Cross-check with Jaccard
    jac = er.mean_jaccard_overlap
    if jac < 0.1 and jsd > 0.6:
        p("  Consistency:       JSD and Jaccard agree — very distinct vocabularies.")
    elif jac > 0.3 and jsd < 0.3:
        p("  Consistency:       JSD and Jaccard agree — significant vocabulary sharing.")
    elif jac > 0.3 and jsd > 0.5:
        p("  Consistency:       DIVERGENCE — High JSD despite overlap. Frequency distributions differ even with shared terms.")
    else:
        p(f"  Consistency:       Jaccard={jac:.3f} is {'low' if jac < 0.2 else 'moderate'}, consistent with JSD signal.")

    # Decision
    p(f"\n  Decision: {'ADOPT' if jsd >= 0.3 else 'INVESTIGATE'}")
    if jsd >= 0.3:
        p("  The pipeline produces personas with sufficiently distinct vocabulary.")
    else:
        p("  Vocabulary overlap is high — consider prompt changes to encourage more distinct language.")

    p("\n" + "=" * 80)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 6.16: Vocabulary uniqueness")
    print("Hypothesis: Personas have distinct word distributions (JSD)")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    print(f"\n  Model: {settings.default_model}")
    print(f"  Tenant: {TENANT_ID}")

    # Stage 1: Ingest + Segment
    print("\n[1/3] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    # Stage 2: Synthesize all personas
    print("\n[2/3] Synthesizing personas...")
    persona_results = await synthesize_all(clusters, backend)

    # Stage 3: Analyze vocab divergence
    print("\n[3/3] Computing vocabulary divergence metrics...")
    experiment_results = analyze_results(persona_results)
    report = print_report(experiment_results)

    # Save outputs
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_results = [pr for pr in persona_results if pr.schema_valid]

    results_data = {
        "experiment": "6.16",
        "title": "Vocabulary uniqueness",
        "hypothesis": "Personas have distinct word distributions measured by Jensen-Shannon divergence",
        "model": settings.default_model,
        "tenant_id": TENANT_ID,
        "primary_metric": {
            "name": "mean_pairwise_jsd",
            "value": experiment_results.mean_pairwise_jsd,
            "min": experiment_results.min_pairwise_jsd,
            "max": experiment_results.max_pairwise_jsd,
        },
        "vocab_overlap": {
            "mean_jaccard": experiment_results.mean_jaccard_overlap,
            "unique_ratio": experiment_results.unique_ratio,
            "total_unique_terms": experiment_results.total_unique_terms,
            "terms_shared": experiment_results.terms_shared,
            "terms_unique_to_one": experiment_results.terms_unique_to_one,
        },
        "standard_dimensions": {
            "mean_groundedness": experiment_results.mean_groundedness,
            "schema_validity_rate": experiment_results.schema_validity_rate,
            "total_cost_usd": experiment_results.total_cost_usd,
            "cost_per_persona_usd": experiment_results.cost_per_persona_usd,
        },
        "pairwise_breakdown": experiment_results.pair_details,
        "per_persona": [
            {
                "cluster_id": pr.cluster_id,
                "persona_name": pr.persona_name,
                "vocabulary": pr.vocabulary,
                "vocab_size": len(pr.vocabulary),
                "groundedness_score": pr.groundedness_score,
                "schema_valid": pr.schema_valid,
                "cost_usd": pr.cost_usd,
                "attempts": pr.attempts,
                "duration_seconds": round(pr.duration_seconds, 2),
            }
            for pr in persona_results
        ],
    }

    results_path = output_dir / "exp_6_16_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_6_16_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
