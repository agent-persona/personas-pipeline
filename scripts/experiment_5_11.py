"""Experiment 5.11: Reference-based vs reference-free judging.

Hypothesis: Providing a proxy reference persona as a calibration anchor
alongside the standard rubric will reduce score variance and improve
inter-rater consistency, but may introduce anchoring bias (scores
clustering around the reference quality level).

Design:
  1. Generate personas from golden tenant (multiple synthesis repeats
     to get adequate sample size)
  2. Score each persona twice:
     - FREE mode: standard rubric, no reference
     - REFERENCE mode: rubric + proxy reference as calibration anchor
  3. Compare score distributions, variance, rank correlation, anchoring

Usage:
    python scripts/experiment_5_11.py
"""

from __future__ import annotations

import asyncio
import json
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
load_dotenv(REPO_ROOT / ".env")  # fallback: repo-root .env

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from evaluation.judges import (  # noqa: E402
    JudgeBackend,
    LLMJudge,
    _JUDGE_SYSTEM_PROMPT,
    _parse_judge_response,
)
from reference_judging import (  # noqa: E402
    build_free_prompt,
    build_reference_prompt,
    compare_modes,
    PROXY_EXPECTED_OVERALL,
)

# ── Config ────────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
SYNTHESIS_REPEATS = 3  # multiple repeats to get adequate sample size
JUDGE_MODEL = "claude-sonnet-4-20250514"

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.11-reference-vs-free-judging"


# ── Pipeline helpers ──────────────────────────────────────────────────

def get_clusters() -> list[ClusterData]:
    """Run ingest + segmentation for golden tenant."""
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


@dataclass
class PersonaSample:
    """A synthesized persona with metadata."""
    persona_dict: dict
    cluster_id: str
    repeat: int
    name: str
    cost_usd: float


async def generate_personas(clusters: list[ClusterData], repeats: int) -> list[PersonaSample]:
    """Generate multiple personas per cluster via repeated synthesis."""
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    samples: list[PersonaSample] = []
    for rep in range(repeats):
        for cluster in clusters:
            print(f"  Synthesizing repeat {rep+1}/{repeats}, cluster {cluster.cluster_id}...")
            try:
                result = await synthesize(cluster, backend)
                persona_dict = result.persona.model_dump(mode="json")
                samples.append(PersonaSample(
                    persona_dict=persona_dict,
                    cluster_id=cluster.cluster_id,
                    repeat=rep,
                    name=result.persona.name,
                    cost_usd=result.total_cost_usd,
                ))
            except Exception as e:
                print(f"    FAILED: {e}")
    return samples


# ── Judging ───────────────────────────────────────────────────────────

@dataclass
class JudgeRun:
    """Scores from one persona under one mode."""
    persona_name: str
    cluster_id: str
    repeat: int
    mode: str  # "free" or "reference"
    overall: float
    dimensions: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


async def judge_persona(
    persona_dict: dict,
    mode: str,
    judge_backend: JudgeBackend,
    judge_model: str,
) -> tuple[float, dict[str, float], str]:
    """Score a persona in the specified mode."""
    if mode == "free":
        prompt = build_free_prompt(persona_dict)
    else:
        prompt = build_reference_prompt(persona_dict)

    response = await judge_backend.score(
        system=_JUDGE_SYSTEM_PROMPT,
        prompt=prompt,
    )
    result = _parse_judge_response(response, judge_model)
    return result.overall, result.dimensions, result.rationale


async def run_judging(
    samples: list[PersonaSample],
    judge_backend: JudgeBackend,
    judge_model: str,
) -> list[JudgeRun]:
    """Score all samples in both modes."""
    runs: list[JudgeRun] = []

    for i, sample in enumerate(samples):
        print(f"  Judging {i+1}/{len(samples)}: {sample.name}")

        # Free mode
        overall_f, dims_f, rat_f = await judge_persona(
            sample.persona_dict, "free", judge_backend, judge_model
        )
        runs.append(JudgeRun(
            persona_name=sample.name,
            cluster_id=sample.cluster_id,
            repeat=sample.repeat,
            mode="free",
            overall=overall_f,
            dimensions=dims_f,
            rationale=rat_f,
        ))

        # Reference mode
        overall_r, dims_r, rat_r = await judge_persona(
            sample.persona_dict, "reference", judge_backend, judge_model
        )
        runs.append(JudgeRun(
            persona_name=sample.name,
            cluster_id=sample.cluster_id,
            repeat=sample.repeat,
            mode="reference",
            overall=overall_r,
            dimensions=dims_r,
            rationale=rat_r,
        ))

    return runs


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Step 1: Generate personas
    print("\n[1/3] Generating personas from golden tenant...")
    clusters = get_clusters()
    print(f"  Found {len(clusters)} clusters")
    samples = await generate_personas(clusters, SYNTHESIS_REPEATS)
    print(f"  Generated {len(samples)} persona samples")

    if not samples:
        print("ERROR: No personas generated. Cannot run experiment.")
        return

    # Step 2: Judge in both modes
    print("\n[2/3] Judging personas in free and reference modes...")
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    judge_backend = JudgeBackend(client=client, model=JUDGE_MODEL)
    runs = await run_judging(samples, judge_backend, JUDGE_MODEL)

    # Step 3: Analyze
    print("\n[3/3] Analyzing results...")
    free_scores = [r.overall for r in runs if r.mode == "free"]
    ref_scores = [r.overall for r in runs if r.mode == "reference"]

    comparison = compare_modes(free_scores, ref_scores)

    # Build full results
    results = {
        "experiment": "5.11",
        "title": "Reference-based vs reference-free judging",
        "config": {
            "tenant": TENANT_ID,
            "synthesis_repeats": SYNTHESIS_REPEATS,
            "judge_model": JUDGE_MODEL,
            "num_clusters": len(clusters),
            "num_samples": len(samples),
            "proxy_reference_quality": PROXY_EXPECTED_OVERALL,
        },
        "comparison": comparison.to_dict(),
        "per_persona": [
            {
                "persona_name": r.persona_name,
                "cluster_id": r.cluster_id,
                "repeat": r.repeat,
                "mode": r.mode,
                "overall": round(r.overall, 3),
                "dimensions": {k: round(v, 3) for k, v in r.dimensions.items()},
                "rationale": r.rationale,
            }
            for r in runs
        ],
        "synthesis_cost_usd": round(sum(s.cost_usd for s in samples), 4),
        "duration_seconds": round(time.monotonic() - t0, 1),
    }

    # Save
    results_path = OUTPUT_DIR / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 5.11 — REFERENCE-BASED vs REFERENCE-FREE JUDGING")
    print("=" * 70)
    c = comparison
    print(f"  Samples:                  {c.free_stats.n}")
    print(f"  Free mode:                mean={c.free_stats.mean:.3f}, std={c.free_stats.std:.3f}")
    print(f"  Reference mode:           mean={c.ref_stats.mean:.3f}, std={c.ref_stats.std:.3f}")
    print(f"  Variance reduction ratio: {c.variance_reduction_ratio:.3f}")
    print(f"  Mean delta (ref - free):  {c.mean_delta:.3f}")
    print(f"  Spearman rho:             {c.spearman_rho:.3f}" if c.spearman_rho == c.spearman_rho else "  Spearman rho:             N/A (too few samples)")
    print(f"  Anchoring detected:       {c.anchoring_detected}")
    print(f"  Anchoring evidence:       {c.anchoring_evidence}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
