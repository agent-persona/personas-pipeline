"""Experiment 2.10: Tree-of-thoughts synthesis."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
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

from evals.judge_helper_2_10 import JudgeBackend, LLMJudge  # noqa: E402
from evals.tree_of_thoughts import result_to_dict, run_tree_of_thoughts  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-2.10-tree-of-thoughts"
)


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
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


def generate_findings(results_data: dict) -> str:
    cluster_lines = []
    for cluster_id, result in results_data["clusters"].items():
        cluster_lines.append(
            f"- `{cluster_id}`: control `{result['control']['overall_score']:.2f}` -> "
            f"refined `{result['refined']['overall_score']:.2f}` "
            f"(delta `{result['convergence_delta']:+.2f}`)"
        )
    return "\n".join(
        [
            "# Experiment 2.10: Tree-of-Thoughts",
            "",
            "## Hypothesis",
            "Generate -> score -> prune -> refine yields better personas than single-shot control at a reasonable cost multiplier.",
            "",
            "## Method",
            "1. Ran a single-shot control on each golden-tenant cluster.",
            "2. Generated 3 stochastic candidates per cluster.",
            "3. Judged and pruned the lowest-scoring candidate.",
            "4. Refined the top candidate using judge feedback as extra context.",
            "",
            f"- Synthesis model: `{results_data['synthesis_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            "",
            "## Cluster Outcomes",
            *cluster_lines,
            "",
            "## Aggregate",
            f"- Mean control score: `{results_data['summary']['mean_control_score']:.2f}`",
            f"- Mean refined score: `{results_data['summary']['mean_refined_score']:.2f}`",
            f"- Mean convergence delta: `{results_data['summary']['mean_convergence_delta']:+.2f}`",
            f"- Mean cost multiplier vs control: `{results_data['summary']['mean_cost_multiplier']:.2f}x`",
            "",
            "## Decision",
            "TBD after reviewing score lift against cost.",
            "",
            "## Caveat",
            "Small sample: 1 tenant, 2 clusters. Diversity comes from synthesis temperature, not extra data.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 2.10: Tree-of-thoughts")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    control_backend = AnthropicBackend(client=client, model=settings.default_model)
    candidate_backend = AnthropicBackend(
        client=client,
        model=settings.default_model,
        temperature=0.7,
    )
    judge_model = "claude-sonnet-4-20250514"
    judge = LLMJudge(
        backend=JudgeBackend(client=client, model=judge_model),
        model=judge_model,
    )

    print("\n[1/3] Running tree-of-thoughts per cluster...")
    t0 = time.monotonic()
    cluster_results = {}
    for cluster in get_clusters():
        result = await run_tree_of_thoughts(
            cluster=cluster,
            control_backend=control_backend,
            candidate_backend=candidate_backend,
            judge=judge,
            synthesize_fn=synthesize,
        )
        cluster_results[cluster.cluster_id] = result_to_dict(result)
        print(
            f"      {cluster.cluster_id}: "
            f"{result.control.overall_score:.2f} -> {result.refined.overall_score:.2f}"
        )

    print("\n[2/3] Aggregating metrics...")
    control_scores = [result["control"]["overall_score"] for result in cluster_results.values()]
    refined_scores = [result["refined"]["overall_score"] for result in cluster_results.values()]
    convergence = [result["convergence_delta"] for result in cluster_results.values()]
    cost_multipliers = [
        result["total_candidate_cost_usd"] / result["control"]["synthesis_cost_usd"]
        for result in cluster_results.values()
        if result["control"]["synthesis_cost_usd"]
    ]
    summary = {
        "mean_control_score": sum(control_scores) / len(control_scores),
        "mean_refined_score": sum(refined_scores) / len(refined_scores),
        "mean_convergence_delta": sum(convergence) / len(convergence),
        "mean_cost_multiplier": sum(cost_multipliers) / len(cost_multipliers),
    }

    print("\n[3/3] Writing artifacts...")
    results_data = {
        "experiment": "2.10",
        "title": "Tree-of-thoughts",
        "synthesis_model": settings.default_model,
        "judge_model": judge_model,
        "clusters": cluster_results,
        "summary": summary,
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      mean_control={summary['mean_control_score']:.2f} "
        f"mean_refined={summary['mean_refined_score']:.2f} "
        f"cost_multiplier={summary['mean_cost_multiplier']:.2f}x"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
