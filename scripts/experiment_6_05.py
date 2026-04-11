"""Experiment 6.05: Stability across reruns."""

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

from evals.stability_reruns import compare_run_to_baseline, comparisons_to_dict  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-6.05-stability-across-reruns"
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


async def synthesize_run(client: AsyncAnthropic, temperature: float, run_index: int) -> dict:
    personas = []
    for cluster in get_clusters():
        result = None
        attempt_plan = [temperature, 0.4, None]
        for attempt_temperature in attempt_plan:
            backend = AnthropicBackend(
                client=client,
                model=settings.default_model,
                temperature=attempt_temperature,
            )
            try:
                result = await synthesize(cluster, backend, max_retries=4)
                break
            except Exception:
                continue
        if result is None:
            raise RuntimeError(f"failed to synthesize cluster {cluster.cluster_id} in run_{run_index}")
        personas.append(result.persona.model_dump(mode="json"))
    return {"run_id": f"run_{run_index}", "personas": personas}


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    comparison_lines = []
    for comparison in summary["comparisons"]:
        comparison_lines.append(
            f"- `{comparison['run_id']}`: mean similarity `{comparison['mean_similarity']:.3f}`, "
            f"archetype recurrence `{comparison['archetype_recurrence_rate']:.1%}`"
        )
    return "\n".join(
        [
            "# Experiment 6.05: Stability Across Reruns",
            "",
            "## Hypothesis",
            "Stable source data should produce consistent persona archetypes across reruns.",
            "",
            "## Method",
            "1. Re-ran the golden tenant pipeline 5 times.",
            "2. Added synthesis stochasticity with `temperature=0.7` while leaving clustering unchanged.",
            "3. Matched each rerun back to run 1 via best overall persona similarity.",
            "4. Measured cross-run similarity on names, summaries, goals, pains, vocabulary, and quotes.",
            "",
            f"- Model: `{results_data['model']}`",
            f"- Temperature: `{results_data['temperature']}`",
            f"- Runs: `{summary['n_runs']}`",
            "",
            "## Baseline-Referenced Comparisons",
            *comparison_lines,
            "",
            "## Aggregate Metrics",
            f"- Mean cross-run similarity: `{summary['mean_cross_run_similarity']:.3f}`",
            f"- Mean archetype recurrence: `{summary['mean_archetype_recurrence_rate']:.1%}`",
            "",
            "## Decision",
            "TBD after reviewing whether recurring archetypes stay recognizably aligned.",
            "",
            "## Caveat",
            "Only 1 tenant and 2 natural clusters. Variation here comes mostly from synthesis stochasticity, not segmentation.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 6.05: Stability across reruns")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    temperature = 0.7
    t0 = time.monotonic()

    print("\n[1/3] Running 5 stochastic reruns...")
    runs = []
    for run_index in range(1, 6):
        run = await synthesize_run(client, temperature=temperature, run_index=run_index)
        runs.append(run)
        names = ", ".join(persona.get("name", "unknown") for persona in run["personas"])
        print(f"      run_{run_index}: {names}")

    print("\n[2/3] Comparing to baseline run...")
    baseline = runs[0]["personas"]
    comparisons = [
        compare_run_to_baseline(baseline, run["personas"], run["run_id"])
        for run in runs[1:]
    ]

    print("\n[3/3] Writing artifacts...")
    mean_similarity = (
        sum(comparison.mean_similarity for comparison in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )
    mean_recurrence = (
        sum(comparison.archetype_recurrence_rate for comparison in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )
    results_data = {
        "experiment": "6.05",
        "title": "Stability across reruns",
        "model": settings.default_model,
        "temperature": temperature,
        "runs": runs,
        "summary": {
            "n_runs": len(runs),
            "comparisons": comparisons_to_dict(comparisons),
            "mean_cross_run_similarity": mean_similarity,
            "mean_archetype_recurrence_rate": mean_recurrence,
        },
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      mean_similarity={mean_similarity:.3f} "
        f"mean_recurrence={mean_recurrence:.1%}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
