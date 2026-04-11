"""Experiment 5.12: Judge prompt sensitivity."""

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
from evaluation.judges import JudgeBackend  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from evals.judge_prompt_sensitivity import (  # noqa: E402
    VARIANT_SPECS,
    results_to_dict,
    score_variant,
    summarize_scores,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-5.12-judge-prompt-sensitivity"
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


async def generate_personas(backend: AnthropicBackend, repeats: int = 3) -> list[tuple[str, dict]]:
    personas: list[tuple[str, dict]] = []
    clusters = get_clusters()
    for repeat in range(repeats):
        for cluster_index, cluster in enumerate(clusters):
            result = None
            for _ in range(3):
                try:
                    result = await synthesize(cluster, backend, max_retries=4)
                    break
                except Exception:
                    continue
            if result is None:
                raise RuntimeError(
                    f"failed to synthesize cluster {cluster.cluster_id} on repeat {repeat + 1}"
                )
            persona_id = f"run_{repeat + 1}_cluster_{cluster_index + 1}"
            personas.append((persona_id, result.persona.model_dump(mode="json")))
    return personas


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    variance_lines = [
        f"- `{persona_id}`: {variance:.3f}"
        for persona_id, variance in summary["per_persona_overall_variance"].items()
    ]
    cv_lines = [
        f"- `{dim}`: {cv:.3f}"
        for dim, cv in summary["per_dimension_cv"].items()
    ]
    variant_lines = [
        f"- `{variant}`: mean overall `{score:.2f}`"
        for variant, score in summary["variant_mean_overall"].items()
    ]
    return "\n".join(
        [
            "# Experiment 5.12: Judge Prompt Sensitivity",
            "",
            "## Hypothesis",
            "Minor prompt rewording produces meaningful score shifts, exposing judge fragility.",
            "",
            "## Method",
            f"1. Generated `{summary['n_personas']}` personas from repeated golden-tenant synthesis.",
            f"2. Scored each persona with `{summary['n_variants']}` rubric prompt variants.",
            "3. Computed per-persona overall variance and per-dimension coefficient of variation.",
            "",
            f"- Synthesis model: `{results_data['synthesis_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            "",
            "## Per-Persona Overall Variance",
            *variance_lines,
            "",
            "## Per-Dimension Coefficient of Variation",
            *cv_lines,
            "",
            "## Variant Mean Overall Scores",
            *variant_lines,
            "",
            "## Sensitivity Readout",
            f"- Most sensitive dimension: `{summary['most_sensitive_dimension']}`",
            f"- Least sensitive dimension: `{summary['least_sensitive_dimension']}`",
            "",
            "## Decision",
            "TBD after reviewing score spread and prompt-fragility signal.",
            "",
            "## Caveat",
            "Small persona set from one stub tenant; score variance is directional, not definitive.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 5.12: Judge prompt sensitivity")
    print(f"Variants: {len(VARIANT_SPECS)}")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)
    judge_model = "claude-sonnet-4-20250514"
    judge_backend = JudgeBackend(client=client, model=judge_model)

    print("\n[1/3] Generating personas...")
    t0 = time.monotonic()
    personas = await generate_personas(synth_backend, repeats=3)
    print(f"      generated={len(personas)}")

    print("\n[2/3] Scoring prompt variants...")
    results = []
    for persona_id, persona in personas:
        for variant in VARIANT_SPECS:
            result = await score_variant(judge_backend, persona_id, persona, variant)
            results.append(result)

    print("\n[3/3] Writing artifacts...")
    summary = summarize_scores(results)
    results_data = results_to_dict(results, summary)
    results_data.update(
        {
            "experiment": "5.12",
            "title": "Judge prompt sensitivity",
            "synthesis_model": settings.default_model,
            "judge_model": judge_model,
            "duration_seconds": time.monotonic() - t0,
        }
    )
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        "      most_sensitive="
        f"{summary.most_sensitive_dimension} "
        f"least_sensitive={summary.least_sensitive_dimension}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
