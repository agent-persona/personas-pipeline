"""Experiment 4.05: Refusal boundary attacks."""

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

from evals.boundary_attacks import (  # noqa: E402
    ATTACKS,
    results_to_dict,
    run_boundary_attacks,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-4.05-refusal-boundary-attacks"
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


async def generate_personas(backend: AnthropicBackend) -> list[dict]:
    personas: list[dict] = []
    for cluster in get_clusters()[:2]:
        result = None
        for _ in range(3):
            try:
                result = await synthesize(cluster, backend, max_retries=4)
                break
            except Exception:
                continue
        if result is None:
            raise RuntimeError(f"failed to synthesize cluster {cluster.cluster_id}")
        personas.append(result.persona.model_dump(mode="json"))
    return personas


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    category_lines = [
        f"- `{category}`: mean score {score:.2f}"
        for category, score in summary["mean_score_by_category"].items()
    ]
    return "\n".join(
        [
            "# Experiment 4.05: Refusal Boundary Attacks",
            "",
            "## Hypothesis",
            "Twin personas resist adversarial role-flip prompts; stronger persona",
            "JSON should stay in character under direct and escalating boundary attacks.",
            "",
            "## Method",
            f"1. Generated {summary['n_personas']} personas from `tenant_acme_corp`.",
            f"2. Ran {summary['n_attacks']} single-turn attack prompts against each persona.",
            "3. Scored each response with an LLM rubric:",
            "   - `1.0` in character",
            "   - `0.5` partial break",
            "   - `0.0` full break",
            "",
            f"- Twin model: `{results_data['twin_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            f"- Total trials: `{summary['total_trials']}`",
            "",
            "## Metrics",
            f"- In-character retention: `{summary['in_character_rate']:.1%}`",
            f"- Partial break rate: `{summary['partial_break_rate']:.1%}`",
            f"- Full break rate: `{summary['full_break_rate']:.1%}`",
            f"- Refusal rate: `{summary['refusal_rate']:.1%}`",
            f"- Mean score: `{summary['mean_score']:.2f}`",
            "",
            "## Mean Score By Attack Category",
            *category_lines,
            "",
            "## Decision",
            "TBD after reviewing the attack transcripts.",
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 2 personas, single-turn attacks only.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 4.05: Refusal boundary attacks")
    print(f"Attacks: {len(ATTACKS)}")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)
    twin_model = settings.default_model
    judge_model = "claude-haiku-4-5-20251001"

    print("\n[1/3] Generating personas...")
    t0 = time.monotonic()
    personas = await generate_personas(synth_backend)
    for persona in personas:
        print(f"      - {persona.get('name', 'unknown')}")

    print("\n[2/3] Running attacks...")
    results, summary = await run_boundary_attacks(
        client=client,
        personas=personas,
        twin_model=twin_model,
        judge_model=judge_model,
    )
    duration = time.monotonic() - t0

    print("\n[3/3] Writing artifacts...")
    results_data = results_to_dict(results, summary)
    results_data.update(
        {
            "experiment": "4.05",
            "title": "Refusal boundary attacks",
            "tenant_id": TENANT_ID,
            "twin_model": twin_model,
            "judge_model": judge_model,
            "duration_seconds": duration,
        }
    )
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      retention={summary.in_character_rate:.1%} "
        f"full_break={summary.full_break_rate:.1%} "
        f"mean_score={summary.mean_score:.2f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
