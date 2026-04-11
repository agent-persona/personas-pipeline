"""Experiment 5.10: Pairwise vs absolute judging."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from evaluation.judges import JudgeBackend, LLMJudge  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from evals.pairwise_judging import (  # noqa: E402
    AbsoluteScoreRow,
    PairwiseRow,
    judge_pair_bidirectional,
    results_to_dict,
    summarize,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-5.10-pairwise-vs-absolute"
)
JUDGE_MODELS = ("claude-haiku-4-5-20251001", "claude-sonnet-4-20250514")


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


async def generate_personas(backend: AnthropicBackend, repeats: int = 2) -> list[tuple[str, dict]]:
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
    abs_lines = [
        f"- `{dimension}`: `{value:.3f}`"
        for dimension, value in summary["absolute_agreement"].items()
    ]
    pair_lines = [
        f"- `{dimension}`: `{value:.3f}`"
        for dimension, value in summary["pairwise_agreement"].items()
    ]
    return "\n".join(
        [
            "# Experiment 5.10: Pairwise vs Absolute Judging",
            "",
            "## Hypothesis",
            "Pairwise preference judging produces higher inter-judge agreement than absolute 1-5 scoring.",
            "",
            "## Method",
            "1. Generated 4 personas from repeated golden-tenant synthesis.",
            "2. Scored each persona with absolute judging using Haiku and Sonnet.",
            "3. Ran bidirectional pairwise judging on every persona pair with the same two models.",
            "4. Converted both modes into rank orderings and compared cross-model Spearman agreement.",
            "",
            "## Absolute-Mode Agreement",
            *abs_lines,
            f"- Mean agreement: `{summary['mean_absolute_agreement']:.3f}`",
            "",
            "## Pairwise-Mode Agreement",
            *pair_lines,
            f"- Mean agreement: `{summary['mean_pairwise_agreement']:.3f}`",
            "",
            "## Distribution Tightness",
            f"- Absolute overall stddev: `{summary['distribution_tightness']['absolute_overall_stddev']:.3f}`",
            f"- Pairwise rank stddev: `{summary['distribution_tightness']['pairwise_rank_stddev']:.3f}`",
            "",
            "## Decision",
            "TBD after reviewing whether pairwise agreement materially exceeds absolute agreement.",
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 4 personas, 2 judge models.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 5.10: Pairwise vs absolute judging")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/4] Generating personas...")
    t0 = time.monotonic()
    personas = await generate_personas(synth_backend, repeats=2)
    print(f"      generated={len(personas)}")

    print("\n[2/4] Absolute scoring...")
    absolute_rows: list[AbsoluteScoreRow] = []
    for judge_model in JUDGE_MODELS:
        judge = LLMJudge(
            backend=JudgeBackend(client=client, model=judge_model),
            model=judge_model,
            calibration="few_shot",
        )
        for persona_id, persona in personas:
            score = await judge.score_persona(persona)
            absolute_rows.append(
                AbsoluteScoreRow(
                    judge_model=judge_model,
                    persona_id=persona_id,
                    overall=score.overall,
                    dimensions=score.dimensions,
                )
            )

    print("\n[3/4] Pairwise scoring...")
    pairwise_rows: list[PairwiseRow] = []
    for judge_model in JUDGE_MODELS:
        backend = JudgeBackend(client=client, model=judge_model)
        for (persona_id_a, persona_a), (persona_id_b, persona_b) in combinations(personas, 2):
            winners, rationale = await judge_pair_bidirectional(backend, persona_a, persona_b)
            pairwise_rows.append(
                PairwiseRow(
                    judge_model=judge_model,
                    persona_a=persona_id_a,
                    persona_b=persona_id_b,
                    winners=winners,
                    rationale=rationale,
                )
            )

    print("\n[4/4] Writing artifacts...")
    summary = summarize(absolute_rows, pairwise_rows)
    results_data = results_to_dict(absolute_rows, pairwise_rows, summary)
    results_data.update(
        {
            "experiment": "5.10",
            "title": "Pairwise vs absolute judging",
            "synthesis_model": settings.default_model,
            "judge_models": list(JUDGE_MODELS),
            "duration_seconds": time.monotonic() - t0,
        }
    )
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      mean_abs={summary['mean_absolute_agreement']:.3f} "
        f"mean_pair={summary['mean_pairwise_agreement']:.3f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
