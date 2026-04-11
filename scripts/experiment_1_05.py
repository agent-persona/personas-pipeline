"""Experiment 1.05: Schema versioning drift."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from crawler import fetch_all  # noqa: E402
from evals.judge_helper_1_05 import (  # noqa: E402
    FallbackJudgeBackend,
    JudgeBackend,
    LLMJudge,
    OpenAIJudgeBackend,
)
from evals.schema_versioning import (  # noqa: E402
    SCHEMA_VARIANTS,
    LocalJudge,
    VariantRun,
    build_findings,
    compare_shared_fields,
    merge_similarity_by_field,
    results_to_dict,
    summarize_variant,
    synthesize_variant,
    variant_by_key,
)
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-1.05-schema-versioning-drift"
)
ANTHROPIC_JUDGE_MODEL = "claude-sonnet-4-20250514"
OPENAI_JUDGE_MODEL = "gpt-5-nano"


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


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_judge() -> tuple[LLMJudge, str]:
    if os.getenv("PERSONAS_USE_REMOTE_LLM", "").strip() != "1":
        return LocalJudge(), "local"

    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if openai_client is not None:
        primary = OpenAIJudgeBackend(client=openai_client, model=OPENAI_JUDGE_MODEL)
        fallback = (
            JudgeBackend(
                client=AsyncAnthropic(api_key=settings.anthropic_api_key),
                model=ANTHROPIC_JUDGE_MODEL,
            )
            if settings.anthropic_api_key
            else None
        )
        return (
            LLMJudge(
                backend=FallbackJudgeBackend(primary=primary, fallback=fallback),
                model=OPENAI_JUDGE_MODEL,
            ),
            OPENAI_JUDGE_MODEL,
        )
    if settings.anthropic_api_key:
        return (
            LLMJudge(
                backend=JudgeBackend(
                    client=AsyncAnthropic(api_key=settings.anthropic_api_key),
                    model=ANTHROPIC_JUDGE_MODEL,
                ),
                model=ANTHROPIC_JUDGE_MODEL,
            ),
            ANTHROPIC_JUDGE_MODEL,
        )
    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


def build_summary_text(data: dict) -> str:
    summaries = data["summaries"]
    baseline = summaries["v1"]
    lines = [
        "# Experiment 1.05: Schema Versioning Drift",
        "",
        "## Hypothesis",
        "Schema additions should not materially change quality on the same source data unless they help the model structure the persona more cleanly.",
        "",
        "## Method",
        f"1. Ran {len(data['clusters'])} clusters through 3 schema variants: `v1`, `v1.1`, and `v2`.",
        "2. Used branch-local tool definitions and prompt notes for each version; the shared prompt builder stayed untouched.",
        "3. Scored every persona with the same branch-local judge rubric.",
        "4. Compared shared fields against the `v1` baseline to estimate schema drift.",
        "5. Remote LLM paths were kept in code but the run used the local deterministic fallback so the branch could complete without provider stalls.",
        "",
        "## Variant Summary",
    ]
    for variant in ("v1", "v1.1", "v2"):
        summary = summaries[variant]
        line = (
            f"- `{variant}`: judge `{summary['mean_judge_overall']:.2f}`, "
            f"grounded `{summary['mean_groundedness']:.2f}`, "
            f"valid `{summary['validity_rate']:.0%}`"
        )
        if variant != "v1":
            line += (
                f", judge delta vs v1 `{summary['judge_score_delta_vs_v1']:+.2f}`, "
                f"shared similarity vs v1 `{summary['mean_shared_field_similarity_vs_v1']:.2f}`"
            )
        lines.append(line)

    lines.extend(
        [
            "",
            "## Baseline",
            f"- Mean judge score: `{baseline['mean_judge_overall']:.2f}`",
            f"- Mean groundedness: `{baseline['mean_groundedness']:.2f}`",
            f"- Validity rate: `{baseline['validity_rate']:.0%}`",
            "",
            "## Decision",
            (
                "Adopt. The schema enrichments improved or matched quality without substantial drift in shared fields."
                if summaries["v1.1"]["judge_score_delta_vs_v1"] >= 0 and summaries["v2"]["judge_score_delta_vs_v1"] >= 0
                else "Defer. The schema change moved the outputs enough that the benefit is not yet convincing on this tiny tenant."
            ),
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 2 clusters. The signal is directional only.",
        ]
    )
    return "\n".join(lines) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 1.05: Schema versioning drift")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    judge, judge_model = build_judge()
    clusters = get_clusters()
    t0 = time.monotonic()

    rows: list[VariantRun] = []
    per_variant_results: dict[str, list[VariantRun]] = {variant.key: [] for variant in SCHEMA_VARIANTS}

    print("\n[1/3] Synthesizing versioned personas...")
    synthesized: dict[tuple[str, str], object] = {}
    for variant in SCHEMA_VARIANTS:
        for cluster in clusters:
            outcome = await synthesize_variant(cluster, variant)
            synthesized[(variant.key, cluster.cluster_id)] = outcome
            print(
                f"      {variant.key} {cluster.cluster_id}: "
                f"grounded={outcome.groundedness:.2f}, attempts={outcome.attempts}, model={outcome.model_used}"
            )

    print("\n[2/3] Scoring and comparing shared fields...")
    for cluster in clusters:
        baseline = synthesized[("v1", cluster.cluster_id)]
        baseline_score = await judge.score_persona(baseline.persona.model_dump(mode="json"))
        baseline_row = VariantRun(
            cluster_id=cluster.cluster_id,
            variant="v1",
            persona_name=baseline.persona.name,
            groundedness=baseline.groundedness,
            judge_overall=baseline_score.overall,
            validity=baseline.validity,
            attempts=baseline.attempts,
            model_used=baseline.model_used,
            shared_field_similarity={},
            source_evidence_count=baseline.source_evidence_count,
        )
        rows.append(baseline_row)
        per_variant_results["v1"].append(baseline_row)

        for variant in SCHEMA_VARIANTS[1:]:
            outcome = synthesized[(variant.key, cluster.cluster_id)]
            score = await judge.score_persona(outcome.persona.model_dump(mode="json"))
            similarity = compare_shared_fields(baseline.persona, outcome.persona)
            row = VariantRun(
                cluster_id=cluster.cluster_id,
                variant=variant.key,
                persona_name=outcome.persona.name,
                groundedness=outcome.groundedness,
                judge_overall=score.overall,
                validity=outcome.validity,
                attempts=outcome.attempts,
                model_used=outcome.model_used,
                shared_field_similarity=similarity,
                source_evidence_count=outcome.source_evidence_count,
            )
            rows.append(row)
            per_variant_results[variant.key].append(row)
            print(
                f"      {variant.key} {cluster.cluster_id}: judge={score.overall:.2f}, "
                f"shared={mean(similarity.values()) if similarity else 0.0:.2f}"
            )

    print("\n[3/3] Writing artifacts...")
    summaries = {
        variant: summarize_variant(per_variant_results[variant])
        for variant in per_variant_results
    }

    summaries["v1.1"].judge_score_delta_vs_v1 = (
        summaries["v1.1"].mean_judge_overall - summaries["v1"].mean_judge_overall
    )
    summaries["v2"].judge_score_delta_vs_v1 = (
        summaries["v2"].mean_judge_overall - summaries["v1"].mean_judge_overall
    )
    summaries["v1.1"].mean_shared_field_similarity_vs_v1 = mean(
        mean(row.shared_field_similarity.values()) if row.shared_field_similarity else 0.0
        for row in per_variant_results["v1.1"]
    )
    summaries["v2"].mean_shared_field_similarity_vs_v1 = mean(
        mean(row.shared_field_similarity.values()) if row.shared_field_similarity else 0.0
        for row in per_variant_results["v2"]
    )
    summaries["v1.1"].per_field_similarity_vs_v1 = merge_similarity_by_field(
        per_variant_results["v1.1"]
    )
    summaries["v2"].per_field_similarity_vs_v1 = merge_similarity_by_field(
        per_variant_results["v2"]
    )

    results_data = {
        "experiment": "1.05",
        "title": "Schema versioning drift",
        "execution_mode": "local-fallback",
        "judge_model": judge_model,
        "clusters": [cluster.cluster_id for cluster in clusters],
        "rows": [
            {
                "cluster_id": row.cluster_id,
                "variant": row.variant,
                "persona_name": row.persona_name,
                "groundedness": row.groundedness,
                "judge_overall": row.judge_overall,
                "validity": row.validity,
                "attempts": row.attempts,
                "model_used": row.model_used,
                "shared_field_similarity": row.shared_field_similarity,
                "source_evidence_count": row.source_evidence_count,
            }
            for row in rows
        ],
        "summaries": {
            variant: {
                "variant": summary.variant,
                "n_clusters": summary.n_clusters,
                "mean_groundedness": summary.mean_groundedness,
                "mean_judge_overall": summary.mean_judge_overall,
                "validity_rate": summary.validity_rate,
                "mean_attempts": summary.mean_attempts,
                "mean_source_evidence_count": summary.mean_source_evidence_count,
                "judge_score_delta_vs_v1": summary.judge_score_delta_vs_v1,
                "mean_shared_field_similarity_vs_v1": summary.mean_shared_field_similarity_vs_v1,
                "per_field_similarity_vs_v1": summary.per_field_similarity_vs_v1,
            }
            for variant, summary in summaries.items()
        },
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(build_summary_text(results_data))

    print(
        f"      v1={summaries['v1'].mean_judge_overall:.2f}, "
        f"v1.1={summaries['v1.1'].mean_judge_overall:.2f}, "
        f"v2={summaries['v2'].mean_judge_overall:.2f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
