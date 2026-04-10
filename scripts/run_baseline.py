"""Generate P1 baseline: run the full pipeline and score all personas.

Usage:
    python scripts/run_baseline.py [--output evaluation/baselines/p1_baseline.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import build_backend_from_settings  # noqa: E402
from synthesis.engine.synthesizer import synthesize_for_schema_version  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402
from synthesis.models.persona_v2 import PersonaV2  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    DEFAULT_STABILITY_FIELDS,
    capability_coherence,
    developmental_fit as schema_developmental_fit,
    historical_fit,
    relational_realism,
    run_core_metrics,
    summarize_metric_runs,
    stability_breakdown,
)
from evaluation.judges import LLMJudge, build_judge_backend_from_settings  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "baselines" / "p1_baseline.json"
DEFAULT_OUTPUT_V2 = REPO_ROOT / "evaluation" / "baselines" / "p2_baseline.json"
EXTENDED_SCHEMA_METRICS = (
    "historical_fit",
    "capability_coherence",
    "relational_realism",
)


def _schema_class(schema_version: str):
    if schema_version == "v2":
        return PersonaV2
    return PersonaV1


def _baseline_version_label(schema_version: str) -> str:
    return "p2" if schema_version == "v2" else "p1"


def _default_output_for_schema(schema_version: str) -> Path:
    return DEFAULT_OUTPUT_V2 if schema_version == "v2" else DEFAULT_OUTPUT


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _fmt(v: float) -> str:
    """Format a metric value for display."""
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.4f}"


def _round_metric_map(values: dict[str, float | None]) -> dict[str, float | None]:
    rounded: dict[str, float | None] = {}
    for key, value in values.items():
        if value is None:
            rounded[key] = None
        elif isinstance(value, float) and math.isnan(value):
            rounded[key] = None
        else:
            rounded[key] = round(float(value), 6)
    return rounded


def _round_metric_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "num_runs": summary["num_runs"],
        "means": _round_metric_map(summary["means"]),  # type: ignore[arg-type]
        "stdevs": _round_metric_map(summary["stdevs"]),  # type: ignore[arg-type]
        "mins": _round_metric_map(summary["mins"]),  # type: ignore[arg-type]
        "maxs": _round_metric_map(summary["maxs"]),  # type: ignore[arg-type]
        "counts": summary["counts"],
    }


def _summary_metric_map(baseline: dict) -> dict[str, float | None]:
    values: dict[str, float | None] = {}

    summary = baseline.get("aggregate_summary")
    if isinstance(summary, dict):
        means = summary.get("means")
        if isinstance(means, dict):
            values.update(means)

    aggregate = baseline.get("aggregate", {})
    if isinstance(aggregate, dict):
        for key, value in aggregate.items():
            values.setdefault(key, value)

    return values


def _metric_value(value: float) -> float | None:
    if math.isnan(value):
        return None
    return round(float(value), 6)


def _mean_score(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _extended_schema_metrics(persona: dict) -> dict[str, float]:
    return {
        "historical_fit": historical_fit(persona),
        "capability_coherence": capability_coherence(persona),
        "relational_realism": relational_realism(persona),
    }


def _persona_metric_bundle(persona: dict) -> dict[str, float | None]:
    bundle = {
        "developmental_fit": schema_developmental_fit(persona),
    }
    bundle.update(_extended_schema_metrics(persona))
    return {key: _metric_value(value) for key, value in bundle.items()}


def _aggregate_extended_schema_metrics(personas: list[dict]) -> dict[str, float]:
    if not personas:
        return {metric: float("nan") for metric in EXTENDED_SCHEMA_METRICS}

    scores: dict[str, list[float]] = {metric: [] for metric in EXTENDED_SCHEMA_METRICS}
    for persona in personas:
        for metric, score in _extended_schema_metrics(persona).items():
            if not math.isnan(score):
                scores[metric].append(score)

    return {
        metric: _mean_score(values)
        for metric, values in scores.items()
    }


async def run_pipeline(
    *,
    num_runs: int = 1,
    schema_version: str = "v1",
    birth_year: int | None = None,
    eval_year: int | None = None,
) -> tuple[list[list[dict]], float]:
    """Run ingest -> segment -> synthesize and return (persona_runs, total_cost).

    persona_runs is list[list[dict]] where each inner list is one full run's persona dicts.
    The first run includes the initial ingest/segment. Subsequent reruns only re-synthesize.
    """
    banner("STAGE 1/3: INGEST")
    crawler_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    print(f"  Fetched {len(records)} records")

    banner("STAGE 2/3: SEGMENT")
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = [ClusterData.model_validate(c) for c in cluster_dicts]
    print(f"  Found {len(clusters)} clusters")

    banner("STAGE 3/3: SYNTHESIZE")
    backend = build_backend_from_settings(model=settings.default_model)

    persona_runs: list[list[dict]] = []
    total_cost = 0.0

    # Run synthesis num_runs times (initial run + num_runs-1 additional reruns)
    for run_idx in range(num_runs):
        persona_entries: list[dict] = []

        if run_idx == 0:
            print(f"\n  === Synthesis Run 1/{num_runs} ===")
        else:
            print(f"\n  === Synthesis Run {run_idx + 1}/{num_runs} (rerun) ===")

        for i, cluster in enumerate(clusters):
            print(f"  [{i + 1}/{len(clusters)}] synthesizing {cluster.cluster_id}...")
            result = await synthesize_for_schema_version(
                cluster,
                backend,
                schema_version=schema_version,
                birth_year=birth_year,
                eval_year=eval_year,
            )
            persona_entries.append({
                "cluster_id": cluster.cluster_id,
                "persona": result.persona.model_dump(mode="json"),
                "cost_usd": result.total_cost_usd,
                "groundedness_score": result.groundedness.score,
                "groundedness_violations": result.groundedness.violations,
                "attempts": result.attempts,
            })
            total_cost += result.total_cost_usd
            print(
                f"      [OK] {result.persona.name} "
                f"(${result.total_cost_usd:.4f}, "
                f"score={result.groundedness.score:.2f}, "
                f"attempts={result.attempts})"
            )

        persona_runs.append(persona_entries)

    print(f"\n  Synthesis total ({num_runs} runs): ${total_cost:.4f}")
    return persona_runs, total_cost


async def score_personas(
    persona_runs: list[list[dict]],
    *,
    schema_version: str,
) -> tuple[list[dict], dict[str, float], dict, int, list[dict[str, float]], dict[str, object]]:
    """Score each persona with core metrics + LLM judge.

    Returns (per_persona, aggregate, stability_details, num_runs, run_aggregates, aggregate_summary).

    persona_runs is list[list[dict]] where each inner list is one full run's persona dicts.
    Uses all runs together to compute stability metrics.
    """
    banner("SCORING: CORE METRICS")

    # Extract persona dicts from all runs for run_core_metrics
    persona_runs_dicts = [
        [e["persona"] for e in run_personas]
        for run_personas in persona_runs
    ]

    # Build groundedness reports from the first run (representative)
    persona_entries = persona_runs[0]
    persona_dicts = [e["persona"] for e in persona_entries]

    class _GReport:
        def __init__(self, score: float):
            self.score = score

    groundedness_reports = [_GReport(e["groundedness_score"]) for e in persona_entries]

    schema_cls = _schema_class(schema_version)

    aggregate = run_core_metrics(
        personas=persona_dicts,
        schema_cls=schema_cls,
        groundedness_reports=groundedness_reports,
        persona_runs=persona_runs_dicts,
        embeddings=None,  # no embeddings in P1
    )
    aggregate.update(_aggregate_extended_schema_metrics(persona_dicts))
    run_aggregates: list[dict[str, float]] = []
    for run_personas in persona_runs:
        run_persona_dicts = [entry["persona"] for entry in run_personas]

        class _RunGReport:
            def __init__(self, score: float):
                self.score = score

        run_groundedness_reports = [_RunGReport(entry["groundedness_score"]) for entry in run_personas]
        run_metrics = run_core_metrics(
            personas=run_persona_dicts,
            schema_cls=schema_cls,
            groundedness_reports=run_groundedness_reports,
            persona_runs=None,
            embeddings=None,
        )
        run_metrics.update(_aggregate_extended_schema_metrics(run_persona_dicts))
        run_aggregates.append({k: v for k, v in run_metrics.items() if k != "stability"})

    aggregate_summary = summarize_metric_runs(run_aggregates)
    stability_details = stability_breakdown(persona_runs_dicts)

    for k, v in aggregate.items():
        print(f"  {k:24s} {_fmt(v)}")

    banner("SCORING: LLM JUDGE")
    backend = build_judge_backend_from_settings(model=settings.resolved_judge_model)
    judge = LLMJudge(backend=backend)
    per_persona: list[dict] = []

    for i, entry in enumerate(persona_entries):
        persona = entry["persona"]
        name = persona.get("name", f"persona_{i}")
        print(f"  [{i + 1}/{len(persona_entries)}] judging {name}...")
        judge_result = await judge.score_persona(persona)
        # Bug fix #4: compute developmental_fit per persona, not from aggregate
        per_persona.append({
            "name": name,
            "cluster_id": entry["cluster_id"],  # stable comparison key across versions
            "persona": persona,  # Bug fix #3: persist full persona JSON for comparison
            "metrics": {
                "groundedness": entry["groundedness_score"],
                "schema_valid": 1.0,  # if we got here it validated
                **_persona_metric_bundle(persona),
                "cost_usd": entry["cost_usd"],
            },
            "judge_score": {
                "overall": judge_result.overall,
                "dimensions": judge_result.dimensions,
                "rationale": judge_result.rationale,
            },
        })
        print(f"      judge_overall={_fmt(judge_result.overall)}")

    return per_persona, aggregate, stability_details, len(persona_runs), run_aggregates, aggregate_summary


async def main(
    output_path: Path,
    *,
    num_runs: int,
    schema_version: str,
    birth_year: int | None,
    eval_year: int | None,
) -> None:
    settings.validate_runtime_settings()
    baseline_version = _baseline_version_label(schema_version)
    banner(f"{baseline_version.upper()} BASELINE GENERATION")
    print(f"  Model:  {settings.default_model}")
    print(f"  Schema: {schema_version}")
    if schema_version == "v2":
        print(f"  Birth:  {birth_year}")
        print(f"  Eval:   {eval_year}")
    print(f"  Runs:   {num_runs} total ({max(0, num_runs - 1)} reruns)")
    print(f"  Output: {output_path}")

    persona_runs, total_cost = await run_pipeline(
        num_runs=num_runs,
        schema_version=schema_version,
        birth_year=birth_year,
        eval_year=eval_year,
    )
    per_persona, aggregate, stability_details, actual_num_runs, run_aggregates, aggregate_summary = await score_personas(
        persona_runs,
        schema_version=schema_version,
    )

    # Build final baseline document
    baseline = {
        "version": baseline_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.default_model,
        "schema_version": schema_version,
        "run_metadata": {
            "num_runs": actual_num_runs,
            "num_reruns": max(0, actual_num_runs - 1),
            "stability_fields": DEFAULT_STABILITY_FIELDS,
            "birth_year": birth_year,
            "eval_year": eval_year,
        },
        "num_personas": len(per_persona),
        "total_cost_usd": round(total_cost, 6),
        "runs": persona_runs,
        "run_aggregates": [_round_metric_map(run_aggregate) for run_aggregate in run_aggregates],
        "per_persona": per_persona,
        "aggregate_summary": _round_metric_summary(aggregate_summary),
        "stability_breakdown": {
            "overall": round(stability_details["overall"], 6)
            if not math.isnan(stability_details["overall"])
            else None,
            "num_runs": stability_details["num_runs"],
            "num_personas": stability_details["num_personas"],
            "fields": {
                field_name: {
                    "similarity": round(field_data["similarity"], 6)
                    if not math.isnan(field_data["similarity"])
                    else None,
                    "comparisons": field_data["comparisons"],
                }
                for field_name, field_data in stability_details["fields"].items()
            },
        },
        "aggregate": {k: round(v, 6) if not math.isnan(v) else None for k, v in aggregate.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2, default=str))
    print(f"\n  Baseline saved to {output_path}")

    # Summary table
    banner("BASELINE SUMMARY")
    print(f"  Version:     {baseline['version']}")
    print(f"  Model:       {baseline['model']}")
    print(f"  Runs:        {baseline['run_metadata']['num_runs']}")
    print(f"  Personas:    {baseline['num_personas']}")
    print(f"  Total cost:  ${baseline['total_cost_usd']:.4f}")
    print()
    print(f"  {'Metric':<28s} {'Value':>10s}")
    print(f"  {'-'*28} {'-'*10}")
    for k, v in _summary_metric_map(baseline).items():
        display = "N/A" if v is None else f"{v:.4f}"
        print(f"  {k:<28s} {display:>10s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate persona baseline")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save baseline JSON",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Total synthesis passes to run (default 3 = 1 initial run + 2 reruns)",
    )
    parser.add_argument(
        "--schema-version",
        choices=("v1", "v2"),
        default=settings.persona_schema_version,
        help="Persona schema version to synthesize",
    )
    parser.add_argument(
        "--birth-year",
        type=int,
        default=settings.persona_birth_year,
        help="Birth year for v2 rendering",
    )
    parser.add_argument(
        "--eval-year",
        type=int,
        default=settings.persona_eval_year,
        help="Evaluation year for v2 rendering",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        help="Deprecated alias for --num-runs",
    )
    args = parser.parse_args()

    num_runs = args.reruns if args.reruns is not None else args.num_runs
    if num_runs < 1:
        parser.error("--num-runs must be at least 1")

    output_path = args.output
    if output_path == DEFAULT_OUTPUT:
        output_path = _default_output_for_schema(args.schema_version)

    asyncio.run(
        main(
            output_path,
            num_runs=num_runs,
            schema_version=args.schema_version,
            birth_year=args.birth_year,
            eval_year=args.eval_year,
        )
    )
