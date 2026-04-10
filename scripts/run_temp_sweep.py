"""Run experiment 2.06: temperature sweep on the P1 pipeline.

Usage:
    python scripts/run_temp_sweep.py --num-runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict
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
from evaluation.judges import LLMJudge, build_judge_backend_from_settings  # noqa: E402
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
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import build_backend_from_settings  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize_for_schema_version  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402
from synthesis.models.persona_v2 import PersonaV2  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "baselines" / "temp_sweep.json"
DEFAULT_OUTPUT_V2 = REPO_ROOT / "evaluation" / "baselines" / "temp_sweep_v2.json"
DEFAULT_TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]
EXTENDED_SCHEMA_METRICS = (
    "historical_fit",
    "capability_coherence",
    "relational_realism",
)


def _schema_class(schema_version: str):
    if schema_version == "v2":
        return PersonaV2
    return PersonaV1


def _default_output_for_schema(schema_version: str) -> Path:
    return DEFAULT_OUTPUT_V2 if schema_version == "v2" else DEFAULT_OUTPUT


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _fmt(v: float | None) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.4f}"


def _round_metric_map(values: dict[str, float]) -> dict[str, float | None]:
    return {
        key: round(value, 6) if not math.isnan(value) else None
        for key, value in values.items()
    }


def _round_metric_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "num_runs": summary["num_runs"],
        "means": _round_metric_map(summary["means"]),  # type: ignore[arg-type]
        "stdevs": _round_metric_map(summary["stdevs"]),  # type: ignore[arg-type]
        "mins": _round_metric_map(summary["mins"]),  # type: ignore[arg-type]
        "maxs": _round_metric_map(summary["maxs"]),  # type: ignore[arg-type]
        "counts": summary["counts"],
    }


def _round_stability_details(stability_details: dict) -> dict:
    return {
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
    }


def _parse_temperatures(raw_values: list[str] | None) -> list[float]:
    if not raw_values:
        return DEFAULT_TEMPERATURES

    parsed: list[float] = []
    for raw in raw_values:
        for piece in raw.split(","):
            value = piece.strip()
            if not value:
                continue
            parsed.append(float(value))

    if not parsed:
        raise ValueError("At least one temperature is required")

    for value in parsed:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Temperature out of range [0, 1]: {value}")

    return parsed


def _comparison_metric_map(result: dict) -> dict[str, float | None]:
    comparison: dict[str, float | None] = {}

    summary = result.get("aggregate_summary")
    if isinstance(summary, dict):
        means = summary.get("means")
        if isinstance(means, dict) and means:
            comparison.update(means)
    aggregate = result.get("aggregate", {})
    if isinstance(aggregate, dict):
        for key, value in aggregate.items():
            comparison.setdefault(key, value)
    return comparison


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


async def ingest_and_segment() -> list[ClusterData]:
    banner("STAGE 1/2: INGEST + SEGMENT")
    crawler_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    print(f"  Fetched {len(records)} records")

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
    return clusters


async def synthesize_runs(
    clusters: list[ClusterData],
    *,
    num_runs: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    schema_version: str,
    birth_year: int | None,
    eval_year: int | None,
) -> tuple[list[list[dict]], float]:
    banner(f"STAGE 2/2: SYNTHESIZE @ temperature={temperature}")
    backend = build_backend_from_settings(
        model=settings.default_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    persona_runs: list[list[dict]] = []
    total_cost = 0.0

    for run_idx in range(num_runs):
        persona_entries: list[dict] = []
        print(f"\n  === Temperature {temperature} | Run {run_idx + 1}/{num_runs} ===")

        for cluster_idx, cluster in enumerate(clusters):
            print(
                f"  [{cluster_idx + 1}/{len(clusters)}] synthesizing {cluster.cluster_id}...",
            )
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
                f"attempts={result.attempts})",
            )

        persona_runs.append(persona_entries)

    print(f"\n  Temperature {temperature} total cost: ${total_cost:.4f}")
    return persona_runs, total_cost


async def score_temperature_variant(
    persona_runs: list[list[dict]],
    *,
    run_judge: bool,
    schema_version: str,
) -> tuple[list[dict], dict[str, float], dict, float, list[dict[str, float]], dict[str, object]]:
    persona_runs_dicts = [
        [entry["persona"] for entry in run_personas]
        for run_personas in persona_runs
    ]
    first_run_entries = persona_runs[0]
    personas = [entry["persona"] for entry in first_run_entries]

    class _GReport:
        def __init__(self, score: float):
            self.score = score

    groundedness_reports = [
        _GReport(entry["groundedness_score"])
        for entry in first_run_entries
    ]

    schema_cls = _schema_class(schema_version)
    aggregate = run_core_metrics(
        personas=personas,
        schema_cls=schema_cls,
        groundedness_reports=groundedness_reports,
        persona_runs=persona_runs_dicts,
        embeddings=None,
    )
    aggregate.update(_aggregate_extended_schema_metrics(personas))
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

    per_persona: list[dict] = []
    judge_overalls: list[float] = []

    judge: LLMJudge | None = None
    if run_judge:
        backend = build_judge_backend_from_settings(model=settings.resolved_judge_model)
        judge = LLMJudge(backend=backend)

    for persona_idx, entry in enumerate(first_run_entries):
        persona = entry["persona"]
        name = persona.get("name", f"persona_{persona_idx}")
        if judge is not None:
            print(f"  judging {name}...")
            judge_result = await judge.score_persona(persona)
            judge_overalls.append(judge_result.overall)
            print(f"      judge_overall={_fmt(judge_result.overall)}")
            judge_score = {
                "overall": judge_result.overall,
                "dimensions": judge_result.dimensions,
                "rationale": judge_result.rationale,
            }
        else:
            judge_score = {
                "overall": None,
                "dimensions": {},
                "rationale": "Skipped via --skip-judge",
            }
        per_persona.append({
            "name": name,
            "cluster_id": entry["cluster_id"],
            "persona": persona,
            "metrics": {
                "groundedness": entry["groundedness_score"],
                "schema_valid": 1.0,
                **_persona_metric_bundle(persona),
                "cost_usd": entry["cost_usd"],
            },
            "judge_score": judge_score,
        })

    mean_judge = sum(judge_overalls) / len(judge_overalls) if judge_overalls else float("nan")
    return per_persona, aggregate, stability_details, mean_judge, run_aggregates, aggregate_summary


def pick_best_temperature(results: list[dict], groundedness_tolerance: float) -> dict:
    successful = [
        result for result in results
        if result.get("status", "ok") == "ok" and result.get("aggregate")
    ]
    if not successful:
        return {
            "selection_rule": (
                "max groundedness within tolerance, then highest stability, "
                "then highest judge mean, then lower temperature, then lower cost"
            ),
            "groundedness_tolerance": groundedness_tolerance,
            "max_groundedness_observed": None,
            "eligible_temperatures": [],
            "best_temperature": None,
            "best_summary": None,
        }

    max_groundedness = max(
        (_comparison_metric_map(result).get("groundedness") or 0.0)
        for result in successful
    )
    eligible = [
        result
        for result in successful
        if (_comparison_metric_map(result).get("groundedness") or 0.0) >= (max_groundedness - groundedness_tolerance)
    ]
    eligible.sort(
        key=lambda result: (
            _comparison_metric_map(result).get("stability") or float("-inf"),
            result.get("judge_overall_mean") or float("-inf"),
            -(result.get("temperature") or 0.0),
            -(result.get("total_cost_usd") or 0.0),
        ),
        reverse=True,
    )
    winner = eligible[0]
    return {
        "selection_rule": (
            "max groundedness within tolerance, then highest stability, "
            "then highest judge mean, then lower temperature, then lower cost"
        ),
        "groundedness_tolerance": groundedness_tolerance,
        "max_groundedness_observed": round(max_groundedness, 6),
        "eligible_temperatures": [result["temperature"] for result in eligible],
        "best_temperature": winner["temperature"],
        "best_summary": {
            "temperature": winner["temperature"],
            "groundedness": _comparison_metric_map(winner).get("groundedness"),
            "stability": _comparison_metric_map(winner).get("stability"),
            "developmental_fit": _comparison_metric_map(winner).get("developmental_fit"),
            "historical_fit": _comparison_metric_map(winner).get("historical_fit"),
            "capability_coherence": _comparison_metric_map(winner).get("capability_coherence"),
            "relational_realism": _comparison_metric_map(winner).get("relational_realism"),
            "judge_overall_mean": winner.get("judge_overall_mean"),
            "total_cost_usd": winner.get("total_cost_usd"),
        },
    }


def _build_report(
    *,
    output_path: Path,
    temperatures: list[float],
    num_runs: int,
    top_p: float | None,
    top_k: int | None,
    schema_version: str,
    birth_year: int | None,
    eval_year: int | None,
    run_judge: bool,
    groundedness_tolerance: float,
    clusters: list[ClusterData],
    experiment_total_cost: float,
    results: list[dict],
) -> dict:
    decision = pick_best_temperature(results, groundedness_tolerance)
    return {
        "experiment_id": "2.06",
        "name": "Temperature & sampling sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.default_model,
        "schema_version": schema_version,
        "tenant_id": TENANT_ID,
        "output_path": str(output_path),
        "run_metadata": {
            "num_runs_per_temperature": num_runs,
            "num_reruns_per_temperature": max(0, num_runs - 1),
            "temperatures": temperatures,
            "top_p": top_p,
            "top_k": top_k,
            "run_judge": run_judge,
            "stability_fields": DEFAULT_STABILITY_FIELDS,
            "num_clusters": len(clusters),
            "birth_year": birth_year,
            "eval_year": eval_year,
        },
        "selection": decision,
        "total_cost_usd": round(experiment_total_cost, 6),
        "results": results,
    }


def _write_report(output_path: Path, report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))


async def main(
    output_path: Path,
    temperatures: list[float],
    *,
    num_runs: int,
    top_p: float | None,
    top_k: int | None,
    schema_version: str,
    birth_year: int | None,
    eval_year: int | None,
    groundedness_tolerance: float,
    run_judge: bool,
) -> None:
    settings.validate_runtime_settings()
    banner("EXPERIMENT 2.06: TEMPERATURE SWEEP")
    print(f"  Model:        {settings.default_model}")
    print(f"  Schema:       {schema_version}")
    if schema_version == "v2":
        print(f"  Birth year:   {birth_year}")
        print(f"  Eval year:    {eval_year}")
    print(f"  Temperatures: {', '.join(str(t) for t in temperatures)}")
    print(f"  Runs/temp:    {num_runs} total ({max(0, num_runs - 1)} reruns)")
    print(f"  Judge:        {'enabled' if run_judge else 'skipped'}")
    print(f"  Output:       {output_path}")

    clusters = await ingest_and_segment()
    results: list[dict] = []
    experiment_total_cost = 0.0

    for temperature in temperatures:
        try:
            persona_runs, total_cost = await synthesize_runs(
                clusters,
                num_runs=num_runs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                schema_version=schema_version,
                birth_year=birth_year,
                eval_year=eval_year,
            )
        except SynthesisError as exc:
            failed_result = {
                "temperature": temperature,
                "status": "failed",
                "num_personas": 0,
                "total_cost_usd": None,
                "judge_overall_mean": None,
                "runs": [],
                "per_persona": [],
                "run_aggregates": [],
                "stability_breakdown": None,
                "aggregate": {},
                "aggregate_summary": {
                    "num_runs": 0,
                    "means": {},
                    "stdevs": {},
                    "mins": {},
                    "maxs": {},
                    "counts": {},
                },
                "failure": {
                    "error": str(exc),
                    "attempts": [asdict(attempt) for attempt in exc.attempts],
                },
            }
            results.append(failed_result)
            print(f"\n  temperature {temperature} failed: {exc}")
            report = _build_report(
                output_path=output_path,
                temperatures=temperatures,
                num_runs=num_runs,
                top_p=top_p,
                top_k=top_k,
                schema_version=schema_version,
                birth_year=birth_year,
                eval_year=eval_year,
                run_judge=run_judge,
                groundedness_tolerance=groundedness_tolerance,
                clusters=clusters,
                experiment_total_cost=experiment_total_cost,
                results=results,
            )
            _write_report(output_path, report)
            continue

        experiment_total_cost += total_cost

        print(f"\n  scoring temperature {temperature}...")
        per_persona, aggregate, stability_details, judge_overall_mean, run_aggregates, aggregate_summary = await score_temperature_variant(
            persona_runs,
            run_judge=run_judge,
            schema_version=schema_version,
        )

        result = {
            "temperature": temperature,
            "num_personas": len(per_persona),
            "total_cost_usd": round(total_cost, 6),
            "judge_overall_mean": round(judge_overall_mean, 6)
            if not math.isnan(judge_overall_mean)
            else None,
            "runs": persona_runs,
            "run_aggregates": [_round_metric_map(run_aggregate) for run_aggregate in run_aggregates],
            "per_persona": per_persona,
            "stability_breakdown": _round_stability_details(stability_details),
            "aggregate": _round_metric_map(aggregate),
            "aggregate_summary": _round_metric_summary(aggregate_summary),
        }
        results.append(result)

        print(
            "  summary:"
            f" groundedness_mean={_fmt(_comparison_metric_map(result).get('groundedness'))},"
            f" stability={_fmt(_comparison_metric_map(result).get('stability'))},"
            f" judge={_fmt(result['judge_overall_mean'])},"
            f" cost=${result['total_cost_usd']:.4f}",
        )
        report = _build_report(
            output_path=output_path,
            temperatures=temperatures,
            num_runs=num_runs,
            top_p=top_p,
            top_k=top_k,
            schema_version=schema_version,
            birth_year=birth_year,
            eval_year=eval_year,
            run_judge=run_judge,
            groundedness_tolerance=groundedness_tolerance,
            clusters=clusters,
            experiment_total_cost=experiment_total_cost,
            results=results,
        )
        _write_report(output_path, report)

    report = _build_report(
        output_path=output_path,
        temperatures=temperatures,
        num_runs=num_runs,
        top_p=top_p,
        top_k=top_k,
        schema_version=schema_version,
        birth_year=birth_year,
        eval_year=eval_year,
        run_judge=run_judge,
        groundedness_tolerance=groundedness_tolerance,
        clusters=clusters,
        experiment_total_cost=experiment_total_cost,
        results=results,
    )

    banner("TEMPERATURE SWEEP SUMMARY")
    print(f"  Saved:            {output_path}")
    print(f"  Total cost:       ${report['total_cost_usd']:.4f}")
    print(f"  Best temperature: {report['selection']['best_temperature']}")
    print()
    print(f"  {'Temp':<8} {'Grounded':>10} {'Stability':>10} {'Judge':>10} {'Cost':>10}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for result in results:
        if result.get("status") == "failed":
            print(
                f"  {result['temperature']:<8.1f}"
                f" {'FAILED':>10}"
                f" {'FAILED':>10}"
                f" {'FAILED':>10}"
                f" {'N/A':>10}",
            )
            continue
        print(
            f"  {result['temperature']:<8.1f}"
            f" {_fmt(_comparison_metric_map(result).get('groundedness')):>10}"
            f" {_fmt(_comparison_metric_map(result).get('stability')):>10}"
            f" {_fmt(result.get('judge_overall_mean')):>10}"
            f" ${result['total_cost_usd']:>8.4f}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment 2.06 temperature sweep")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save sweep JSON",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Total synthesis passes per temperature (default 3 = 1 initial run + 2 reruns)",
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
    parser.add_argument(
        "--temperature",
        action="append",
        dest="temperatures",
        help="Temperature to test. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional fixed top_p to apply to all temperature runs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional fixed top_k to apply to all temperature runs",
    )
    parser.add_argument(
        "--groundedness-tolerance",
        type=float,
        default=0.01,
        help="Allowable groundedness drop from the max when selecting best temperature",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge calls for cheaper synthesis-only canary runs",
    )
    args = parser.parse_args()

    num_runs = args.reruns if args.reruns is not None else args.num_runs
    if num_runs < 1:
        parser.error("--num-runs must be at least 1")

    try:
        temperatures = _parse_temperatures(args.temperatures)
    except ValueError as exc:
        parser.error(str(exc))

    output_path = args.output
    if output_path == DEFAULT_OUTPUT:
        output_path = _default_output_for_schema(args.schema_version)

    asyncio.run(
        main(
            output_path,
            temperatures,
            num_runs=num_runs,
            top_p=args.top_p,
            top_k=args.top_k,
            schema_version=args.schema_version,
            birth_year=args.birth_year,
            eval_year=args.eval_year,
            groundedness_tolerance=args.groundedness_tolerance,
            run_judge=not args.skip_judge,
        ),
    )
