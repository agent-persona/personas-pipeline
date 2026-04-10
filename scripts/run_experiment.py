"""Run a single experiment from the catalog.

Usage:
    python scripts/run_experiment.py --experiment 2.16 [--dry-run] [--repetitions N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

CATALOG_PATH = REPO_ROOT / "experiments" / "catalog.json"
OUTPUT_DIR = REPO_ROOT / "output"


def load_experiment(experiment_id: str) -> dict:
    """Load experiment spec from catalog.json."""
    if not CATALOG_PATH.exists():
        print(f"ERROR: {CATALOG_PATH} not found. Run catalog fetch first.")
        sys.exit(1)
    catalog = json.loads(CATALOG_PATH.read_text())
    exp = catalog["experiments"].get(experiment_id)
    if not exp:
        print(f"ERROR: Experiment {experiment_id} not found in catalog.")
        print(f"Available: {', '.join(sorted(catalog['experiments'].keys()))}")
        sys.exit(1)
    return exp


def slugify(title: str) -> str:
    """Convert title to branch-name-safe slug."""
    return title.lower().replace(" ", "-").replace("_", "-")


def collect_metrics() -> dict[str, float]:
    """Collect standard metrics from output/persona_*.json files."""
    persona_files = sorted(OUTPUT_DIR.glob("persona_*.json"))
    if not persona_files:
        return {
            "personas_generated": 0,
            "schema_validity": 0.0,
            "mean_groundedness": 0.0,
            "total_cost_usd": 0.0,
            "cost_per_persona": 0.0,
        }

    personas = [json.loads(f.read_text()) for f in persona_files]
    n = len(personas)

    # Schema validity
    valid_count = 0
    try:
        from synthesis.models.persona import PersonaV1
        from pydantic import ValidationError

        for p in personas:
            persona_data = p.get("persona", p)
            try:
                PersonaV1.model_validate(persona_data)
                valid_count += 1
            except (ValidationError, Exception):
                pass
    except ImportError:
        valid_count = n

    # Groundedness
    scores = [p.get("groundedness", 0.0) for p in personas]
    mean_g = sum(scores) / n

    # Cost
    costs = [p.get("cost_usd", 0.0) for p in personas]
    total_cost = sum(costs)

    return {
        "personas_generated": float(n),
        "schema_validity": valid_count / n,
        "mean_groundedness": mean_g,
        "total_cost_usd": total_cost,
        "cost_per_persona": total_cost / n,
    }


async def run_pipeline() -> str:
    """Run the full pipeline and capture output."""
    import io
    import contextlib

    # Import pipeline components
    from anthropic import AsyncAnthropic
    from crawler import fetch_all
    from orchestration import Pipeline, Stage
    from segmentation.models.record import RawRecord
    from segmentation.pipeline import segment
    from synthesis.config import settings
    from synthesis.engine.model_backend import AnthropicBackend
    from synthesis.engine.synthesizer import synthesize
    from synthesis.models.cluster import ClusterData
    from twin import TwinChat

    TENANT_ID = "tenant_acme_corp"
    TENANT_INDUSTRY = "B2B SaaS"
    TENANT_PRODUCT = "Project management tool for engineering teams"

    def stage_ingest(_):
        records = fetch_all(TENANT_ID)
        return [RawRecord.model_validate(r.model_dump()) for r in records]

    def stage_segment(records):
        dicts = segment(
            records,
            tenant_industry=TENANT_INDUSTRY,
            tenant_product=TENANT_PRODUCT,
            existing_persona_names=[],
            similarity_threshold=0.15,
            min_cluster_size=2,
        )
        return [ClusterData.model_validate(c) for c in dicts]

    async def stage_synthesize(clusters):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        backend = AnthropicBackend(client=client, model=settings.default_model)
        personas = []
        total_cost = 0.0
        for i, cluster in enumerate(clusters):
            print(f"  [{i+1}/{len(clusters)}] synthesizing {cluster.cluster_id}...")
            result = await synthesize(cluster, backend)
            personas.append({
                "cluster_id": cluster.cluster_id,
                "persona": result.persona.model_dump(mode="json"),
                "cost_usd": result.total_cost_usd,
                "groundedness": result.groundedness.score,
                "attempts": result.attempts,
            })
            total_cost += result.total_cost_usd
            print(f"      [OK] {result.persona.name} (${result.total_cost_usd:.4f})")
        print(f"\n  Synthesis total: ${total_cost:.4f}")
        return personas

    async def stage_twin_chat(personas):
        if not settings.anthropic_api_key:
            return personas
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        question = "What's the single biggest frustration you have with your current tools?"
        for entry in personas:
            twin = TwinChat(entry["persona"], client=client, model=settings.default_model)
            reply = await twin.reply(question)
            entry["twin_demo_reply"] = reply.text
            entry["twin_demo_cost"] = reply.estimated_cost_usd
        return personas

    def stage_persist(personas):
        OUTPUT_DIR.mkdir(exist_ok=True)
        for i, entry in enumerate(personas):
            path = OUTPUT_DIR / f"persona_{i:02d}.json"
            path.write_text(json.dumps(entry, indent=2, default=str))
        print(f"  Wrote {len(personas)} personas to {OUTPUT_DIR}/")
        return personas

    pipeline = Pipeline([
        Stage(name="ingest", fn=stage_ingest, description="Pull from connectors"),
        Stage(name="segment", fn=stage_segment, description="Cluster by behavior"),
        Stage(name="synthesize", fn=stage_synthesize, description="Generate personas"),
        Stage(name="twin_chat", fn=stage_twin_chat, description="Demo twin replies"),
        Stage(name="persist", fn=stage_persist, description="Save outputs"),
    ])

    state = await pipeline.run(initial_input=None, tenant_id=TENANT_ID)
    summary = (
        f"Run ID: {state.run_id}\n"
        f"Success: {state.success}\n"
        f"Duration: {state.total_duration_ms} ms\n"
    )
    for s in state.stages:
        status = "ok" if s.success else "FAILED"
        summary += f"  {s.name:14s} {status:8s} {s.duration_ms:6d} ms\n"
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run a personas-pipeline experiment")
    parser.add_argument("--experiment", required=True, help="Experiment ID (e.g. 2.16)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions")
    args = parser.parse_args()

    exp = load_experiment(args.experiment)
    slug = slugify(exp["title"])

    print(f"\n{'='*72}")
    print(f"EXPERIMENT {exp['id']}: {exp['title']}")
    print(f"{'='*72}")
    print(f"  Branch:     exp-{exp['id']}-{slug}")
    print(f"  Metric:     {exp['metric']}")
    print(f"  Hypothesis: {exp['hypothesis']}")
    print(f"  Files:      {', '.join(exp['files'])}")
    print(f"  Repetitions: {args.repetitions}")

    if args.dry_run:
        print("\n  [DRY RUN] Would execute pipeline and collect metrics.")
        print(f"  Change: {exp['change_description']}")
        return

    results_dir = OUTPUT_DIR / "experiments" / f"exp-{exp['id']}-{slug}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for rep in range(args.repetitions):
        if args.repetitions > 1:
            print(f"\n  --- Repetition {rep+1}/{args.repetitions} ---")

        # Clear output
        for f in OUTPUT_DIR.glob("persona_*.json"):
            f.unlink()

        # Run pipeline
        summary = await run_pipeline()
        print(summary)

        # Collect metrics
        metrics = collect_metrics()
        all_metrics.append(metrics)
        print(f"\n  Metrics: {json.dumps(metrics, indent=2)}")

    # Average if multiple reps
    if len(all_metrics) > 1:
        avg = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            avg[key] = sum(values) / len(values)
        final_metrics = avg
    else:
        final_metrics = all_metrics[0]

    # Write results
    result = {
        "experiment": exp,
        "metrics": final_metrics,
        "all_runs": all_metrics,
        "repetitions": args.repetitions,
    }
    results_path = results_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Results written to {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
