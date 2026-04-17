"""Records-in, personas-out CLI for the personas pipeline.

Reads a JSON array of RawRecord dicts, runs segment -> synthesize, writes a
JSON result blob. Designed to be shelled out to from the SaaS backend.

Usage:
    python scripts/run_pipeline_cli.py \\
        --records records.json \\
        --out result.json \\
        --tenant-industry "B2B SaaS" \\
        --tenant-product "Project management tool" \\
        --tier standard

Input file: JSON list of RawRecord dicts (see segmentation/models/record.py).
Output: {
    "run": {"success": bool, "error": str|null, "total_cost_usd": float,
            "duration_ms": int},
    "stages": [{"name": str, "success": bool, "duration_ms": int,
                "output_summary": str, "error": str|null}, ...],
    "personas": [{"cluster_id": str, "persona": {...},
                  "groundedness": float, "cost_usd": float,
                  "model": str, "attempts": int}, ...]
}
Exit code: 0 on success, 1 on fatal error (stages/error still emitted).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from anthropic import AsyncAnthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Load synthesis/.env as a fallback so local runs work without the SaaS passing
# ANTHROPIC_API_KEY through the process env. The SaaS deploy should set the env
# var directly; this just preserves the scripts/run_full_pipeline.py behavior.
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the personas pipeline on a batch of records.")
    p.add_argument("--records", type=Path, help="Path to JSON array of RawRecord dicts. Reads stdin if omitted.")
    p.add_argument("--out", type=Path, help="Path to write result JSON. Writes stdout if omitted.")
    p.add_argument("--tenant-industry", default=None)
    p.add_argument("--tenant-product", default=None)
    p.add_argument("--existing-personas", default="", help="Comma-separated existing persona names.")
    p.add_argument("--similarity-threshold", type=float, default=0.15)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--tier", choices=["standard", "premium"], default="standard")
    p.add_argument("--model", default=None, help="Override model id. Defaults by tier.")
    return p.parse_args()


def load_records(path: Path | None) -> list[RawRecord]:
    raw = path.read_text(encoding="utf-8") if path else sys.stdin.read()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of record objects.")
    return [RawRecord.model_validate(r) for r in data]


def resolve_model(tier: str, override: str | None) -> str:
    if override:
        return override
    return settings.premium_model if tier == "premium" else settings.default_model


async def run(args: argparse.Namespace) -> dict:
    stages: list[dict] = []
    total_cost = 0.0
    personas: list[dict] = []
    run_started = time.monotonic()
    error: str | None = None

    def stage(name: str, started: float, success: bool, summary: str, err: str | None = None) -> None:
        stages.append({
            "name": name,
            "success": success,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "output_summary": summary,
            "error": err,
        })

    try:
        # Stage 1: ingest (load records)
        t0 = time.monotonic()
        records = load_records(args.records)
        stage("ingest", t0, True, f"{len(records)} records loaded")

        if not records:
            stage("segment", time.monotonic(), False, "No records to segment", "Empty input")
            return _result(False, "Empty input", total_cost, run_started, stages, personas)

        # Stage 2: segment
        t0 = time.monotonic()
        existing = [n.strip() for n in args.existing_personas.split(",") if n.strip()]
        cluster_dicts = segment(
            records,
            tenant_industry=args.tenant_industry,
            tenant_product=args.tenant_product,
            existing_persona_names=existing,
            similarity_threshold=args.similarity_threshold,
            min_cluster_size=args.min_cluster_size,
        )
        clusters = [ClusterData.model_validate(c) for c in cluster_dicts]
        stage("segment", t0, True, f"{len(clusters)} clusters found")

        if not clusters:
            return _result(True, None, total_cost, run_started, stages, personas)

        # Stage 3: synthesize
        t0 = time.monotonic()
        api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        model = resolve_model(args.tier, args.model)
        client = AsyncAnthropic(api_key=api_key)
        backend = AnthropicBackend(client=client, model=model)

        for cluster in clusters:
            try:
                result = await synthesize(cluster, backend)
                personas.append({
                    "cluster_id": cluster.cluster_id,
                    "persona": result.persona.model_dump(mode="json"),
                    "groundedness": result.groundedness.score,
                    "cost_usd": result.total_cost_usd,
                    "model": model,
                    "attempts": result.attempts,
                })
                total_cost += result.total_cost_usd
            except Exception as e:  # noqa: BLE001 — record per-cluster failure, keep going
                personas.append({
                    "cluster_id": cluster.cluster_id,
                    "persona": None,
                    "groundedness": 0.0,
                    "cost_usd": 0.0,
                    "model": model,
                    "attempts": 0,
                    "error": f"{type(e).__name__}: {e}",
                })

        successful = sum(1 for p in personas if p.get("persona") is not None)
        stage("synthesize", t0, successful > 0, f"{successful}/{len(clusters)} personas generated")

        if successful == 0:
            first_err = next((p["error"] for p in personas if p.get("error")), "All syntheses failed")
            return _result(False, first_err, total_cost, run_started, stages, personas)

        return _result(True, None, total_cost, run_started, stages, personas)

    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        sys.stderr.write(traceback.format_exc())
        return _result(False, error, total_cost, run_started, stages, personas)


def _result(success: bool, error: str | None, cost: float, started: float,
            stages: list[dict], personas: list[dict]) -> dict:
    return {
        "run": {
            "success": success,
            "error": error,
            "total_cost_usd": cost,
            "duration_ms": int((time.monotonic() - started) * 1000),
        },
        "stages": stages,
        "personas": personas,
    }


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))

    payload = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
        sys.stdout.write("\n")

    return 0 if result["run"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
