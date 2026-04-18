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
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from anthropic import AsyncAnthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

try:
    from openai import AsyncOpenAI  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency at runtime
    AsyncOpenAI = None  # type: ignore[assignment]

# Load synthesis/.env as a fallback so local runs work without the SaaS passing
# ANTHROPIC_API_KEY through the process env. The SaaS deploy should set the env
# var directly; this just preserves the scripts/run_full_pipeline.py behavior.
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, OpenAIJsonBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PublicPersonPersonaV1  # noqa: E402

OPENAI_DEFAULT_MODEL = "gpt-5-nano"


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
    p.add_argument("--mode", choices=["normal", "public-person"], default="normal")
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


def build_backend(args: argparse.Namespace):
    anthropic_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_model = args.model or os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)

    if openai_key:
        if AsyncOpenAI is None:
            raise RuntimeError("OPENAI_API_KEY set but openai package is not installed")
        client = AsyncOpenAI(api_key=openai_key)
        return OpenAIJsonBackend(client=client, model=openai_model), "openai", openai_model

    if anthropic_key:
        model = resolve_model(args.tier, args.model)
        client = AsyncAnthropic(api_key=anthropic_key)
        return AnthropicBackend(client=client, model=model), "anthropic", model

    raise RuntimeError("ANTHROPIC_API_KEY or OPENAI_API_KEY required")


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
        if args.mode == "public-person":
            clusters = [build_public_person_cluster(
                records,
                tenant_industry=args.tenant_industry,
                tenant_product=args.tenant_product,
            )]
            stage("segment", t0, True, "1 public-person cluster built")
        else:
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
        backend, provider, model = build_backend(args)

        for cluster in clusters:
            try:
                if args.mode == "public-person":
                    result = await synthesize(
                        cluster,
                        backend,
                        schema_cls=PublicPersonPersonaV1,
                        prompt_kind="public_person",
                    )
                else:
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
        stages.append({
            "name": "synthesize",
            "success": successful > 0,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "output_summary": f"{successful}/{len(clusters)} personas generated",
            "error": None,
            "details": {
                "provider": provider,
                "model": model,
                "cluster_count": len(clusters),
                "generated_count": successful,
                "failed_count": len(clusters) - successful,
            },
        })

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


def build_public_person_cluster(
    records: list[RawRecord],
    *,
    tenant_industry: str | None = None,
    tenant_product: str | None = None,
) -> ClusterData:
    """Force one public-person cluster from all usable public crawl records."""
    tenant_id = records[0].tenant_id
    behavior_counts = Counter(
        behavior
        for record in records
        for behavior in record.behaviors
    )
    page_counts = Counter(
        page
        for record in records
        for page in record.pages
    )
    first_payload = records[0].payload if records else {}
    submitted_profile = first_payload.get("submitted_profile", {})
    public_identity = first_payload.get("public_identity", {})
    source_audit = first_payload.get("source_audit", {})
    crawl_notes = first_payload.get("crawl_notes", [])

    return ClusterData.model_validate({
        "cluster_id": "public_person",
        "tenant": {
            "tenant_id": tenant_id,
            "industry": tenant_industry,
            "product_description": tenant_product,
            "existing_persona_names": [],
        },
        "summary": {
            "cluster_size": len(records),
            "top_behaviors": [item for item, _count in behavior_counts.most_common(12)],
            "top_pages": [item for item, _count in page_counts.most_common(12)],
            "extra": {
                "mode": "public_person",
                "submitted_profile": submitted_profile,
                "public_identity": public_identity,
                "source_audit": source_audit,
                "crawl_notes": crawl_notes,
            },
        },
        "sample_records": [
            {
                "record_id": record.record_id,
                "source": record.source,
                "timestamp": record.timestamp,
                "payload": record.payload,
            }
            for record in records
        ],
        "enrichment": {
            "extra": {
                "public_person": True,
            },
        },
    })


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
