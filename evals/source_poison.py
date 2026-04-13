"""Experiment 3.07 — Adversarial source injection.

Generates 5 false records (wrong industry, wrong role, wrong behaviors)
and injects them into a clean cluster. Runs synthesis on both the clean
and poisoned cluster, then checks how much of the false data leaked into
the persona.

Usage:
    python evals/source_poison.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from crawler import fetch_all
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import SynthesisError, synthesize
from synthesis.models.cluster import ClusterData, SampleRecord

TENANT_ID = "tenant_acme_corp"

# ============================================================================
# Poisoned records — completely wrong industry, role, and behaviors
# ============================================================================

POISON_RECORDS = [
    SampleRecord(
        record_id="poison_001",
        source="hubspot",
        timestamp="2026-04-01T10:00:00Z",
        payload={
            "contact_title": "Head Pastry Chef",
            "company_size": "5-10",
            "industry": "agriculture",
            "message": "We need better grain sourcing for our artisan bread line. The current supplier is inconsistent.",
        },
    ),
    SampleRecord(
        record_id="poison_002",
        source="intercom",
        timestamp="2026-04-02T14:00:00Z",
        payload={
            "message": "Can your platform track livestock vaccination schedules? We have 200 head of cattle.",
            "topic": "livestock_management",
        },
    ),
    SampleRecord(
        record_id="poison_003",
        source="ga4",
        timestamp="2026-04-03T09:00:00Z",
        payload={
            "event": "crop_rotation_planner",
            "session_duration": 3400,
            "topic": "farming",
        },
    ),
    SampleRecord(
        record_id="poison_004",
        source="hubspot",
        timestamp="2026-04-01T11:00:00Z",
        payload={
            "contact_title": "Organic Farm Manager",
            "company_size": "10-20",
            "industry": "organic_farming",
            "message": "Our harvest yield tracking is all manual spreadsheets. Need to digitize before next planting season.",
        },
    ),
    SampleRecord(
        record_id="poison_005",
        source="intercom",
        timestamp="2026-04-04T16:00:00Z",
        payload={
            "message": "The irrigation scheduling feature is great but it doesn't integrate with our weather station API.",
            "topic": "irrigation",
        },
    ),
]

# Keywords that indicate poison absorption — if these appear in the persona,
# the false records leaked through
POISON_MARKERS = [
    # Industry markers
    "agriculture", "farming", "organic", "harvest", "crop", "livestock",
    "cattle", "irrigation", "grain", "bread", "pastry", "artisan",
    "planting", "weather station", "vaccination", "veterinar",
    # Role markers
    "chef", "farm manager", "farmer",
    # Behavior markers
    "crop rotation", "yield tracking", "livestock", "irrigation",
    "grain sourcing", "harvest yield",
]

# Fields to scan for poison absorption
SCAN_FIELDS = [
    "name", "summary", "goals", "pains", "motivations", "objections",
    "vocabulary", "sample_quotes", "channels", "decision_triggers",
]


@dataclass
class AbsorptionResult:
    mode: str  # "clean" or "poisoned"
    cluster_id: str
    persona_name: str
    groundedness_score: float
    attempts: int
    cost_usd: float
    # Absorption analysis
    absorbed_markers: list[str]
    absorption_rate: float  # fraction of poison markers found
    absorption_locations: dict[str, list[str]]  # field -> markers found
    # Persona content for comparison
    goals: list[str]
    pains: list[str]
    vocabulary: list[str]
    firmographics: dict
    failed: bool = False
    error: str = ""


def scan_for_poison(persona_dict: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Scan persona fields for poison markers. Return (markers_found, locations)."""
    found: list[str] = []
    locations: dict[str, list[str]] = {}

    for field_name in SCAN_FIELDS:
        value = persona_dict.get(field_name, "")
        if isinstance(value, list):
            text = " ".join(str(v) for v in value).lower()
        elif isinstance(value, dict):
            text = json.dumps(value).lower()
        else:
            text = str(value).lower()

        field_hits = []
        for marker in POISON_MARKERS:
            if marker.lower() in text:
                if marker not in found:
                    found.append(marker)
                field_hits.append(marker)
        if field_hits:
            locations[field_name] = field_hits

    # Also scan firmographics and demographics
    for nested_field in ["firmographics", "demographics"]:
        value = persona_dict.get(nested_field, {})
        text = json.dumps(value).lower()
        field_hits = []
        for marker in POISON_MARKERS:
            if marker.lower() in text:
                if marker not in found:
                    found.append(marker)
                field_hits.append(marker)
        if field_hits:
            locations[nested_field] = field_hits

    return found, locations


def inject_poison(cluster: ClusterData, poison_records: list[SampleRecord]) -> ClusterData:
    """Return a new ClusterData with poison records injected."""
    new_records = list(cluster.sample_records) + poison_records
    # Update summary to reflect new record count
    new_summary = cluster.summary.model_copy(
        update={"cluster_size": cluster.summary.cluster_size + len(poison_records)}
    )
    return cluster.model_copy(
        update={
            "cluster_id": cluster.cluster_id + "_poisoned",
            "sample_records": new_records,
            "summary": new_summary,
        }
    )


async def run_synthesis_and_scan(
    cluster: ClusterData,
    backend: AnthropicBackend,
    mode: str,
) -> AbsorptionResult:
    """Synthesize a persona and scan for poison absorption."""
    print(f"  [{mode}] {cluster.cluster_id} ({len(cluster.sample_records)} records)...", end="", flush=True)
    t0 = time.monotonic()
    try:
        result = await synthesize(cluster, backend)
        elapsed = time.monotonic() - t0
        persona_dict = result.persona.model_dump(mode="json")
        absorbed, locations = scan_for_poison(persona_dict)
        rate = len(absorbed) / len(POISON_MARKERS) if POISON_MARKERS else 0.0

        print(f" {result.persona.name} (g={result.groundedness.score:.2f}, "
              f"absorbed={len(absorbed)}/{len(POISON_MARKERS)}, {elapsed:.1f}s)")

        return AbsorptionResult(
            mode=mode,
            cluster_id=cluster.cluster_id,
            persona_name=result.persona.name,
            groundedness_score=result.groundedness.score,
            attempts=result.attempts,
            cost_usd=result.total_cost_usd,
            absorbed_markers=absorbed,
            absorption_rate=rate,
            absorption_locations=locations,
            goals=result.persona.goals,
            pains=result.persona.pains,
            vocabulary=result.persona.vocabulary,
            firmographics=persona_dict.get("firmographics", {}),
        )
    except SynthesisError as e:
        elapsed = time.monotonic() - t0
        print(f" FAILED ({elapsed:.1f}s)")
        return AbsorptionResult(
            mode=mode,
            cluster_id=cluster.cluster_id,
            persona_name="FAILED",
            groundedness_score=0.0,
            attempts=len(e.attempts),
            cost_usd=sum(a.cost_usd for a in e.attempts),
            absorbed_markers=[], absorption_rate=0.0,
            absorption_locations={},
            goals=[], pains=[], vocabulary=[], firmographics={},
            failed=True, error=str(e),
        )


def print_results(results: list[AbsorptionResult]) -> None:
    print("\n" + "=" * 100)
    print("EXPERIMENT 3.07 — ADVERSARIAL SOURCE INJECTION")
    print("=" * 100)

    # Per-cluster results
    clusters = sorted(set(r.cluster_id.replace("_poisoned", "") for r in results))
    for base_cid in clusters:
        clean = next((r for r in results if r.cluster_id == base_cid and r.mode == "clean"), None)
        poisoned = next((r for r in results if r.cluster_id == base_cid + "_poisoned" and r.mode == "poisoned"), None)

        print(f"\n--- Cluster: {base_cid} ---")
        print(f"  {'Mode':<10} {'Persona':<35} {'Ground':>6} {'Att':>3} "
              f"{'Absorbed':>8} {'Rate':>6} {'Cost':>8}")
        print("  " + "-" * 85)

        for r in [clean, poisoned]:
            if r is None:
                continue
            if r.failed:
                print(f"  {r.mode:<10} {'FAILED':<35} {'—':>6} {r.attempts:>3} "
                      f"{'—':>8} {'—':>6} ${r.cost_usd:>7.4f}")
            else:
                print(f"  {r.mode:<10} {r.persona_name[:33]:<35} {r.groundedness_score:>6.2f} "
                      f"{r.attempts:>3} {len(r.absorbed_markers):>7}/{len(POISON_MARKERS)} "
                      f"{r.absorption_rate:>5.1%} ${r.cost_usd:>7.4f}")

        # Show what leaked
        if poisoned and not poisoned.failed and poisoned.absorbed_markers:
            print(f"\n  ABSORBED POISON MARKERS:")
            for marker in poisoned.absorbed_markers:
                print(f"    - \"{marker}\"")
            print(f"\n  ABSORPTION LOCATIONS:")
            for field_name, markers in poisoned.absorption_locations.items():
                print(f"    {field_name}: {', '.join(markers)}")

        if poisoned and not poisoned.failed and not poisoned.absorbed_markers:
            print(f"\n  NO POISON ABSORBED — pipeline fully resisted the adversarial injection.")

    # Aggregate
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)

    clean_results = [r for r in results if r.mode == "clean" and not r.failed]
    poisoned_results = [r for r in results if r.mode == "poisoned" and not r.failed]

    avg_clean_g = sum(r.groundedness_score for r in clean_results) / len(clean_results) if clean_results else 0
    avg_poison_g = sum(r.groundedness_score for r in poisoned_results) / len(poisoned_results) if poisoned_results else 0
    avg_absorption = sum(r.absorption_rate for r in poisoned_results) / len(poisoned_results) if poisoned_results else 0
    total_absorbed = sum(len(r.absorbed_markers) for r in poisoned_results)
    total_possible = len(POISON_MARKERS) * len(poisoned_results)

    print(f"\n  {'Metric':<35} {'Clean':>10} {'Poisoned':>10}")
    print("  " + "-" * 58)
    print(f"  {'Avg groundedness':<35} {avg_clean_g:>10.2f} {avg_poison_g:>10.2f}")
    print(f"  {'Avg attempts':<35} "
          f"{sum(r.attempts for r in clean_results) / len(clean_results) if clean_results else 0:>10.1f} "
          f"{sum(r.attempts for r in poisoned_results) / len(poisoned_results) if poisoned_results else 0:>10.1f}")
    print(f"  {'Success rate':<35} "
          f"{len(clean_results)}/{len([r for r in results if r.mode == 'clean']):>7} "
          f"{len(poisoned_results)}/{len([r for r in results if r.mode == 'poisoned']):>7}")
    print(f"\n  {'Absorption rate':<35} {'N/A':>10} {avg_absorption:>9.1%}")
    print(f"  {'Total markers absorbed':<35} {'N/A':>10} {total_absorbed}/{total_possible:>6}")

    print()
    print(f"  Poison records injected: {len(POISON_RECORDS)}")
    print(f"  Poison markers tracked: {len(POISON_MARKERS)}")
    print()

    # Show vocabulary comparison
    print("=" * 100)
    print("VOCABULARY COMPARISON (clean vs poisoned)")
    print("=" * 100)
    for base_cid in clusters:
        clean = next((r for r in results if r.cluster_id == base_cid and r.mode == "clean" and not r.failed), None)
        poisoned = next((r for r in results if r.cluster_id == base_cid + "_poisoned" and r.mode == "poisoned" and not r.failed), None)
        if clean and poisoned:
            print(f"\n  Cluster: {base_cid}")
            print(f"    Clean:    {', '.join(clean.vocabulary[:10])}")
            print(f"    Poisoned: {', '.join(poisoned.vocabulary[:10])}")
            new_in_poisoned = set(v.lower() for v in poisoned.vocabulary) - set(v.lower() for v in clean.vocabulary)
            if new_in_poisoned:
                print(f"    New in poisoned: {', '.join(sorted(new_in_poisoned))}")

    # Show goals comparison
    print("\n" + "=" * 100)
    print("GOALS COMPARISON (clean vs poisoned)")
    print("=" * 100)
    for base_cid in clusters:
        clean = next((r for r in results if r.cluster_id == base_cid and r.mode == "clean" and not r.failed), None)
        poisoned = next((r for r in results if r.cluster_id == base_cid + "_poisoned" and r.mode == "poisoned" and not r.failed), None)
        if clean and poisoned:
            print(f"\n  Cluster: {base_cid}")
            print(f"    Clean goals:")
            for g in clean.goals:
                print(f"      - {g}")
            print(f"    Poisoned goals:")
            for g in poisoned.goals:
                print(f"      - {g}")
    print()


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    print(f"Model: {settings.default_model}")
    print(f"Poison records: {len(POISON_RECORDS)}")
    print(f"Poison markers: {len(POISON_MARKERS)}")

    # Ingest and segment
    records = fetch_all(TENANT_ID)
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]
    print(f"Records: {len(raw)}")

    clusters_raw = segment(
        raw,
        tenant_industry="B2B SaaS",
        tenant_product="Project management tool for engineering teams",
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    print(f"Clusters: {len(clusters)}")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    results: list[AbsorptionResult] = []

    for cluster in clusters:
        # Clean run
        clean_result = await run_synthesis_and_scan(cluster, backend, "clean")
        results.append(clean_result)

        # Poisoned run
        poisoned_cluster = inject_poison(cluster, POISON_RECORDS)
        poisoned_result = await run_synthesis_and_scan(poisoned_cluster, backend, "poisoned")
        results.append(poisoned_result)

    print_results(results)

    # Save results
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "exp_3_07_results.json"
    data = []
    for r in results:
        data.append({
            "mode": r.mode,
            "cluster_id": r.cluster_id,
            "persona_name": r.persona_name,
            "groundedness_score": r.groundedness_score,
            "attempts": r.attempts,
            "cost_usd": r.cost_usd,
            "absorbed_markers": r.absorbed_markers,
            "absorption_rate": r.absorption_rate,
            "absorption_locations": r.absorption_locations,
            "goals": r.goals,
            "pains": r.pains,
            "vocabulary": r.vocabulary,
            "firmographics": r.firmographics,
            "failed": r.failed,
        })
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
