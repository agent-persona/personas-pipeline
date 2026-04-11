"""Experiment 2.06: Temperature sweep.

Hypothesis: Lower temperatures produce more grounded, schema-valid personas
with fewer retries, while higher temperatures increase distinctiveness
(vocabulary diversity) at the cost of groundedness.

Sweep grid (API forbids both params simultaneously, so swept independently):
  Phase 1: temperature ∈ {0.0, 0.4, 0.7, 1.0} (top_p=default)
  Phase 2: top_p ∈ {0.8, 0.9} (temperature=default)
  control = API defaults (no temperature/top_p specified)

Metrics:
  - Groundedness score
  - Schema validity / success rate
  - Retry rate (attempts > 1)
  - Cost per persona
  - Distinctiveness (vocabulary overlap heuristic)

Usage:
    python scripts/experiment_2_06.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))

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

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

# Sweep grid — API does not allow both temperature and top_p simultaneously,
# so we sweep each independently against the control.
TEMPERATURES = [0.0, 0.4, 0.7, 1.0]
TOP_PS = [0.8, 0.9]

# Build variant list: control + temperature sweep + top_p sweep
VARIANTS: list[dict] = [
    {"name": "control (API defaults)", "temperature": None, "top_p": None},
]
for temp in TEMPERATURES:
    VARIANTS.append({
        "name": f"temp={temp}",
        "temperature": temp,
        "top_p": None,
    })
for top_p in TOP_PS:
    VARIANTS.append({
        "name": f"top_p={top_p}",
        "temperature": None,
        "top_p": top_p,
    })
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-2.06-temperature-sweep"


# ── Metrics ───────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    variant_name: str
    temperature: float | None
    top_p: float | None
    cluster_id: str = ""
    persona_name: str = ""
    groundedness_score: float = 0.0
    schema_valid: bool = False
    cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0
    vocabulary: list[str] = field(default_factory=list)
    error: str = ""


def extract_vocabulary(persona_dict: dict) -> list[str]:
    """Extract meaningful words from persona for distinctiveness measurement."""
    words: list[str] = []
    text_fields = ["goals", "pains", "motivations", "objections",
                   "channels", "vocabulary", "decision_triggers", "sample_quotes"]
    for f in text_fields:
        items = persona_dict.get(f, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    words.extend(item.lower().split())
    return words


def compute_distinctiveness(results: list[VariantResult]) -> float:
    """Jaccard distance between vocabulary sets of personas in same variant group."""
    if len(results) < 2:
        return 0.0
    distances = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            set_a = set(results[i].vocabulary)
            set_b = set(results[j].vocabulary)
            if not set_a and not set_b:
                continue
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union > 0 else 0.0
            distances.append(1.0 - jaccard)  # distance = 1 - similarity
    return statistics.mean(distances) if distances else 0.0


# ── Pipeline ──────────────────────────────────────────────────────────

def get_clusters() -> list[ClusterData]:
    """Run ingest + segmentation."""
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
    return [ClusterData.model_validate(c) for c in cluster_dicts]


async def run_variant(
    variant: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
) -> VariantResult:
    """Run synthesis for one variant on one cluster."""
    result = VariantResult(
        variant_name=variant["name"],
        temperature=variant["temperature"],
        top_p=variant["top_p"],
        cluster_id=cluster.cluster_id,
    )

    backend = AnthropicBackend(
        client=client,
        model=settings.default_model,
        temperature=variant["temperature"],
        top_p=variant["top_p"],
    )

    t0 = time.monotonic()
    try:
        synth_result = await synthesize(cluster, backend)
        result.schema_valid = True
        result.groundedness_score = synth_result.groundedness.score
        result.cost_usd = synth_result.total_cost_usd
        result.attempts = synth_result.attempts
        result.persona_name = synth_result.persona.name
        result.vocabulary = extract_vocabulary(
            synth_result.persona.model_dump(mode="json")
        )
    except Exception as e:
        result.error = str(e)
        result.schema_valid = False
        print(f"    FAILED: {e}")

    result.duration_seconds = time.monotonic() - t0
    return result


# ── Reporting ─────────────────────────────────────────────────────────

def build_report(all_results: list[VariantResult]) -> str:
    """Generate comparison report."""
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 80)
    p("EXPERIMENT 2.06 — TEMPERATURE SWEEP — RESULTS")
    p("=" * 80)

    # Group by variant
    variant_groups: dict[str, list[VariantResult]] = {}
    for r in all_results:
        variant_groups.setdefault(r.variant_name, []).append(r)

    # Summary table
    p(f"\n{'Variant':<28} {'Ground':>7} {'Valid':>6} {'Retry%':>7} "
      f"{'Cost':>8} {'Distinct':>9} {'N':>3}")
    p("-" * 80)

    for vname, results in variant_groups.items():
        valid = [r for r in results if r.schema_valid]
        n = len(results)
        n_valid = len(valid)
        avg_ground = statistics.mean([r.groundedness_score for r in valid]) if valid else 0
        validity_rate = n_valid / n if n else 0
        retry_rate = sum(1 for r in valid if r.attempts > 1) / n_valid if n_valid else 0
        avg_cost = statistics.mean([r.cost_usd for r in valid]) if valid else 0
        distinctiveness = compute_distinctiveness(valid)

        p(f"{vname:<28} {avg_ground:>7.3f} {validity_rate:>5.0%} {retry_rate:>6.0%} "
          f"${avg_cost:>7.4f} {distinctiveness:>9.3f} {n:>3}")

    p("\n" + "=" * 80)

    # Detailed per-variant analysis
    p("\n── DETAILED ANALYSIS ──\n")

    control_results = variant_groups.get("control (API defaults)", [])
    ctrl_ground = statistics.mean(
        [r.groundedness_score for r in control_results if r.schema_valid]
    ) if any(r.schema_valid for r in control_results) else 0

    for vname, results in variant_groups.items():
        if vname == "control (API defaults)":
            continue
        valid = [r for r in results if r.schema_valid]
        if not valid:
            p(f"  {vname}: ALL FAILED")
            continue
        avg_ground = statistics.mean([r.groundedness_score for r in valid])
        delta = avg_ground - ctrl_ground
        direction = "BETTER" if delta > 0.02 else ("WORSE" if delta < -0.02 else "NEUTRAL")
        p(f"  {vname}: groundedness delta={delta:+.3f} ({direction})")

    p("\n" + "=" * 80)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 2.06: Temperature sweep")
    print("Hypothesis: Lower temp → better groundedness; higher temp → more diversity")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Shared input: ingest + segment once
    print("\n[1/3] Running shared ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    # Run all variants
    print("\n[2/3] Running sweep...")
    all_results: list[VariantResult] = []

    for variant in VARIANTS:
        print(f"\n  ── {variant['name']} ──")
        for cluster in clusters:
            print(f"    Cluster: {cluster.cluster_id}")
            r = await run_variant(variant, cluster, client)
            all_results.append(r)
            if r.schema_valid:
                print(f"      {r.persona_name}: groundedness={r.groundedness_score:.3f}, "
                      f"attempts={r.attempts}, cost=${r.cost_usd:.4f}")

    # Report
    print("\n[3/3] Comparing results...")
    report = build_report(all_results)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "2.06",
        "title": "Temperature sweep",
        "hypothesis": "Lower temp → better groundedness; higher temp → more diversity",
        "model": settings.default_model,
        "sweep_grid": {
            "temperatures": TEMPERATURES,
            "top_ps": TOP_PS,
        },
        "results": [
            {
                "variant": r.variant_name,
                "temperature": r.temperature,
                "top_p": r.top_p,
                "cluster_id": r.cluster_id,
                "persona_name": r.persona_name,
                "groundedness_score": r.groundedness_score,
                "schema_valid": r.schema_valid,
                "cost_usd": r.cost_usd,
                "attempts": r.attempts,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in all_results
        ],
    }

    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to: {results_path}")

    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text(report)
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
