"""Experiment 1.19: Schema artifact format.

Hypothesis: The artifact format the model sees (Pydantic, JSON Schema,
TypeScript) changes what it produces, even when the fields are identical.

Variants:
  - control:    schema_format=None      (no schema in system prompt)
  - pydantic:   schema_format="pydantic" (Python class definition)
  - jsonschema: schema_format="jsonschema" (JSON Schema)
  - typescript: schema_format="typescript" (TypeScript interfaces)

All variants use the same tool schema (PersonaV1.model_json_schema()).
Only the system prompt description changes.

Metrics:
  - Field output variance across formats (per-field text similarity)
  - Schema validity (Pydantic pass/fail)
  - Groundedness score
  - Cost per persona
  - Field lengths (mean tokens per field)
  - List item counts per field
  - Vocabulary overlap across variants (Jaccard similarity)

Usage:
    python scripts/experiment_1_19.py
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

VARIANTS: dict[str, str | None] = {
    "control (no schema)": None,
    "pydantic": "pydantic",
    "jsonschema": "jsonschema",
    "typescript": "typescript",
}

LIST_FIELDS = [
    "goals", "pains", "motivations", "objections",
    "channels", "vocabulary", "decision_triggers", "sample_quotes",
]

SCALAR_FIELDS = ["name", "summary"]


# ── Metrics ───────────────────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    variant: str
    schema_format: str | None
    cluster_id: str = ""
    persona_name: str = ""
    # Pipeline metrics
    groundedness_score: float = 0.0
    schema_valid: bool = False
    cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0
    # Field analysis
    field_item_counts: dict = field(default_factory=dict)    # field -> count
    field_mean_tokens: dict = field(default_factory=dict)    # field -> mean token len
    summary_length: int = 0
    total_list_items: int = 0
    mean_tokens_per_item: float = 0.0
    # Raw persona for cross-variant comparison
    persona_dict: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def analyze_persona(persona_dict: dict) -> dict:
    """Extract field-level stats from a persona dict."""
    item_counts = {}
    mean_tokens = {}
    all_token_lengths = []

    for field_name in LIST_FIELDS:
        items = persona_dict.get(field_name, [])
        if not isinstance(items, list):
            continue
        item_counts[field_name] = len(items)
        lengths = [estimate_tokens(str(item)) for item in items]
        mean_tokens[field_name] = statistics.mean(lengths) if lengths else 0
        all_token_lengths.extend(lengths)

    return {
        "item_counts": item_counts,
        "mean_tokens": mean_tokens,
        "summary_length": estimate_tokens(persona_dict.get("summary", "")),
        "total_items": sum(item_counts.values()),
        "mean_tokens_per_item": (
            statistics.mean(all_token_lengths) if all_token_lengths else 0
        ),
    }


def compute_cross_variant_similarity(
    metrics_by_variant: dict[str, list[ExperimentMetrics]],
    cluster_id: str,
) -> dict:
    """Compute pairwise Jaccard similarity of list fields across variants."""
    # Collect persona dicts for this cluster across variants
    personas = {}
    for variant, mlist in metrics_by_variant.items():
        for m in mlist:
            if m.cluster_id == cluster_id and m.schema_valid:
                personas[variant] = m.persona_dict
                break

    if len(personas) < 2:
        return {}

    variant_names = list(personas.keys())
    results = {}

    for field_name in LIST_FIELDS:
        field_results = {}
        for i, v1 in enumerate(variant_names):
            for v2 in variant_names[i + 1:]:
                items1 = set(str(x).lower() for x in personas[v1].get(field_name, []))
                items2 = set(str(x).lower() for x in personas[v2].get(field_name, []))
                sim = jaccard_similarity(items1, items2)
                field_results[f"{v1} vs {v2}"] = sim
        results[field_name] = field_results

    # Also compute name/summary similarity via word overlap
    for field_name in SCALAR_FIELDS:
        field_results = {}
        for i, v1 in enumerate(variant_names):
            for v2 in variant_names[i + 1:]:
                words1 = set(str(personas[v1].get(field_name, "")).lower().split())
                words2 = set(str(personas[v2].get(field_name, "")).lower().split())
                sim = jaccard_similarity(words1, words2)
                field_results[f"{v1} vs {v2}"] = sim
        results[field_name] = field_results

    return results


# ── Pipeline ──────────────────────────────────────────────────────────

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
    return [ClusterData.model_validate(c) for c in cluster_dicts]


async def run_variant(
    variant_name: str,
    schema_format: str | None,
    cluster: ClusterData,
    backend: AnthropicBackend,
) -> ExperimentMetrics:
    metrics = ExperimentMetrics(
        variant=variant_name,
        schema_format=schema_format,
        cluster_id=cluster.cluster_id,
    )

    t0 = time.monotonic()
    try:
        result = await synthesize(
            cluster, backend, schema_format=schema_format,
        )
        metrics.schema_valid = True
        metrics.groundedness_score = result.groundedness.score
        metrics.cost_usd = result.total_cost_usd
        metrics.attempts = result.attempts
        metrics.persona_name = result.persona.name

        persona_dict = result.persona.model_dump(mode="json")
        metrics.persona_dict = persona_dict

        analysis = analyze_persona(persona_dict)
        metrics.field_item_counts = analysis["item_counts"]
        metrics.field_mean_tokens = analysis["mean_tokens"]
        metrics.summary_length = analysis["summary_length"]
        metrics.total_list_items = analysis["total_items"]
        metrics.mean_tokens_per_item = analysis["mean_tokens_per_item"]

    except Exception as e:
        print(f"    FAILED: {e}")
        metrics.schema_valid = False

    metrics.duration_seconds = time.monotonic() - t0
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────

def print_comparison(all_metrics: list[ExperimentMetrics], clusters: list[ClusterData]) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    seen = []
    for m in all_metrics:
        if m.variant not in seen:
            seen.append(m.variant)

    # Group by variant
    by_variant: dict[str, list[ExperimentMetrics]] = {}
    for m in all_metrics:
        by_variant.setdefault(m.variant, []).append(m)

    p("\n" + "=" * 100)
    p("EXPERIMENT 1.19 — SCHEMA ARTIFACT FORMAT — RESULTS")
    p("=" * 100)

    # ── Summary table ──
    header = f"{'Metric':<30}"
    for v in seen:
        header += f"{v:>22}"
    p(header)
    p("-" * (30 + 22 * len(seen)))

    def row(label, getter, fmt=".3f"):
        line = f"{label:<30}"
        for v in seen:
            valid = [m for m in by_variant[v] if m.schema_valid]
            if valid:
                avg = statistics.mean([getter(m) for m in valid])
                line += f"{avg:>22{fmt}}"
            else:
                line += f"{'FAILED':>22}"
        p(line)

    row("Groundedness",        lambda m: m.groundedness_score)
    row("Cost (USD)",          lambda m: m.cost_usd, fmt=".4f")
    row("Attempts",            lambda m: m.attempts, fmt=".1f")
    row("Duration (s)",        lambda m: m.duration_seconds, fmt=".1f")
    row("Total list items",    lambda m: m.total_list_items, fmt=".0f")
    row("Mean tokens/item",    lambda m: m.mean_tokens_per_item)
    row("Summary length (tok)", lambda m: m.summary_length, fmt=".0f")
    row("Success rate",        lambda m: 1.0 if m.schema_valid else 0.0)

    p("-" * (30 + 22 * len(seen)))

    # ── Per-field item counts ──
    p("\n── FIELD ITEM COUNTS (avg across clusters) ──")
    header2 = f"{'Field':<25}"
    for v in seen:
        header2 += f"{v:>22}"
    p(header2)
    p("-" * (25 + 22 * len(seen)))

    for field_name in LIST_FIELDS:
        line = f"{field_name:<25}"
        for v in seen:
            valid = [m for m in by_variant[v] if m.schema_valid]
            if valid:
                avg = statistics.mean([m.field_item_counts.get(field_name, 0) for m in valid])
                line += f"{avg:>22.1f}"
            else:
                line += f"{'N/A':>22}"
        p(line)

    # ── Per-field mean token lengths ──
    p("\n── FIELD MEAN TOKENS/ITEM (avg across clusters) ──")
    p(header2)
    p("-" * (25 + 22 * len(seen)))

    for field_name in LIST_FIELDS:
        line = f"{field_name:<25}"
        for v in seen:
            valid = [m for m in by_variant[v] if m.schema_valid]
            if valid:
                avg = statistics.mean([m.field_mean_tokens.get(field_name, 0) for m in valid])
                line += f"{avg:>22.1f}"
            else:
                line += f"{'N/A':>22}"
        p(line)

    # ── Cross-variant similarity ──
    p("\n── CROSS-VARIANT FIELD SIMILARITY (Jaccard) ──")
    for cluster in clusters:
        p(f"\n  Cluster: {cluster.cluster_id}")
        sim = compute_cross_variant_similarity(by_variant, cluster.cluster_id)
        if not sim:
            p("    (insufficient data)")
            continue
        for field_name in LIST_FIELDS + SCALAR_FIELDS:
            if field_name not in sim:
                continue
            p(f"    {field_name}:")
            for pair, score in sim[field_name].items():
                signal = "HIGH" if score > 0.5 else ("MODERATE" if score > 0.2 else "LOW")
                p(f"      {pair}: {score:.3f} ({signal})")

    # ── Signal assessment ──
    p("\n── SIGNAL ASSESSMENT ──")

    ctrl_metrics = [m for m in all_metrics if m.schema_format is None and m.schema_valid]
    if ctrl_metrics:
        ctrl_cost = statistics.mean([m.cost_usd for m in ctrl_metrics])
        ctrl_ground = statistics.mean([m.groundedness_score for m in ctrl_metrics])
        ctrl_items = statistics.mean([m.total_list_items for m in ctrl_metrics])
        ctrl_tok = statistics.mean([m.mean_tokens_per_item for m in ctrl_metrics])

        for variant_name in seen:
            if "control" in variant_name:
                continue
            v_metrics = [m for m in all_metrics if m.variant == variant_name and m.schema_valid]
            if not v_metrics:
                p(f"\n  {variant_name}: ALL FAILED")
                continue

            v_cost = statistics.mean([m.cost_usd for m in v_metrics])
            v_ground = statistics.mean([m.groundedness_score for m in v_metrics])
            v_items = statistics.mean([m.total_list_items for m in v_metrics])
            v_tok = statistics.mean([m.mean_tokens_per_item for m in v_metrics])

            d_cost = v_cost - ctrl_cost
            d_ground = v_ground - ctrl_ground
            d_items = v_items - ctrl_items
            d_tok = v_tok - ctrl_tok

            # Compute average cross-variant similarity vs control
            avg_sims = []
            for cluster in clusters:
                sim = compute_cross_variant_similarity(by_variant, cluster.cluster_id)
                for field_name in LIST_FIELDS:
                    for pair, score in sim.get(field_name, {}).items():
                        if "control" in pair and variant_name in pair:
                            avg_sims.append(score)

            avg_sim = statistics.mean(avg_sims) if avg_sims else 0

            # Determine if format causes meaningful output variance
            variance_signal = (
                "HIGH VARIANCE" if avg_sim < 0.3
                else ("MODERATE VARIANCE" if avg_sim < 0.5 else "LOW VARIANCE")
            )

            signals = []
            if abs(d_ground) > 0.03:
                signals.append("groundedness")
            if abs(d_items) > 3:
                signals.append("item_count")
            if abs(d_tok) > 2:
                signals.append("verbosity")
            if avg_sim < 0.4:
                signals.append("content_divergence")

            strength = (
                "STRONG" if len(signals) >= 2
                else ("MODERATE" if len(signals) == 1 else "WEAK")
            )

            p(f"\n  {variant_name}:")
            p(f"    Groundedness:      {d_ground:+.4f}")
            p(f"    Cost:              {d_cost:+.4f}")
            p(f"    List items:        {d_items:+.1f}")
            p(f"    Tokens/item:       {d_tok:+.1f}")
            p(f"    Avg similarity to ctrl: {avg_sim:.3f} ({variance_signal})")
            p(f"    Signal strength:   {strength}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 1.19: Schema artifact format")
    print("Hypothesis: Schema format (Pydantic/JSON Schema/TS) changes output")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/3] Running shared ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    print("\n[2/3] Running variants...")
    all_metrics: list[ExperimentMetrics] = []

    for variant_name, fmt in VARIANTS.items():
        print(f"\n  ── {variant_name} (format={fmt}) ──")
        for cluster in clusters:
            print(f"    Cluster: {cluster.cluster_id}")
            m = await run_variant(variant_name, fmt, cluster, backend)
            all_metrics.append(m)
            if m.schema_valid:
                print(f"      {m.persona_name}: ground={m.groundedness_score:.3f}, "
                      f"items={m.total_list_items}, "
                      f"tok/item={m.mean_tokens_per_item:.1f}, "
                      f"cost=${m.cost_usd:.4f}")

    print("\n[3/3] Comparing results...")
    report = print_comparison(all_metrics, clusters)

    # Save results
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "exp_1_19_results.json"
    results_data = {
        "experiment": "1.19",
        "title": "Schema artifact format",
        "hypothesis": "Schema description format influences synthesis output even when fields are identical",
        "model": settings.default_model,
        "variants": {k: v for k, v in VARIANTS.items()},
        "metrics": [
            {
                "variant": m.variant,
                "schema_format": m.schema_format,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "groundedness_score": m.groundedness_score,
                "schema_valid": m.schema_valid,
                "cost_usd": m.cost_usd,
                "attempts": m.attempts,
                "duration_seconds": m.duration_seconds,
                "field_item_counts": m.field_item_counts,
                "field_mean_tokens": m.field_mean_tokens,
                "summary_length": m.summary_length,
                "total_list_items": m.total_list_items,
                "mean_tokens_per_item": m.mean_tokens_per_item,
            }
            for m in all_metrics
        ],
    }
    results_path.write_text(json.dumps(results_data, indent=2))

    report_path = output_dir / "exp_1_19_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
