"""Experiment 1.17: Length budgets per field.

Hypothesis: Tighter token budgets force the model toward higher-information
density and reduce hedged filler.

Variants:
  - control:  budget_multiplier=None   (unbounded)
  - tight:    budget_multiplier=0.4    (~20 tokens per field)
  - moderate: budget_multiplier=1.0    (~50 tokens per field)
  - relaxed:  budget_multiplier=4.0    (~200 tokens per field)

Metrics:
  - Information density   (content chars / total chars, excluding whitespace)
  - Hedged-filler rate    (fraction of list items containing hedge words)
  - Groundedness score    (from existing groundedness checker)
  - Schema validity       (Pydantic pass/fail)
  - Cost per persona      (USD)
  - Field lengths         (mean tokens per field across list fields)
  - Output token count    (total LLM output tokens)

Usage:
    python scripts/experiment_1_17.py
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import sys
import time
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

VARIANTS: dict[str, float | None] = {
    "control (unbounded)": None,
    "tight (~20 tok)": 0.4,
    "moderate (~50 tok)": 1.0,
    "relaxed (~200 tok)": 4.0,
}

# Words/phrases that indicate hedging or filler
HEDGE_PATTERNS = re.compile(
    r"\b("
    r"potentially|possibly|perhaps|maybe|might|could|may|generally|"
    r"tend to|likely|somewhat|relatively|arguably|in some cases|"
    r"it depends|more or less|to some extent|in general|often|"
    r"typically|usually|sometimes|occasionally|various|certain|"
    r"a number of|some degree|kind of|sort of|seems to|appears to"
    r")\b",
    re.IGNORECASE,
)

# List-type fields on PersonaV1 we measure
LIST_FIELDS = [
    "goals", "pains", "motivations", "objections",
    "channels", "vocabulary", "decision_triggers", "sample_quotes",
]


# ── Metrics ───────────────────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    variant: str
    budget_multiplier: float | None
    # Core experiment metrics
    info_density: float = 0.0           # non-whitespace chars / total chars
    hedge_filler_rate: float = 0.0      # fraction of list items with hedges
    hedge_count: int = 0                # total hedge instances
    total_list_items: int = 0           # total list items across fields
    # Field length stats
    mean_field_item_tokens: float = 0.0 # avg tokens per list item
    median_field_item_tokens: float = 0.0
    max_field_item_tokens: int = 0
    min_field_item_tokens: int = 0
    # Pipeline metrics
    groundedness_score: float = 0.0
    schema_valid: bool = False
    cost_usd: float = 0.0
    attempts: int = 0
    output_tokens: int = 0
    # Persona identity
    persona_name: str = ""
    cluster_id: str = ""
    # Timing
    duration_seconds: float = 0.0


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def compute_info_density(persona_dict: dict) -> float:
    """Ratio of non-whitespace chars to total chars in all string values."""
    total_chars = 0
    content_chars = 0
    for value in _extract_all_strings(persona_dict):
        total_chars += len(value)
        content_chars += len(value.replace(" ", "").replace("\n", "").replace("\t", ""))
    return content_chars / total_chars if total_chars > 0 else 0.0


def compute_hedge_rate(persona_dict: dict) -> tuple[float, int, int]:
    """Fraction of list-field items containing hedge words."""
    total_items = 0
    hedged_items = 0
    hedge_total = 0
    for field_name in LIST_FIELDS:
        items = persona_dict.get(field_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            text = item if isinstance(item, str) else json.dumps(item)
            total_items += 1
            matches = HEDGE_PATTERNS.findall(text)
            if matches:
                hedged_items += 1
                hedge_total += len(matches)
    rate = hedged_items / total_items if total_items > 0 else 0.0
    return rate, hedge_total, total_items


def compute_field_length_stats(persona_dict: dict) -> dict:
    """Token-length stats for list-field items."""
    all_lengths: list[int] = []
    for field_name in LIST_FIELDS:
        items = persona_dict.get(field_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            text = item if isinstance(item, str) else json.dumps(item)
            all_lengths.append(estimate_tokens(text))
    if not all_lengths:
        return {"mean": 0, "median": 0, "max": 0, "min": 0}
    return {
        "mean": statistics.mean(all_lengths),
        "median": statistics.median(all_lengths),
        "max": max(all_lengths),
        "min": min(all_lengths),
    }


def _extract_all_strings(obj, depth=0) -> list[str]:
    """Recursively extract all string values from a dict/list structure."""
    if depth > 10:
        return []
    strings = []
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            strings.extend(_extract_all_strings(v, depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            strings.extend(_extract_all_strings(v, depth + 1))
    return strings


# ── Pipeline ──────────────────────────────────────────────────────────

def get_clusters() -> list[ClusterData]:
    """Run ingest + segmentation (shared across all variants)."""
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
    budget_multiplier: float | None,
    cluster: ClusterData,
    backend: AnthropicBackend,
) -> ExperimentMetrics:
    """Run synthesis for one variant on one cluster, collect all metrics."""
    metrics = ExperimentMetrics(
        variant=variant_name,
        budget_multiplier=budget_multiplier,
        cluster_id=cluster.cluster_id,
    )

    t0 = time.monotonic()
    try:
        result = await synthesize(
            cluster, backend, budget_multiplier=budget_multiplier,
        )
        metrics.schema_valid = True
        metrics.groundedness_score = result.groundedness.score
        metrics.cost_usd = result.total_cost_usd
        metrics.attempts = result.attempts
        metrics.persona_name = result.persona.name

        persona_dict = result.persona.model_dump(mode="json")

        # Information density
        metrics.info_density = compute_info_density(persona_dict)

        # Hedge/filler rate
        rate, hedge_count, total_items = compute_hedge_rate(persona_dict)
        metrics.hedge_filler_rate = rate
        metrics.hedge_count = hedge_count
        metrics.total_list_items = total_items

        # Field length stats
        fstats = compute_field_length_stats(persona_dict)
        metrics.mean_field_item_tokens = fstats["mean"]
        metrics.median_field_item_tokens = fstats["median"]
        metrics.max_field_item_tokens = fstats["max"]
        metrics.min_field_item_tokens = fstats["min"]

    except Exception as e:
        print(f"    FAILED: {e}")
        metrics.schema_valid = False

    metrics.duration_seconds = time.monotonic() - t0
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────

def print_comparison(all_metrics: list[ExperimentMetrics]) -> str:
    """Print and return a formatted comparison table."""
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 90)
    p("EXPERIMENT 1.17 — LENGTH BUDGETS PER FIELD — RESULTS")
    p("=" * 90)

    # Build headers from unique variants in order
    seen = []
    for m in all_metrics:
        if m.variant not in seen:
            seen.append(m.variant)

    header = f"{'Metric':<30}"
    for v in seen:
        header += f"{v:>22}"
    p(header)
    p("-" * (30 + 22 * len(seen)))

    def row(label, getter, fmt=".3f"):
        vals = {}
        for m in all_metrics:
            vals.setdefault(m.variant, []).append(getter(m))
        line = f"{label:<30}"
        for v in seen:
            avg = statistics.mean(vals[v]) if vals[v] else 0
            line += f"{avg:>22{fmt}}"
        p(line)

    row("Info density",          lambda m: m.info_density)
    row("Hedge-filler rate",     lambda m: m.hedge_filler_rate)
    row("Hedge count",           lambda m: m.hedge_count, fmt=".1f")
    row("Total list items",      lambda m: m.total_list_items, fmt=".0f")
    row("Mean tokens/item",      lambda m: m.mean_field_item_tokens)
    row("Median tokens/item",    lambda m: m.median_field_item_tokens)
    row("Max tokens/item",       lambda m: m.max_field_item_tokens, fmt=".0f")
    row("Groundedness score",    lambda m: m.groundedness_score)
    row("Cost (USD)",            lambda m: m.cost_usd, fmt=".4f")
    row("Attempts",              lambda m: m.attempts, fmt=".1f")
    row("Duration (s)",          lambda m: m.duration_seconds, fmt=".1f")

    p("-" * (30 + 18 * len(seen)))

    # Signal strength assessment
    p("\n── SIGNAL ASSESSMENT ──")

    control_densities = [m.info_density for m in all_metrics if m.budget_multiplier is None]
    control_hedges = [m.hedge_filler_rate for m in all_metrics if m.budget_multiplier is None]
    control_tokens = [m.mean_field_item_tokens for m in all_metrics if m.budget_multiplier is None]

    if control_densities:
        ctrl_density = statistics.mean(control_densities)
        ctrl_hedge = statistics.mean(control_hedges)
        ctrl_tokens_mean = statistics.mean(control_tokens)

        for variant_name in seen:
            if "control" in variant_name:
                continue
            v_densities = [m.info_density for m in all_metrics if m.variant == variant_name]
            v_hedges = [m.hedge_filler_rate for m in all_metrics if m.variant == variant_name]
            v_tokens = [m.mean_field_item_tokens for m in all_metrics if m.variant == variant_name]

            if not v_densities:
                continue

            d_density = statistics.mean(v_densities) - ctrl_density
            d_hedge = statistics.mean(v_hedges) - ctrl_hedge
            d_tokens = statistics.mean(v_tokens) - ctrl_tokens_mean

            density_signal = "BETTER" if d_density > 0.005 else ("WORSE" if d_density < -0.005 else "NEUTRAL")
            hedge_signal = "BETTER" if d_hedge < -0.03 else ("WORSE" if d_hedge > 0.03 else "NEUTRAL")
            tokens_signal = "SHORTER" if d_tokens < -1 else ("LONGER" if d_tokens > 1 else "SIMILAR")

            signals = [density_signal != "NEUTRAL", hedge_signal != "NEUTRAL", tokens_signal != "SIMILAR"]
            strength = "STRONG" if sum(signals) >= 2 else ("MODERATE" if sum(signals) == 1 else "WEAK")

            p(f"\n  {variant_name}:")
            p(f"    Info density:    {d_density:+.4f} ({density_signal})")
            p(f"    Hedge rate:      {d_hedge:+.4f} ({hedge_signal})")
            p(f"    Tokens/item:     {d_tokens:+.1f} ({tokens_signal})")
            p(f"    Signal strength: {strength}")

    p("\n" + "=" * 90)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 1.17: Length budgets per field")
    print("Hypothesis: Tighter budgets → higher info density, less hedging")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Shared input: ingest + segment once
    print("\n[1/3] Running shared ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    # Run all variants
    print("\n[2/3] Running variants...")
    all_metrics: list[ExperimentMetrics] = []

    for variant_name, multiplier in VARIANTS.items():
        print(f"\n  ── {variant_name} (multiplier={multiplier}) ──")
        for cluster in clusters:
            print(f"    Cluster: {cluster.cluster_id}")
            m = await run_variant(variant_name, multiplier, cluster, backend)
            all_metrics.append(m)
            if m.schema_valid:
                print(f"      {m.persona_name}: density={m.info_density:.3f}, "
                      f"hedge_rate={m.hedge_filler_rate:.3f}, "
                      f"mean_tok={m.mean_field_item_tokens:.1f}, "
                      f"cost=${m.cost_usd:.4f}")

    # Report
    print("\n[3/3] Comparing results...")
    report = print_comparison(all_metrics)

    # Save results
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "exp_1_17_results.json"
    results_data = {
        "experiment": "1.17",
        "title": "Length budgets per field",
        "hypothesis": "Tighter token budgets force higher-information density and reduce hedged filler",
        "model": settings.default_model,
        "variants": {k: v for k, v in VARIANTS.items()},
        "metrics": [
            {
                "variant": m.variant,
                "budget_multiplier": m.budget_multiplier,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "info_density": m.info_density,
                "hedge_filler_rate": m.hedge_filler_rate,
                "hedge_count": m.hedge_count,
                "total_list_items": m.total_list_items,
                "mean_field_item_tokens": m.mean_field_item_tokens,
                "median_field_item_tokens": m.median_field_item_tokens,
                "max_field_item_tokens": m.max_field_item_tokens,
                "min_field_item_tokens": m.min_field_item_tokens,
                "groundedness_score": m.groundedness_score,
                "schema_valid": m.schema_valid,
                "cost_usd": m.cost_usd,
                "attempts": m.attempts,
                "duration_seconds": m.duration_seconds,
            }
            for m in all_metrics
        ],
    }
    results_path.write_text(json.dumps(results_data, indent=2))

    report_path = output_dir / "exp_1_17_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
