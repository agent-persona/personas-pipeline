"""Experiment 3.03: Retrieval-augmented synthesis.

Hypothesis: Per-section top-k record retrieval improves groundedness
and reduces cost compared to dumping all records in context.

Variants:
  - control:  retrieval_k=None  (all records, flat list — current behavior)
  - k=3:      retrieval_k=3     (top-3 records per section)
  - k=10:     retrieval_k=10    (top-10 records per section)
  - k=30:     retrieval_k=30    (top-30; likely >= all records for mock data)

Metrics:
  - Groundedness score
  - Long-tail hallucination rate (claims with confidence < 0.7)
  - Cost per persona
  - Prompt token count (input tokens)
  - Evidence coverage (fraction of record IDs cited in source_evidence)
  - Unique records cited

Usage:
    python scripts/experiment_3_03.py
"""

from __future__ import annotations

import asyncio
import json
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

VARIANTS: dict[str, int | None] = {
    "control (all records)": None,
    "k=3": 3,
    "k=10": 10,
    "k=30": 30,
}


# ── Metrics ───────────────────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    variant: str
    retrieval_k: int | None
    cluster_id: str = ""
    persona_name: str = ""
    # Core metrics
    groundedness_score: float = 0.0
    schema_valid: bool = False
    cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0
    # Hallucination / evidence metrics
    longtail_hallucination_rate: float = 0.0  # evidence entries with confidence < 0.7
    total_evidence_entries: int = 0
    low_confidence_entries: int = 0
    unique_records_cited: int = 0
    total_records_available: int = 0
    evidence_coverage: float = 0.0  # fraction of available records cited
    mean_confidence: float = 0.0
    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0


def analyze_evidence(persona_dict: dict, available_record_ids: list[str]) -> dict:
    """Analyze evidence quality and coverage."""
    evidence = persona_dict.get("source_evidence", [])
    total = len(evidence)
    low_conf = sum(1 for e in evidence if e.get("confidence", 1.0) < 0.7)
    confidences = [e.get("confidence", 1.0) for e in evidence]

    cited_ids: set[str] = set()
    for e in evidence:
        cited_ids.update(e.get("record_ids", []))

    valid_cited = cited_ids & set(available_record_ids)

    return {
        "total_entries": total,
        "low_confidence": low_conf,
        "longtail_hallucination_rate": low_conf / total if total > 0 else 0.0,
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
        "unique_cited": len(valid_cited),
        "total_available": len(available_record_ids),
        "coverage": len(valid_cited) / len(available_record_ids) if available_record_ids else 0.0,
    }


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
    retrieval_k: int | None,
    cluster: ClusterData,
    backend: AnthropicBackend,
) -> ExperimentMetrics:
    metrics = ExperimentMetrics(
        variant=variant_name,
        retrieval_k=retrieval_k,
        cluster_id=cluster.cluster_id,
        total_records_available=len(cluster.all_record_ids),
    )

    t0 = time.monotonic()
    try:
        result = await synthesize(cluster, backend, retrieval_k=retrieval_k)
        metrics.schema_valid = True
        metrics.groundedness_score = result.groundedness.score
        metrics.cost_usd = result.total_cost_usd
        metrics.attempts = result.attempts
        metrics.persona_name = result.persona.name

        persona_dict = result.persona.model_dump(mode="json")
        ev = analyze_evidence(persona_dict, cluster.all_record_ids)
        metrics.longtail_hallucination_rate = ev["longtail_hallucination_rate"]
        metrics.total_evidence_entries = ev["total_entries"]
        metrics.low_confidence_entries = ev["low_confidence"]
        metrics.unique_records_cited = ev["unique_cited"]
        metrics.evidence_coverage = ev["coverage"]
        metrics.mean_confidence = ev["mean_confidence"]

    except Exception as e:
        print(f"    FAILED: {e}")
        metrics.schema_valid = False

    metrics.duration_seconds = time.monotonic() - t0
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────

def print_comparison(all_metrics: list[ExperimentMetrics]) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    seen = []
    for m in all_metrics:
        if m.variant not in seen:
            seen.append(m.variant)

    by_variant: dict[str, list[ExperimentMetrics]] = {}
    for m in all_metrics:
        by_variant.setdefault(m.variant, []).append(m)

    p("\n" + "=" * 100)
    p("EXPERIMENT 3.03 — RETRIEVAL-AUGMENTED SYNTHESIS — RESULTS")
    p("=" * 100)

    header = f"{'Metric':<35}"
    for v in seen:
        header += f"{v:>20}"
    p(header)
    p("-" * (35 + 20 * len(seen)))

    def row(label, getter, fmt=".3f"):
        line = f"{label:<35}"
        for v in seen:
            valid = [m for m in by_variant[v] if m.schema_valid]
            if valid:
                avg = statistics.mean([getter(m) for m in valid])
                line += f"{avg:>20{fmt}}"
            else:
                line += f"{'FAILED':>20}"
        p(line)

    row("Groundedness",                 lambda m: m.groundedness_score)
    row("Long-tail hallucination rate", lambda m: m.longtail_hallucination_rate)
    row("Mean evidence confidence",     lambda m: m.mean_confidence)
    row("Evidence entries",             lambda m: m.total_evidence_entries, fmt=".1f")
    row("Low-confidence entries",       lambda m: m.low_confidence_entries, fmt=".1f")
    row("Unique records cited",         lambda m: m.unique_records_cited, fmt=".1f")
    row("Evidence coverage",            lambda m: m.evidence_coverage)
    row("Cost (USD)",                   lambda m: m.cost_usd, fmt=".4f")
    row("Attempts",                     lambda m: m.attempts, fmt=".1f")
    row("Duration (s)",                 lambda m: m.duration_seconds, fmt=".1f")

    p("-" * (35 + 20 * len(seen)))

    # Signal assessment
    p("\n── SIGNAL ASSESSMENT ──")
    ctrl = [m for m in all_metrics if m.retrieval_k is None and m.schema_valid]
    if ctrl:
        ctrl_ground = statistics.mean([m.groundedness_score for m in ctrl])
        ctrl_halluc = statistics.mean([m.longtail_hallucination_rate for m in ctrl])
        ctrl_cost = statistics.mean([m.cost_usd for m in ctrl])
        ctrl_coverage = statistics.mean([m.evidence_coverage for m in ctrl])
        ctrl_conf = statistics.mean([m.mean_confidence for m in ctrl])

        for vname in seen:
            if "control" in vname:
                continue
            v = [m for m in by_variant[vname] if m.schema_valid]
            if not v:
                p(f"\n  {vname}: ALL FAILED")
                continue

            d_ground = statistics.mean([m.groundedness_score for m in v]) - ctrl_ground
            d_halluc = statistics.mean([m.longtail_hallucination_rate for m in v]) - ctrl_halluc
            d_cost = statistics.mean([m.cost_usd for m in v]) - ctrl_cost
            d_coverage = statistics.mean([m.evidence_coverage for m in v]) - ctrl_coverage
            d_conf = statistics.mean([m.mean_confidence for m in v]) - ctrl_conf

            ground_sig = "BETTER" if d_ground > 0.02 else ("WORSE" if d_ground < -0.02 else "SAME")
            halluc_sig = "BETTER" if d_halluc < -0.03 else ("WORSE" if d_halluc > 0.03 else "SAME")
            cost_sig = "CHEAPER" if d_cost < -0.003 else ("COSTLIER" if d_cost > 0.003 else "SAME")
            coverage_sig = "HIGHER" if d_coverage > 0.05 else ("LOWER" if d_coverage < -0.05 else "SAME")

            signals = []
            if ground_sig != "SAME": signals.append(ground_sig.lower() + " groundedness")
            if halluc_sig != "SAME": signals.append(halluc_sig.lower() + " hallucination")
            if cost_sig != "SAME": signals.append(cost_sig.lower())
            if coverage_sig != "SAME": signals.append(coverage_sig.lower() + " coverage")

            strength = "STRONG" if len(signals) >= 2 else ("MODERATE" if len(signals) == 1 else "WEAK")

            p(f"\n  {vname}:")
            p(f"    Groundedness:     {d_ground:+.4f} ({ground_sig})")
            p(f"    Hallucination:    {d_halluc:+.4f} ({halluc_sig})")
            p(f"    Cost:             {d_cost:+.4f} ({cost_sig})")
            p(f"    Coverage:         {d_coverage:+.4f} ({coverage_sig})")
            p(f"    Confidence:       {d_conf:+.4f}")
            p(f"    Signal strength:  {strength}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 3.03: Retrieval-augmented synthesis")
    print("Hypothesis: Top-k retrieval → better groundedness, lower cost")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/3] Running shared ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")
    for c in clusters:
        print(f"        {c.cluster_id}: {len(c.sample_records)} records")

    print("\n[2/3] Running variants...")
    all_metrics: list[ExperimentMetrics] = []

    for variant_name, k in VARIANTS.items():
        print(f"\n  ── {variant_name} (k={k}) ──")
        for cluster in clusters:
            print(f"    Cluster: {cluster.cluster_id}")
            m = await run_variant(variant_name, k, cluster, backend)
            all_metrics.append(m)
            if m.schema_valid:
                print(f"      {m.persona_name}: ground={m.groundedness_score:.3f}, "
                      f"halluc={m.longtail_hallucination_rate:.3f}, "
                      f"cited={m.unique_records_cited}/{m.total_records_available}, "
                      f"cost=${m.cost_usd:.4f}")

    print("\n[3/3] Comparing results...")
    report = print_comparison(all_metrics)

    # Save
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "3.03",
        "title": "Retrieval-augmented synthesis",
        "hypothesis": "Per-section top-k retrieval improves groundedness and reduces cost",
        "model": settings.default_model,
        "variants": {k_name: v for k_name, v in VARIANTS.items()},
        "metrics": [
            {
                "variant": m.variant,
                "retrieval_k": m.retrieval_k,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "groundedness_score": m.groundedness_score,
                "schema_valid": m.schema_valid,
                "cost_usd": m.cost_usd,
                "attempts": m.attempts,
                "longtail_hallucination_rate": m.longtail_hallucination_rate,
                "mean_confidence": m.mean_confidence,
                "total_evidence_entries": m.total_evidence_entries,
                "low_confidence_entries": m.low_confidence_entries,
                "unique_records_cited": m.unique_records_cited,
                "total_records_available": m.total_records_available,
                "evidence_coverage": m.evidence_coverage,
                "duration_seconds": m.duration_seconds,
            }
            for m in all_metrics
        ],
    }

    results_path = output_dir / "exp_3_03_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_3_03_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
