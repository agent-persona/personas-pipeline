"""Experiment 3.04: Evidence-first generation.

Hypothesis: Pre-selecting evidence before writing the persona forces grounding,
prevents demographic hallucination, and anchors voice in authentic customer
language — producing higher groundedness and voice fidelity than single-pass.

Variants:
  - control:        Standard single-pass synthesis
  - evidence-first: Two-pass (select evidence, then conditioned synthesis)

Metrics:
  - Groundedness score
  - Voice fidelity (LLM judge)
  - Demographic hallucination rate (demographics claims without evidence)
  - Evidence coverage (fraction of evidence categories used)
  - Cost overhead

Usage:
    python scripts/experiment_3_04.py
"""

from __future__ import annotations

import asyncio
import json
import math
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
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.engine.evidence_first import evidence_first_synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

JUDGE_DIMENSIONS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")

JUDGE_PROMPT = """\
You are an expert evaluator of synthetic marketing personas. Score the \
persona below on each dimension using a 0.0-1.0 scale.

- **grounded**: Claims traceable to source data with evidence entries.
- **distinctive**: Feels like a real individual, not a generic average.
- **coherent**: Internally consistent across all fields.
- **actionable**: Goals/pains specific enough to drive product decisions.
- **voice_fidelity**: Quotes and vocabulary sound like one consistent speaker.

Respond with ONLY a JSON object:
{"grounded":<float>,"distinctive":<float>,"coherent":<float>,\
"actionable":<float>,"voice_fidelity":<float>,\
"overall":<float>,"rationale":"<1-2 sentences>"}

## Persona to evaluate

"""


# ── Metrics ──────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    variant: str
    cluster_id: str = ""
    persona_name: str = ""
    success: bool = False
    # Groundedness
    groundedness_score: float = 0.0
    evidence_count: int = 0
    avg_confidence: float = 0.0
    # Evidence-first specifics
    selected_records: int = 0
    evidence_categories: int = 0
    # Demographics
    demo_hallucination_rate: float = 0.0
    # Judge
    judge_overall: float = float("nan")
    judge_voice_fidelity: float = float("nan")
    judge_dimensions: dict = field(default_factory=dict)
    judge_rationale: str = ""
    # Cost
    selection_cost_usd: float = 0.0
    synthesis_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0


def compute_demo_hallucination_rate(persona_dict: dict) -> float:
    """Fraction of demographic fields that look fabricated (specific but unevidenced).

    A 'hallucinated' demographic is one with a specific value (not 'unknown' or
    generic) that has no corresponding source_evidence entry with 'demographic'
    in the field_path or claim.
    """
    demo = persona_dict.get("demographics", {})
    if not demo:
        return 0.0

    evidence = persona_dict.get("source_evidence", [])
    demo_evidence_claims = set()
    for ev in evidence:
        fp = ev.get("field_path", "")
        claim = ev.get("claim", "").lower()
        if "demographic" in fp or "age" in claim or "gender" in claim or "location" in claim or "education" in claim or "income" in claim:
            demo_evidence_claims.add(fp)

    generic_patterns = re.compile(
        r"(unknown|not specified|varies|mixed|diverse|n/a|none)",
        re.IGNORECASE,
    )

    total_specific = 0
    unevidenced = 0
    for key, value in demo.items():
        if value is None or (isinstance(value, list) and not value):
            continue
        text = str(value)
        if generic_patterns.search(text):
            continue
        total_specific += 1
        # Check if any evidence supports this demographic field
        has_evidence = any(
            key in fp or key.replace("_", " ") in ev.get("claim", "").lower()
            for ev in evidence
            for fp in [ev.get("field_path", "")]
        )
        if not has_evidence:
            unevidenced += 1

    return unevidenced / total_specific if total_specific > 0 else 0.0


async def judge_persona(
    client: AsyncAnthropic,
    persona_dict: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[dict, float]:
    stripped = {k: v for k, v in persona_dict.items() if not k.startswith("_")}
    prompt = JUDGE_PROMPT + json.dumps(stripped, indent=2)

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    elif "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    return json.loads(text), cost


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


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(all_metrics: list[RunMetrics]) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    by_variant: dict[str, list[RunMetrics]] = {}
    for m in all_metrics:
        by_variant.setdefault(m.variant, []).append(m)
    variants = list(by_variant.keys())

    p("\n" + "=" * 100)
    p("EXPERIMENT 3.04 — EVIDENCE-FIRST GENERATION — RESULTS")
    p("=" * 100)

    header = f"{'Metric':<40}"
    for v in variants:
        header += f"{v:>28}"
    p(header)
    p("-" * (40 + 28 * len(variants)))

    def row(label, getter, fmt=".3f"):
        line = f"{label:<40}"
        for v in variants:
            valid = [m for m in by_variant[v] if m.success]
            if valid:
                vals = [getter(m) for m in valid]
                vals = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
                if vals:
                    avg = statistics.mean(vals)
                    line += f"{avg:>28{fmt}}"
                else:
                    line += f"{'N/A':>28}"
            else:
                line += f"{'FAILED':>28}"
        p(line)

    row("Groundedness score",        lambda m: m.groundedness_score)
    row("Evidence count",            lambda m: m.evidence_count, fmt=".0f")
    row("Avg evidence confidence",   lambda m: m.avg_confidence)
    row("Selected records (pass 1)", lambda m: m.selected_records, fmt=".0f")
    row("Evidence categories used",  lambda m: m.evidence_categories, fmt=".0f")
    row("Demo hallucination rate",   lambda m: m.demo_hallucination_rate)
    row("Judge: overall",            lambda m: m.judge_overall)
    row("Judge: voice_fidelity",     lambda m: m.judge_voice_fidelity)
    for dim in JUDGE_DIMENSIONS:
        if dim != "voice_fidelity":
            row(f"  {dim}",          lambda m, d=dim: m.judge_dimensions.get(d, float("nan")))
    row("Selection cost (USD)",      lambda m: m.selection_cost_usd, fmt=".4f")
    row("Synthesis cost (USD)",      lambda m: m.synthesis_cost_usd, fmt=".4f")
    row("Total cost (USD)",          lambda m: m.total_cost_usd, fmt=".4f")
    row("Attempts",                  lambda m: m.attempts, fmt=".1f")
    row("Duration (s)",              lambda m: m.duration_seconds, fmt=".1f")

    p("-" * (40 + 28 * len(variants)))

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    ctrl = [m for m in by_variant.get("control", []) if m.success]
    ef = [m for m in by_variant.get("evidence-first", []) if m.success]

    if ctrl and ef:
        ctrl_ground = statistics.mean([m.groundedness_score for m in ctrl])
        ef_ground = statistics.mean([m.groundedness_score for m in ef])

        ctrl_voice = [m.judge_voice_fidelity for m in ctrl if not math.isnan(m.judge_voice_fidelity)]
        ef_voice = [m.judge_voice_fidelity for m in ef if not math.isnan(m.judge_voice_fidelity)]

        ctrl_halluc = statistics.mean([m.demo_hallucination_rate for m in ctrl])
        ef_halluc = statistics.mean([m.demo_hallucination_rate for m in ef])

        ctrl_cost = statistics.mean([m.total_cost_usd for m in ctrl])
        ef_cost = statistics.mean([m.total_cost_usd for m in ef])

        d_ground = ef_ground - ctrl_ground
        d_voice = (statistics.mean(ef_voice) - statistics.mean(ctrl_voice)) if ctrl_voice and ef_voice else float("nan")
        d_halluc = ef_halluc - ctrl_halluc

        p(f"\n  Groundedness:           {d_ground:+.4f} "
          f"({'IMPROVED' if d_ground > 0.01 else 'SIMILAR' if d_ground > -0.01 else 'DEGRADED'})")
        if not math.isnan(d_voice):
            p(f"  Voice fidelity:         {d_voice:+.4f} "
              f"({'IMPROVED' if d_voice > 0.03 else 'SIMILAR' if d_voice > -0.03 else 'DEGRADED'})")
        p(f"  Demo hallucination:     {d_halluc:+.4f} "
          f"({'REDUCED' if d_halluc < -0.05 else 'SIMILAR' if d_halluc < 0.05 else 'INCREASED'})")
        p(f"  Cost multiplier:        {ef_cost / ctrl_cost:.1f}x" if ctrl_cost > 0 else "")

        signals = []
        if d_ground > 0.01:
            signals.append("GROUNDEDNESS_UP")
        if not math.isnan(d_voice) and d_voice > 0.03:
            signals.append("VOICE_IMPROVED")
        if d_halluc < -0.05:
            signals.append("HALLUCINATION_REDUCED")

        strength = "STRONG FINDING" if len(signals) >= 2 else ("MODERATE FINDING" if signals else "WEAK FINDING")
        p(f"\n  Signal: {strength}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 3.04: Evidence-first generation")
    print("Hypothesis: Pre-selecting evidence improves groundedness,")
    print("  voice fidelity, and reduces demographic hallucination")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)
    model = settings.default_model

    print("\n[1/4] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    all_metrics: list[RunMetrics] = []

    # Control
    print("\n[2/4] Running control (single-pass synthesis)...")
    for cluster in clusters:
        print(f"  Cluster: {cluster.cluster_id}")
        t0 = time.monotonic()
        m = RunMetrics(variant="control", cluster_id=cluster.cluster_id)
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            ev = pd.get("source_evidence", [])

            m.success = True
            m.persona_name = result.persona.name
            m.groundedness_score = result.groundedness.score
            m.evidence_count = len(ev)
            m.avg_confidence = statistics.mean([e.get("confidence", 0) for e in ev]) if ev else 0
            m.demo_hallucination_rate = compute_demo_hallucination_rate(pd)
            m.synthesis_cost_usd = result.total_cost_usd
            m.total_cost_usd = result.total_cost_usd
            m.attempts = result.attempts

            scores, jcost = await judge_persona(client, pd)
            m.judge_overall = float(scores.get("overall", float("nan")))
            m.judge_voice_fidelity = float(scores.get("voice_fidelity", float("nan")))
            m.judge_dimensions = {d: float(scores.get(d, float("nan"))) for d in JUDGE_DIMENSIONS}
            m.judge_rationale = scores.get("rationale", "")
            m.total_cost_usd += jcost

            print(f"    {m.persona_name}: ground={m.groundedness_score:.2f}, "
                  f"voice={m.judge_voice_fidelity:.2f}, "
                  f"halluc={m.demo_hallucination_rate:.2f}, cost=${m.total_cost_usd:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
        m.duration_seconds = time.monotonic() - t0
        all_metrics.append(m)

    # Evidence-first
    print("\n[3/4] Running evidence-first (two-pass synthesis)...")
    for cluster in clusters:
        print(f"  Cluster: {cluster.cluster_id}")
        t0 = time.monotonic()
        m = RunMetrics(variant="evidence-first", cluster_id=cluster.cluster_id)
        try:
            result = await evidence_first_synthesize(cluster, client, model)
            pd = result.persona.model_dump(mode="json")
            ev = pd.get("source_evidence", [])

            m.success = True
            m.persona_name = result.persona.name
            m.groundedness_score = result.groundedness.score
            m.evidence_count = len(ev)
            m.avg_confidence = statistics.mean([e.get("confidence", 0) for e in ev]) if ev else 0
            m.selected_records = len(result.evidence_package.selected_records)
            cats = {r.category for r in result.evidence_package.selected_records}
            m.evidence_categories = len(cats)
            m.demo_hallucination_rate = compute_demo_hallucination_rate(pd)
            m.selection_cost_usd = result.selection_cost_usd
            m.synthesis_cost_usd = result.synthesis_cost_usd
            m.total_cost_usd = result.total_cost_usd
            m.attempts = result.synthesis_attempts

            scores, jcost = await judge_persona(client, pd)
            m.judge_overall = float(scores.get("overall", float("nan")))
            m.judge_voice_fidelity = float(scores.get("voice_fidelity", float("nan")))
            m.judge_dimensions = {d: float(scores.get(d, float("nan"))) for d in JUDGE_DIMENSIONS}
            m.judge_rationale = scores.get("rationale", "")
            m.total_cost_usd += jcost

            print(f"    {m.persona_name}: ground={m.groundedness_score:.2f}, "
                  f"voice={m.judge_voice_fidelity:.2f}, "
                  f"halluc={m.demo_hallucination_rate:.2f}, "
                  f"selected={m.selected_records} records, "
                  f"cost=${m.total_cost_usd:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
        m.duration_seconds = time.monotonic() - t0
        all_metrics.append(m)

    # Report
    print("\n[4/4] Comparing results...")
    report = print_results(all_metrics)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    results_data = {
        "experiment": "3.04",
        "title": "Evidence-first generation",
        "hypothesis": "Pre-selecting evidence improves groundedness and voice fidelity",
        "model": model,
        "metrics": [
            {
                "variant": m.variant,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "success": m.success,
                "groundedness_score": m.groundedness_score,
                "evidence_count": m.evidence_count,
                "avg_confidence": m.avg_confidence,
                "selected_records": m.selected_records,
                "evidence_categories": m.evidence_categories,
                "demo_hallucination_rate": m.demo_hallucination_rate,
                "judge_overall": safe(m.judge_overall),
                "judge_voice_fidelity": safe(m.judge_voice_fidelity),
                "judge_dimensions": {k: safe(v) for k, v in m.judge_dimensions.items()},
                "selection_cost_usd": m.selection_cost_usd,
                "synthesis_cost_usd": m.synthesis_cost_usd,
                "total_cost_usd": m.total_cost_usd,
                "attempts": m.attempts,
                "duration_seconds": m.duration_seconds,
            }
            for m in all_metrics
        ],
    }

    results_path = output_dir / "exp_3_04_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_3_04_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
