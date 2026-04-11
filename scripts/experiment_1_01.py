"""Experiment 1.01: Schema width.

Hypothesis: Minimal schemas (~5 fields) under-specify the persona, leading to
generic twins with low consistency. Maximal schemas (~25 fields) may plateau
or degrade as the added surface area creates contradictions. The current schema
(~12 fields) is the sweet spot.

Variants:
  - minimal:  PersonaMinimal  (~5 fields:  name, summary, goals, pains, evidence)
  - current:  PersonaV1       (~12 fields: control)
  - maximal:  PersonaMaximal  (~25 fields: PersonaV1 + backstory, routine, etc.)

Metrics:
  - Twin consistency (drift):  word-overlap across multi-turn conversation
  - Schema validity:           fraction of outputs passing Pydantic validation
  - Judge scores:              LLM-as-judge on grounded/distinctive/coherent/
                               actionable/voice_fidelity
  - Field count:               actual fields produced per variant
  - Cost per persona

Usage:
    python scripts/experiment_1_01.py
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
sys.path.insert(0, str(REPO_ROOT / "twin"))
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
from synthesis.models.cluster import ClusterData  # noqa: E402
from twin import TwinChat  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

VARIANTS = {
    "minimal (~5 fields)": "minimal",
    "current (~12 fields, control)": "current",
    "maximal (~25 fields)": "maximal",
}

TWIN_QUESTIONS = [
    "Tell me about yourself and your background.",
    "What tools do you use day-to-day?",
    "What's the most frustrating part of your job?",
    "How do you make purchasing decisions for new tools?",
    "Describe a typical workday for someone like you.",
]

JUDGE_DIMENSIONS = (
    "grounded", "distinctive", "coherent", "actionable", "voice_fidelity",
)

JUDGE_RUBRIC_PROMPT = """\
You are an expert evaluator of synthetic marketing personas. Score the \
persona below on each dimension using a 0.0-1.0 scale.

## Scoring rubric
- **grounded** (0.0-1.0): Claims traceable to source data with evidence entries.
- **distinctive** (0.0-1.0): Feels like a real individual, not a generic average.
- **coherent** (0.0-1.0): Internally consistent across all fields.
- **actionable** (0.0-1.0): Goals/pains specific enough to drive product decisions.
- **voice_fidelity** (0.0-1.0): Quotes and vocabulary sound like one consistent speaker.

Respond with ONLY a JSON object:
{"grounded":<float>,"distinctive":<float>,"coherent":<float>,\
"actionable":<float>,"voice_fidelity":<float>,\
"overall":<float>,"rationale":"<1-2 sentences>"}

## Persona to evaluate

"""


# ── Metrics ──────────────────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    variant: str
    schema_width: str
    cluster_id: str = ""
    persona_name: str = ""
    # Pipeline
    groundedness_score: float = 0.0
    schema_valid: bool = False
    cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0
    # Schema width
    field_count: int = 0
    # Twin drift
    twin_drift_score: float = 0.0
    twin_cost_usd: float = 0.0
    # Judge
    judge_overall: float = float("nan")
    judge_dimensions: dict = field(default_factory=dict)
    judge_rationale: str = ""
    judge_cost_usd: float = 0.0


def count_fields(persona_dict: dict) -> int:
    """Count non-meta, non-null top-level fields."""
    skip = {"schema_version", "_meta"}
    return sum(
        1 for k, v in persona_dict.items()
        if k not in skip and v is not None and v != [] and v != ""
    )


def compute_twin_drift(responses: list[str]) -> float:
    """Word-overlap consistency between consecutive twin responses."""
    if len(responses) < 2:
        return 1.0
    sims = []
    for i in range(len(responses) - 1):
        words_a = set(responses[i].lower().split())
        words_b = set(responses[i + 1].lower().split())
        union = words_a | words_b
        intersection = words_a & words_b
        sims.append(len(intersection) / len(union) if union else 1.0)
    return statistics.mean(sims)


async def judge_persona(
    client: AsyncAnthropic,
    persona_dict: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[dict, float]:
    """Score a persona with an LLM judge. Returns (scores_dict, cost)."""
    stripped = {k: v for k, v in persona_dict.items() if not k.startswith("_")}
    prompt = JUDGE_RUBRIC_PROMPT + json.dumps(stripped, indent=2)

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
    if "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    elif "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    parsed = json.loads(text)
    return parsed, cost


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
    schema_width: str,
    cluster: ClusterData,
    synth_backend: AnthropicBackend,
    client: AsyncAnthropic,
) -> ExperimentMetrics:
    metrics = ExperimentMetrics(
        variant=variant_name,
        schema_width=schema_width,
        cluster_id=cluster.cluster_id,
    )

    t0 = time.monotonic()

    try:
        result = await synthesize(
            cluster, synth_backend,
            schema_width=schema_width,
        )
        metrics.schema_valid = True
        metrics.groundedness_score = result.groundedness.score
        metrics.cost_usd = result.total_cost_usd
        metrics.attempts = result.attempts
        metrics.persona_name = result.persona.name

        persona_dict = result.persona.model_dump(mode="json")
        metrics.field_count = count_fields(persona_dict)

        # Twin conversation for drift
        twin = TwinChat(persona_dict, client=client, model=settings.default_model)
        responses = []
        history = []
        twin_cost = 0.0
        for question in TWIN_QUESTIONS:
            reply = await twin.reply(question, history=history)
            responses.append(reply.text)
            twin_cost += reply.estimated_cost_usd
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": reply.text})

        metrics.twin_drift_score = compute_twin_drift(responses)
        metrics.twin_cost_usd = twin_cost

        # Judge scoring
        try:
            scores, jcost = await judge_persona(client, persona_dict)
            metrics.judge_overall = float(scores.get("overall", float("nan")))
            metrics.judge_dimensions = {
                d: float(scores.get(d, float("nan"))) for d in JUDGE_DIMENSIONS
            }
            metrics.judge_rationale = scores.get("rationale", "")
            metrics.judge_cost_usd = jcost
        except Exception as e:
            print(f"      Judge failed: {e}")

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

    p("\n" + "=" * 110)
    p("EXPERIMENT 1.01 — SCHEMA WIDTH — RESULTS")
    p("=" * 110)

    header = f"{'Metric':<35}"
    for v in seen:
        header += f"{v:>25}"
    p(header)
    p("-" * (35 + 25 * len(seen)))

    def row(label, getter, fmt=".3f"):
        line = f"{label:<35}"
        for v in seen:
            valid = [m for m in by_variant[v] if m.schema_valid]
            if valid:
                avg = statistics.mean([getter(m) for m in valid])
                line += f"{avg:>25{fmt}}"
            else:
                line += f"{'FAILED':>25}"
        p(line)

    row("Field count",             lambda m: m.field_count, fmt=".0f")
    row("Schema validity",         lambda m: 1.0)
    row("Groundedness",            lambda m: m.groundedness_score)
    row("Twin drift (consistency)", lambda m: m.twin_drift_score)
    row("Judge: overall",          lambda m: m.judge_overall)
    for dim in JUDGE_DIMENSIONS:
        row(f"  {dim}",            lambda m, d=dim: m.judge_dimensions.get(d, float("nan")))
    row("Synthesis cost (USD)",    lambda m: m.cost_usd, fmt=".4f")
    row("Twin cost (USD)",         lambda m: m.twin_cost_usd, fmt=".4f")
    row("Judge cost (USD)",        lambda m: m.judge_cost_usd, fmt=".4f")
    row("Attempts",                lambda m: m.attempts, fmt=".1f")
    row("Duration (s)",            lambda m: m.duration_seconds, fmt=".1f")

    p("-" * (35 + 25 * len(seen)))

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    ctrl = [m for m in all_metrics if m.schema_width == "current" and m.schema_valid]
    if ctrl:
        ctrl_drift = statistics.mean([m.twin_drift_score for m in ctrl])
        ctrl_judge = statistics.mean(
            [m.judge_overall for m in ctrl if not math.isnan(m.judge_overall)]
        ) if any(not math.isnan(m.judge_overall) for m in ctrl) else float("nan")
        ctrl_cost = statistics.mean([m.cost_usd for m in ctrl])

        for vname in seen:
            if "control" in vname:
                continue
            v = [m for m in by_variant[vname] if m.schema_valid]
            if not v:
                p(f"\n  {vname}: ALL FAILED")
                continue

            d_drift = statistics.mean([m.twin_drift_score for m in v]) - ctrl_drift
            v_judge_vals = [m.judge_overall for m in v if not math.isnan(m.judge_overall)]
            d_judge = (statistics.mean(v_judge_vals) - ctrl_judge) if v_judge_vals and not math.isnan(ctrl_judge) else float("nan")
            d_cost = statistics.mean([m.cost_usd for m in v]) - ctrl_cost
            v_fields = statistics.mean([m.field_count for m in v])

            drift_sig = "LESS DRIFT" if d_drift > 0.02 else ("MORE DRIFT" if d_drift < -0.02 else "SIMILAR")
            judge_sig = "HIGHER" if not math.isnan(d_judge) and d_judge > 0.03 else (
                "LOWER" if not math.isnan(d_judge) and d_judge < -0.03 else "SIMILAR"
            )

            signals = [s for s in [drift_sig, judge_sig] if s not in ("SIMILAR",)]
            strength = "STRONG" if len(signals) >= 2 else ("MODERATE" if len(signals) == 1 else "WEAK")

            p(f"\n  {vname} (avg {v_fields:.0f} fields):")
            p(f"    Twin drift:     {d_drift:+.4f} ({drift_sig})")
            if not math.isnan(d_judge):
                p(f"    Judge overall:  {d_judge:+.4f} ({judge_sig})")
            p(f"    Cost delta:     {d_cost:+.4f}")
            p(f"    Signal:         {strength}")

    p("\n" + "=" * 110)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 1.01: Schema width")
    print("Hypothesis: Minimal under-specifies, maximal over-specifies,")
    print("  current (~12 fields) is the sweet spot")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[1/3] Running shared ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    print("\n[2/3] Running variants (synthesis + twin + judge)...")
    all_metrics: list[ExperimentMetrics] = []

    for variant_name, width in VARIANTS.items():
        print(f"\n  -- {variant_name} (width={width}) --")
        for cluster in clusters:
            print(f"    Cluster: {cluster.cluster_id}")
            m = await run_variant(variant_name, width, cluster, synth_backend, client)
            all_metrics.append(m)
            if m.schema_valid:
                judge_str = f"judge={m.judge_overall:.3f}" if not math.isnan(m.judge_overall) else "judge=N/A"
                print(f"      {m.persona_name}: fields={m.field_count}, "
                      f"drift={m.twin_drift_score:.3f}, {judge_str}, "
                      f"cost=${m.cost_usd:.4f}")

    print("\n[3/3] Comparing results...")
    report = print_comparison(all_metrics)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    def safe(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    results_data = {
        "experiment": "1.01",
        "title": "Schema width",
        "hypothesis": "Minimal under-specifies, maximal may over-specify, current is sweet spot",
        "model": settings.default_model,
        "variants": {k: v for k, v in VARIANTS.items()},
        "metrics": [
            {
                "variant": m.variant,
                "schema_width": m.schema_width,
                "cluster_id": m.cluster_id,
                "persona_name": m.persona_name,
                "field_count": m.field_count,
                "groundedness_score": m.groundedness_score,
                "schema_valid": m.schema_valid,
                "cost_usd": m.cost_usd,
                "attempts": m.attempts,
                "twin_drift_score": m.twin_drift_score,
                "twin_cost_usd": m.twin_cost_usd,
                "judge_overall": safe(m.judge_overall),
                "judge_dimensions": {k: safe(v) for k, v in m.judge_dimensions.items()},
                "judge_rationale": m.judge_rationale,
                "judge_cost_usd": m.judge_cost_usd,
                "duration_seconds": m.duration_seconds,
            }
            for m in all_metrics
        ],
    }

    results_path = output_dir / "exp_1_01_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_1_01_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
