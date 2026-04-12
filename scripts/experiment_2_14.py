"""Experiment 2.14: Constitutional persona.

Hypothesis: Injecting a self-critique constitution into the system prompt
reduces hedging language and increases grounded first-person quotes,
without the cost of an external critique loop.

Variants:
  - control:        Standard system prompt
  - constitutional: System prompt + constitution (self-critique principles)

Metrics:
  - Hedging rate: fraction of text fields containing hedge words
  - Grounded-quote rate: fraction of sample_quotes that are first-person
    and contain specific details from the data
  - Judge scores (all 5 dimensions)
  - Cost (should be ~1.0x since it's a single call)

Usage:
    python scripts/experiment_2_14.py
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
from pydantic import ValidationError  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.engine.prompt_builder import (  # noqa: E402
    SYSTEM_PROMPT,
    build_constitutional_system_prompt,
    build_tool_definition,
    build_user_message,
    build_retry_messages,
)
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

HEDGE_WORDS = re.compile(
    r"\b(might|could|potentially|perhaps|possibly|it seems|generally|tend to|"
    r"may be|likely|probably|arguably|somewhat|rather|fairly|quite possibly|"
    r"in some cases|to some extent)\b",
    re.IGNORECASE,
)

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

DIMENSIONS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")


# ── Metrics ──────────────────────────────────────────────────────────

def compute_hedging_rate(persona_dict: dict) -> float:
    """Fraction of text-containing fields that include hedging language."""
    text_fields = []
    for key in ("summary", "goals", "pains", "motivations", "objections",
                "decision_triggers", "sample_quotes"):
        val = persona_dict.get(key)
        if isinstance(val, str):
            text_fields.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    text_fields.append(item)

    if not text_fields:
        return 0.0

    hedged = sum(1 for t in text_fields if HEDGE_WORDS.search(t))
    return hedged / len(text_fields)


def compute_grounded_quote_rate(persona_dict: dict) -> float:
    """Fraction of sample_quotes that are first-person and specific.

    A 'grounded quote' starts with first-person language (I, We, My, Our)
    and contains at least one specific detail (number, tool name, or
    concrete action).
    """
    quotes = persona_dict.get("sample_quotes", [])
    if not quotes:
        return 0.0

    first_person = re.compile(r"^(I |We |My |Our |I'm |I've |We're |We've )", re.IGNORECASE)
    specific = re.compile(
        r"(\d+|%|\$|API|SDK|CI/CD|Terraform|Figma|Slack|GitHub|webhook|"
        r"hours?|minutes?|seconds?|days?|weeks?|sprint|deploy|client|"
        r"project|invoice|billing|template|dashboard)",
        re.IGNORECASE,
    )

    grounded = 0
    for q in quotes:
        if first_person.match(q) and specific.search(q):
            grounded += 1

    return grounded / len(quotes)


@dataclass
class RunMetrics:
    variant: str
    cluster_id: str = ""
    persona_name: str = ""
    success: bool = False
    groundedness: float = 0.0
    hedging_rate: float = 0.0
    grounded_quote_rate: float = 0.0
    hedge_count: int = 0
    total_text_fields: int = 0
    judge_overall: float = float("nan")
    judge_dimensions: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    attempts: int = 0
    duration_seconds: float = 0.0


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


async def synthesize_constitutional(
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
    max_retries: int = 2,
) -> tuple[PersonaV1, float, float, int]:
    """Synthesize with constitutional system prompt. Returns (persona, groundedness, cost, attempts)."""
    system = build_constitutional_system_prompt()
    tool = build_tool_definition()
    total_cost = 0.0
    errors_for_retry: list[str] = []

    for attempt in range(1, max_retries + 2):
        if errors_for_retry:
            msgs = build_retry_messages(cluster, errors_for_retry)
        else:
            msgs = [{"role": "user", "content": build_user_message(cluster)}]

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=msgs,
            tools=[tool],
            tool_choice={"type": "tool", "name": "create_persona"},
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        if "haiku" in model:
            cost = (in_tok * 1 + out_tok * 5) / 1_000_000
        elif "opus" in model:
            cost = (in_tok * 15 + out_tok * 75) / 1_000_000
        else:
            cost = (in_tok * 3 + out_tok * 15) / 1_000_000
        total_cost += cost

        errors_for_retry = []
        try:
            persona = PersonaV1.model_validate(tool_block.input)
        except ValidationError as e:
            errors_for_retry = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            continue

        ground = check_groundedness(persona, cluster)
        if not ground.passed:
            errors_for_retry = ground.violations
            continue

        return persona, ground.score, total_cost, attempt

    raise RuntimeError(f"Constitutional synthesis failed after {max_retries + 1} attempts")


async def judge_persona(client: AsyncAnthropic, persona_dict: dict, model: str) -> tuple[dict, float]:
    stripped = {k: v for k, v in persona_dict.items() if not k.startswith("_")}
    prompt = JUDGE_PROMPT + json.dumps(stripped, indent=2)
    response = await client.messages.create(
        model=model, max_tokens=1024,
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
    cost = (in_tok * 1 + out_tok * 5) / 1_000_000 if "haiku" in model else (in_tok * 3 + out_tok * 15) / 1_000_000
    return json.loads(text), cost


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
    p("EXPERIMENT 2.14 — CONSTITUTIONAL PERSONA — RESULTS")
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
                    line += f"{statistics.mean(vals):>28{fmt}}"
                else:
                    line += f"{'N/A':>28}"
            else:
                line += f"{'FAILED':>28}"
        p(line)

    row("Hedging rate",              lambda m: m.hedging_rate)
    row("Grounded-quote rate",       lambda m: m.grounded_quote_rate)
    row("Groundedness",              lambda m: m.groundedness)
    row("Judge: overall",            lambda m: m.judge_overall)
    for dim in DIMENSIONS:
        row(f"  {dim}",              lambda m, d=dim: m.judge_dimensions.get(d, float("nan")))
    row("Cost (USD)",                lambda m: m.cost_usd, fmt=".4f")
    row("Attempts",                  lambda m: m.attempts, fmt=".1f")

    p("-" * (40 + 28 * len(variants)))

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    ctrl = [m for m in by_variant.get("control", []) if m.success]
    con = [m for m in by_variant.get("constitutional", []) if m.success]

    if ctrl and con:
        ctrl_hedge = statistics.mean([m.hedging_rate for m in ctrl])
        con_hedge = statistics.mean([m.hedging_rate for m in con])
        d_hedge = con_hedge - ctrl_hedge

        ctrl_gq = statistics.mean([m.grounded_quote_rate for m in ctrl])
        con_gq = statistics.mean([m.grounded_quote_rate for m in con])
        d_gq = con_gq - ctrl_gq

        ctrl_judge = [m.judge_overall for m in ctrl if not math.isnan(m.judge_overall)]
        con_judge = [m.judge_overall for m in con if not math.isnan(m.judge_overall)]
        d_judge = (statistics.mean(con_judge) - statistics.mean(ctrl_judge)) if ctrl_judge and con_judge else float("nan")

        ctrl_cost = statistics.mean([m.cost_usd for m in ctrl])
        con_cost = statistics.mean([m.cost_usd for m in con])

        hedge_sig = "REDUCED" if d_hedge < -0.03 else ("SIMILAR" if d_hedge < 0.03 else "INCREASED")
        gq_sig = "IMPROVED" if d_gq > 0.05 else ("SIMILAR" if d_gq > -0.05 else "DEGRADED")

        p(f"\n  Hedging rate:        {d_hedge:+.4f} ({hedge_sig})")
        p(f"    control={ctrl_hedge:.3f}, constitutional={con_hedge:.3f}")
        p(f"  Grounded-quote rate: {d_gq:+.4f} ({gq_sig})")
        p(f"    control={ctrl_gq:.3f}, constitutional={con_gq:.3f}")
        if not math.isnan(d_judge):
            p(f"  Judge overall:       {d_judge:+.4f}")
        p(f"  Cost multiplier:     {con_cost / ctrl_cost:.2f}x" if ctrl_cost > 0 else "")

        signals = []
        if d_hedge < -0.03:
            signals.append("HEDGING_REDUCED")
        if d_gq > 0.05:
            signals.append("QUOTES_IMPROVED")
        if not math.isnan(d_judge) and d_judge > 0.03:
            signals.append("JUDGE_UP")

        strength = "STRONG FINDING" if len(signals) >= 2 else ("MODERATE FINDING" if signals else "WEAK FINDING")
        p(f"\n  Signal: {strength}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 2.14: Constitutional persona")
    print("Hypothesis: Self-critique constitution reduces hedging and")
    print("  increases grounded first-person quotes at zero extra cost")
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
    print("\n[2/4] Running control (standard prompt)...")
    for cluster in clusters:
        t0 = time.monotonic()
        m = RunMetrics(variant="control", cluster_id=cluster.cluster_id)
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            m.success = True
            m.persona_name = result.persona.name
            m.groundedness = result.groundedness.score
            m.hedging_rate = compute_hedging_rate(pd)
            m.grounded_quote_rate = compute_grounded_quote_rate(pd)
            m.cost_usd = result.total_cost_usd
            m.attempts = result.attempts
            scores, jcost = await judge_persona(client, pd, model)
            m.judge_overall = float(scores.get("overall", float("nan")))
            m.judge_dimensions = {d: float(scores.get(d, float("nan"))) for d in DIMENSIONS}
            m.cost_usd += jcost
            print(f"    {m.persona_name}: hedge={m.hedging_rate:.3f}, "
                  f"quotes={m.grounded_quote_rate:.3f}, cost=${m.cost_usd:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
        m.duration_seconds = time.monotonic() - t0
        all_metrics.append(m)

    # Constitutional
    print("\n[3/4] Running constitutional (self-critique prompt)...")
    for cluster in clusters:
        t0 = time.monotonic()
        m = RunMetrics(variant="constitutional", cluster_id=cluster.cluster_id)
        try:
            persona, ground, cost, attempts = await synthesize_constitutional(
                cluster, client, model,
            )
            pd = persona.model_dump(mode="json")
            m.success = True
            m.persona_name = persona.name
            m.groundedness = ground
            m.hedging_rate = compute_hedging_rate(pd)
            m.grounded_quote_rate = compute_grounded_quote_rate(pd)
            m.cost_usd = cost
            m.attempts = attempts
            scores, jcost = await judge_persona(client, pd, model)
            m.judge_overall = float(scores.get("overall", float("nan")))
            m.judge_dimensions = {d: float(scores.get(d, float("nan"))) for d in DIMENSIONS}
            m.cost_usd += jcost
            print(f"    {m.persona_name}: hedge={m.hedging_rate:.3f}, "
                  f"quotes={m.grounded_quote_rate:.3f}, cost=${m.cost_usd:.4f}")
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
        "experiment": "2.14",
        "title": "Constitutional persona",
        "hypothesis": "Self-critique constitution reduces hedging and improves quote grounding",
        "model": model,
        "metrics": [
            {
                "variant": m.variant, "cluster_id": m.cluster_id,
                "persona_name": m.persona_name, "success": m.success,
                "groundedness": m.groundedness, "hedging_rate": m.hedging_rate,
                "grounded_quote_rate": m.grounded_quote_rate,
                "judge_overall": safe(m.judge_overall),
                "judge_dimensions": {k: safe(v) for k, v in m.judge_dimensions.items()},
                "cost_usd": m.cost_usd, "attempts": m.attempts,
            }
            for m in all_metrics
        ],
    }

    (output_dir / "exp_2_14_results.json").write_text(json.dumps(results_data, indent=2))
    (output_dir / "exp_2_14_report.txt").write_text(report)
    print(f"\nResults saved to: {output_dir / 'exp_2_14_results.json'}")
    print(f"Report saved to:  {output_dir / 'exp_2_14_report.txt'}")


if __name__ == "__main__":
    asyncio.run(main())
