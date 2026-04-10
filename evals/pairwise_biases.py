"""Experiment 5.04: Position & verbosity bias in pairwise LLM judging.

Hypothesis: The LLM judge systematically prefers longer or first-presented
options, independent of actual quality.

Tests:
  1. Position bias: Present (A,B) then (B,A). Measure flip rate — a fair judge
     should pick the same winner regardless of order.
  2. Verbosity bias: Pad the weaker option with filler text. Measure whether
     length alone flips the judgment.

Metrics:
  - Flip rate:       fraction of pairs where swapping order changes the winner
  - Length-wins rate: fraction of padded trials where the padded (longer) option wins

Usage:
    python evals/pairwise_biases.py
"""

from __future__ import annotations

import asyncio
import copy
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
from evaluation.judges import LLMJudge  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

# Filler sentences to pad personas for verbosity bias test
FILLER_SENTENCES = [
    "This persona represents a significant segment of the user base.",
    "Understanding their needs is crucial for product development.",
    "Further research may reveal additional insights about this group.",
    "The data suggests interesting patterns in their behavior.",
    "Market trends indicate growing importance of this segment.",
    "Cross-referencing with industry benchmarks confirms these observations.",
    "Longitudinal analysis would strengthen confidence in these findings.",
    "Stakeholder alignment on this persona definition is recommended.",
]

FILLER_FIELDS = ["goals", "pains", "motivations", "objections"]


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class PositionBiasResult:
    """Result of one position-bias trial (A,B then B,A)."""
    pair_label: str
    order_ab_winner: str       # 'A' or 'B' or 'TIE' (A=persona_1, B=persona_2)
    order_ba_winner: str       # 'A' or 'B' or 'TIE' (A=persona_2, B=persona_1)
    order_ab_rationale: str = ""
    order_ba_rationale: str = ""
    flipped: bool = False      # True if winner changed based on position
    consistent_winner: str = ""  # Which persona won consistently, or "INCONSISTENT"


@dataclass
class VerbosityBiasResult:
    """Result of one verbosity-bias trial."""
    pair_label: str
    padded_side: str           # Which persona was padded ('A' or 'B')
    original_winner: str       # Winner without padding
    padded_winner: str         # Winner with padding
    original_rationale: str = ""
    padded_rationale: str = ""
    length_won: bool = False   # True if padding flipped result toward padded side


@dataclass
class BiasReport:
    # Position bias
    position_trials: list[PositionBiasResult] = field(default_factory=list)
    flip_rate: float = 0.0
    position_a_preference: float = 0.0  # rate of choosing first-presented

    # Verbosity bias
    verbosity_trials: list[VerbosityBiasResult] = field(default_factory=list)
    length_wins_rate: float = 0.0

    # Overall
    total_judge_calls: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────

def pad_persona_with_filler(persona: dict) -> dict:
    """Add filler text to list fields of a persona to increase verbosity."""
    padded = copy.deepcopy(persona)
    for i, field_name in enumerate(FILLER_FIELDS):
        if field_name in padded and isinstance(padded[field_name], list):
            filler = FILLER_SENTENCES[i % len(FILLER_SENTENCES)]
            padded[field_name] = padded[field_name] + [filler, filler]
    # Also pad summary
    if "summary" in padded:
        padded["summary"] = (
            padded["summary"] + " " +
            " ".join(FILLER_SENTENCES[:3])
        )
    return padded


def normalize_winner(winner: str, swapped: bool) -> str:
    """Convert a winner label back to canonical form when order was swapped."""
    if winner == "TIE":
        return "TIE"
    if swapped:
        return "B" if winner == "A" else "A"
    return winner


class AnthropicJudgeBackend:
    """Wraps AsyncAnthropic for the JudgeBackend protocol."""

    def __init__(self, client: AsyncAnthropic, model: str):
        self.client = client
        self.model = model
        self.total_cost = 0.0

    async def score(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        # Track cost
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        if "haiku" in self.model:
            self.total_cost += (inp * 1 + out * 5) / 1_000_000
        elif "opus" in self.model:
            self.total_cost += (inp * 15 + out * 75) / 1_000_000
        else:
            self.total_cost += (inp * 3 + out * 15) / 1_000_000
        return response.content[0].text


# ── Pipeline ──────────────────────────────────────────────────────────

def get_personas() -> list[dict]:
    """Run ingest + segment + synthesize to get real persona dicts."""
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


async def generate_personas(backend: AnthropicBackend) -> list[dict]:
    """Synthesize personas from clusters. Retries failed clusters up to 3 times."""
    clusters = get_personas()
    personas = []
    for cluster in clusters:
        for attempt in range(3):
            try:
                result = await synthesize(cluster, backend)
                personas.append(result.persona.model_dump(mode="json"))
                break
            except Exception as e:
                print(f"      Attempt {attempt + 1} failed for {cluster.cluster_id}: {e}")
                if attempt == 2:
                    print(f"      Skipping cluster {cluster.cluster_id}")
    return personas


# ── Bias tests ────────────────────────────────────────────────────────

async def test_position_bias(
    judge: LLMJudge,
    personas: list[dict],
) -> list[PositionBiasResult]:
    """Run pairwise judging in both orders for all persona pairs."""
    results = []
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            p1, p2 = personas[i], personas[j]
            label = f"{p1.get('name', f'P{i}')} vs {p2.get('name', f'P{j}')}"

            # Order 1: A=p1, B=p2
            winner_ab, rationale_ab = await judge.pairwise(p1, p2)

            # Order 2: A=p2, B=p1 (swapped)
            winner_ba_raw, rationale_ba = await judge.pairwise(p2, p1)
            # Normalize: convert back to p1/p2 reference frame
            winner_ba = normalize_winner(winner_ba_raw, swapped=True)

            flipped = winner_ab != winner_ba
            if not flipped and winner_ab != "TIE":
                consistent = f"persona_{i + 1}" if winner_ab == "A" else f"persona_{j + 1}"
            elif not flipped and winner_ab == "TIE":
                consistent = "TIE"
            else:
                consistent = "INCONSISTENT"

            results.append(PositionBiasResult(
                pair_label=label,
                order_ab_winner=winner_ab,
                order_ba_winner=winner_ba,
                order_ab_rationale=rationale_ab,
                order_ba_rationale=rationale_ba,
                flipped=flipped,
                consistent_winner=consistent,
            ))
    return results


async def test_verbosity_bias(
    judge: LLMJudge,
    personas: list[dict],
) -> list[VerbosityBiasResult]:
    """Pad one persona with filler and see if it wins more often."""
    results = []
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            p1, p2 = personas[i], personas[j]
            label = f"{p1.get('name', f'P{i}')} vs {p2.get('name', f'P{j}')}"

            # Baseline: unpadded comparison
            original_winner, original_rationale = await judge.pairwise(p1, p2)

            # Pad persona A (p1) with filler
            p1_padded = pad_persona_with_filler(p1)
            padded_winner_a, padded_rationale_a = await judge.pairwise(p1_padded, p2)
            length_won_a = (original_winner != "A" and padded_winner_a == "A")

            results.append(VerbosityBiasResult(
                pair_label=f"{label} (A padded)",
                padded_side="A",
                original_winner=original_winner,
                padded_winner=padded_winner_a,
                original_rationale=original_rationale,
                padded_rationale=padded_rationale_a,
                length_won=length_won_a,
            ))

            # Pad persona B (p2) with filler
            p2_padded = pad_persona_with_filler(p2)
            padded_winner_b, padded_rationale_b = await judge.pairwise(p1, p2_padded)
            length_won_b = (original_winner != "B" and padded_winner_b == "B")

            results.append(VerbosityBiasResult(
                pair_label=f"{label} (B padded)",
                padded_side="B",
                original_winner=original_winner,
                padded_winner=padded_winner_b,
                original_rationale=original_rationale,
                padded_rationale=padded_rationale_b,
                length_won=length_won_b,
            ))

    return results


# ── Reporting ─────────────────────────────────────────────────────────

def compute_report(
    position_results: list[PositionBiasResult],
    verbosity_results: list[VerbosityBiasResult],
    judge_backend: AnthropicJudgeBackend,
    duration: float,
) -> BiasReport:
    report = BiasReport()
    report.position_trials = position_results
    report.verbosity_trials = verbosity_results
    report.duration_seconds = duration
    report.total_cost_usd = judge_backend.total_cost

    # Position bias metrics
    if position_results:
        report.flip_rate = sum(1 for r in position_results if r.flipped) / len(position_results)
        # A-preference: how often does the first-presented win?
        # Count original order where A won + swapped order where A won (=B in canonical)
        a_wins = sum(1 for r in position_results if r.order_ab_winner == "A")
        total_non_tie = sum(1 for r in position_results if r.order_ab_winner != "TIE")
        report.position_a_preference = a_wins / total_non_tie if total_non_tie else 0.5

    # Verbosity bias metrics
    if verbosity_results:
        report.length_wins_rate = sum(1 for r in verbosity_results if r.length_won) / len(verbosity_results)

    # Total judge calls
    report.total_judge_calls = len(position_results) * 2 + len(verbosity_results) * 1 + len(position_results)
    # (position: 2 per pair, verbosity: 1 baseline + 2 padded per pair, but baseline shared)

    return report


def print_report(report: BiasReport) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 90)
    p("EXPERIMENT 5.04 — POSITION & VERBOSITY BIAS — RESULTS")
    p("=" * 90)

    # Position bias
    p("\n── POSITION BIAS (A/B swap test) ──")
    p(f"  Trials:        {len(report.position_trials)}")
    p(f"  Flip rate:     {report.flip_rate:.1%}")
    p(f"  A-preference:  {report.position_a_preference:.1%} (50% = no bias)")

    for r in report.position_trials:
        flip_marker = " ** FLIPPED **" if r.flipped else ""
        p(f"\n  {r.pair_label}:")
        p(f"    Order (A,B): winner={r.order_ab_winner}")
        p(f"    Order (B,A): winner={r.order_ba_winner} (canonical){flip_marker}")
        p(f"    Consistent: {r.consistent_winner}")

    # Position bias signal
    p("\n  ── Position Bias Signal ──")
    if report.flip_rate > 0.3:
        p(f"  STRONG BIAS: {report.flip_rate:.0%} of judgments flip on position swap")
    elif report.flip_rate > 0.1:
        p(f"  MODERATE BIAS: {report.flip_rate:.0%} of judgments flip on position swap")
    else:
        p(f"  LOW/NO BIAS: {report.flip_rate:.0%} of judgments flip on position swap")

    if abs(report.position_a_preference - 0.5) > 0.2:
        direction = "first-presented (A)" if report.position_a_preference > 0.5 else "second-presented (B)"
        p(f"  DIRECTIONAL BIAS: judge prefers {direction} ({report.position_a_preference:.0%} A-win rate)")
    else:
        p(f"  No strong directional preference ({report.position_a_preference:.0%} A-win rate)")

    # Verbosity bias
    p("\n── VERBOSITY BIAS (filler padding test) ──")
    p(f"  Trials:            {len(report.verbosity_trials)}")
    p(f"  Length-wins rate:  {report.length_wins_rate:.1%}")

    for r in report.verbosity_trials:
        flip_marker = " ** LENGTH WON **" if r.length_won else ""
        p(f"\n  {r.pair_label}:")
        p(f"    Original winner:  {r.original_winner}")
        p(f"    Padded winner:    {r.padded_winner}{flip_marker}")

    # Verbosity bias signal
    p("\n  ── Verbosity Bias Signal ──")
    if report.length_wins_rate > 0.3:
        p(f"  STRONG BIAS: {report.length_wins_rate:.0%} of padded trials won by length")
    elif report.length_wins_rate > 0.1:
        p(f"  MODERATE BIAS: {report.length_wins_rate:.0%} of padded trials won by length")
    else:
        p(f"  LOW/NO BIAS: {report.length_wins_rate:.0%} of padded trials won by length")

    # Summary
    p("\n── OVERALL ──")
    p(f"  Judge calls:  {report.total_judge_calls}")
    p(f"  Total cost:   ${report.total_cost_usd:.4f}")
    p(f"  Duration:     {report.duration_seconds:.1f}s")

    # Overall signal
    p("\n  ── Overall Signal Assessment ──")
    signals = []
    if report.flip_rate > 0.1:
        signals.append(f"position bias ({report.flip_rate:.0%} flip rate)")
    if report.length_wins_rate > 0.1:
        signals.append(f"verbosity bias ({report.length_wins_rate:.0%} length-wins)")
    if abs(report.position_a_preference - 0.5) > 0.2:
        signals.append(f"directional preference ({report.position_a_preference:.0%} A-rate)")

    if signals:
        strength = "STRONG" if len(signals) >= 2 else "MODERATE"
        p(f"  Signal: {strength} — biases detected: {', '.join(signals)}")
        p(f"  RECOMMENDATION: Implement debiasing before using this judge as ground truth")
    else:
        p(f"  Signal: WEAK/NONE — no significant biases detected at this sample size")
        p(f"  RECOMMENDATION: Judge appears fair, but validate with larger sample")

    p("\n" + "=" * 90)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 5.04: Position & verbosity bias")
    print("Hypothesis: LLM judge prefers longer or first-presented options")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    # Use the same model for judging (as the experiment tests the judge itself)
    judge_backend = AnthropicJudgeBackend(client=client, model=settings.default_model)
    judge = LLMJudge(backend=judge_backend, model=settings.default_model)

    # Step 1: Generate personas
    print("\n[1/4] Generating personas from pipeline...")
    t0 = time.monotonic()
    personas = await generate_personas(synth_backend)
    print(f"      Generated {len(personas)} personas")
    for p in personas:
        print(f"        - {p.get('name', 'unnamed')}")

    # Step 2: Position bias test
    print("\n[2/4] Testing position bias (A/B swap)...")
    position_results = await test_position_bias(judge, personas)
    flips = sum(1 for r in position_results if r.flipped)
    print(f"      {flips}/{len(position_results)} pairs flipped on swap")

    # Step 3: Verbosity bias test
    print("\n[3/4] Testing verbosity bias (filler padding)...")
    verbosity_results = await test_verbosity_bias(judge, personas)
    length_wins = sum(1 for r in verbosity_results if r.length_won)
    print(f"      {length_wins}/{len(verbosity_results)} trials won by padded version")

    duration = time.monotonic() - t0

    # Step 4: Report
    print("\n[4/4] Computing report...")
    report = compute_report(position_results, verbosity_results, judge_backend, duration)
    report_text = print_report(report)

    # Save results
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "5.04",
        "title": "Position & verbosity bias",
        "hypothesis": "LLM judge systematically prefers longer or first-presented options",
        "model": settings.default_model,
        "position_bias": {
            "flip_rate": report.flip_rate,
            "a_preference": report.position_a_preference,
            "trials": [
                {
                    "pair": r.pair_label,
                    "order_ab_winner": r.order_ab_winner,
                    "order_ba_winner": r.order_ba_winner,
                    "flipped": r.flipped,
                    "consistent_winner": r.consistent_winner,
                    "rationale_ab": r.order_ab_rationale[:200],
                    "rationale_ba": r.order_ba_rationale[:200],
                }
                for r in position_results
            ],
        },
        "verbosity_bias": {
            "length_wins_rate": report.length_wins_rate,
            "trials": [
                {
                    "pair": r.pair_label,
                    "padded_side": r.padded_side,
                    "original_winner": r.original_winner,
                    "padded_winner": r.padded_winner,
                    "length_won": r.length_won,
                    "rationale_original": r.original_rationale[:200],
                    "rationale_padded": r.padded_rationale[:200],
                }
                for r in verbosity_results
            ],
        },
        "total_cost_usd": report.total_cost_usd,
        "duration_seconds": report.duration_seconds,
    }

    results_path = output_dir / "exp_5_04_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))

    report_path = output_dir / "exp_5_04_report.txt"
    report_path.write_text(report_text)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
