"""Experiment 4.03: Drift over turn count.

Hypothesis: Twin personas maintain strong in-character behavior for the
first 5-10 turns, then gradually drift. The drift curve has a measurable
half-life that characterizes the runtime's consistency ceiling.

Setup:
  1. Synthesize personas from the golden tenant.
  2. Run 50-turn scripted conversations with each persona.
  3. Score vocabulary overlap at checkpoints: turns 1, 5, 10, 25, 50.
  4. Compute the drift half-life for each persona.

Metrics:
  - Vocab overlap at each checkpoint (0.0-1.0)
  - Drift half-life (turn where overlap drops below 50% of baseline)
  - Decay rate (overlap loss per turn)

Usage:
    python scripts/experiment_4_03.py
"""

from __future__ import annotations

import asyncio
import json
import math
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
sys.path.insert(0, str(REPO_ROOT / "evals"))
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
from twin.conversations import SCRIPTED_QUESTIONS, DRIFT_CHECKPOINTS  # noqa: E402
from drift_curve import (  # noqa: E402
    DriftCurve,
    compute_drift_curve,
    extract_persona_words,
    score_turn,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

MAX_TURNS = 50  # full conversation length


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

def print_results(curves: list[DriftCurve], total_cost: float) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 4.03 — DRIFT OVER TURN COUNT — RESULTS")
    p("=" * 100)

    # Drift curves table
    p(f"\n-- DRIFT CURVES (vocab overlap at each checkpoint) --")
    header = f"  {'Persona':<35}"
    for cp in DRIFT_CHECKPOINTS:
        header += f"{'T' + str(cp):>10}"
    header += f"{'Half-life':>12}{'Decay/turn':>12}"
    p(header)
    p("  " + "-" * (35 + 10 * len(DRIFT_CHECKPOINTS) + 24))

    for curve in curves:
        short_name = curve.persona_name[:33]
        row = f"  {short_name:<35}"
        for cp_target in DRIFT_CHECKPOINTS:
            cp_match = next((c for c in curve.checkpoints if c.turn == cp_target), None)
            if cp_match:
                row += f"{cp_match.vocab_overlap:>10.3f}"
            else:
                row += f"{'N/A':>10}"
        hl = str(curve.half_life) if curve.half_life > 0 else "never"
        row += f"{hl:>12}"
        row += f"{curve.decay_rate:>12.5f}"
        p(row)

    # Per-persona detail
    for curve in curves:
        p(f"\n-- {curve.persona_name} ({curve.persona_word_count} persona words) --")
        for cp in curve.checkpoints:
            bar_len = int(cp.vocab_overlap * 50)
            bar = "#" * bar_len + "." * (50 - bar_len)
            p(f"  T{cp.turn:>3}: {cp.vocab_overlap:.3f} [{bar}]")
            p(f"        {cp.response_snippet}...")

    # Aggregate stats
    p("\n-- AGGREGATE STATS --")
    if curves:
        baselines = [c.baseline_overlap for c in curves]
        finals = [c.final_overlap for c in curves]
        decays = [c.decay_rate for c in curves]
        half_lives = [c.half_life for c in curves if c.half_life > 0]

        p(f"  Avg baseline overlap (T1):   {statistics.mean(baselines):.3f}")
        p(f"  Avg final overlap (T50):     {statistics.mean(finals):.3f}")
        p(f"  Avg decay per turn:          {statistics.mean(decays):.5f}")
        p(f"  Total overlap loss:          {statistics.mean(baselines) - statistics.mean(finals):.3f}")

        if half_lives:
            p(f"  Half-lives:                  {half_lives}")
            p(f"  Avg half-life:               {statistics.mean(half_lives):.0f} turns")
        else:
            p(f"  Half-life:                   NEVER reached (personas stay in character)")

    # Signal assessment
    p("\n-- SIGNAL ASSESSMENT --")
    if curves:
        avg_decay = statistics.mean([c.decay_rate for c in curves])
        avg_baseline = statistics.mean([c.baseline_overlap for c in curves])
        avg_final = statistics.mean([c.final_overlap for c in curves])
        total_drop = avg_baseline - avg_final
        half_lives = [c.half_life for c in curves if c.half_life > 0]

        if half_lives and statistics.mean(half_lives) < 15:
            strength = "STRONG FINDING"
            detail = (f"Personas drift quickly — half-life of "
                      f"{statistics.mean(half_lives):.0f} turns. Significant "
                      f"character loss by mid-conversation.")
        elif total_drop > 0.05:
            strength = "MODERATE FINDING"
            detail = (f"Measurable drift detected: overlap drops {total_drop:.3f} "
                      f"over 50 turns ({avg_baseline:.3f} -> {avg_final:.3f}). "
                      f"Drift is gradual but real.")
        elif total_drop > 0.02:
            strength = "WEAK FINDING"
            detail = (f"Minor drift: overlap drops {total_drop:.3f} over 50 turns. "
                      f"Personas mostly maintain character.")
        else:
            strength = "NULL RESULT"
            detail = (f"No meaningful drift: overlap stable at {avg_final:.3f} "
                      f"through 50 turns. Personas are highly consistent.")

        p(f"\n  Signal: {strength}")
        p(f"  {detail}")

    p(f"\n  Total conversation cost: ${total_cost:.4f}")
    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 4.03: Drift over turn count")
    print("Hypothesis: Twin personas drift out of character over long")
    print("  conversations; measuring the half-life of this drift")
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

    print("\n[2/4] Synthesizing personas...")
    personas: list[dict] = []
    for cluster in clusters:
        try:
            result = await synthesize(cluster, synth_backend)
            pd = result.persona.model_dump(mode="json")
            pd["_cluster_id"] = cluster.cluster_id
            personas.append(pd)
            pw = extract_persona_words(pd)
            print(f"    {result.persona.name}: {len(pw)} persona words")
        except Exception as e:
            print(f"    FAILED: {e}")

    if not personas:
        print("ERROR: No personas synthesized")
        sys.exit(1)

    print(f"\n[3/4] Running {MAX_TURNS}-turn conversations...")
    all_curves: list[DriftCurve] = []
    total_cost = 0.0

    for persona in personas:
        name = persona.get("name", "?")
        persona_words = extract_persona_words(persona)
        print(f"\n  {name} ({len(persona_words)} words)")

        twin = TwinChat(persona, client=client, model=model)
        history: list[dict] = []
        responses: list[str] = []

        questions = SCRIPTED_QUESTIONS[:MAX_TURNS]

        for i, question in enumerate(questions):
            turn_num = i + 1
            reply = await twin.reply(question, history=history)
            responses.append(reply.text)
            total_cost += reply.estimated_cost_usd
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": reply.text})

            if turn_num in DRIFT_CHECKPOINTS:
                overlap = score_turn(reply.text, persona_words)
                print(f"    T{turn_num:>3}: overlap={overlap:.3f} "
                      f"cost_so_far=${total_cost:.4f}")

        curve = compute_drift_curve(responses, persona_words, DRIFT_CHECKPOINTS)
        curve.persona_name = name
        all_curves.append(curve)

        hl = curve.half_life if curve.half_life > 0 else "never"
        print(f"    Half-life: {hl}, decay={curve.decay_rate:.5f}/turn")

    print("\n[4/4] Generating report...")
    report = print_results(all_curves, total_cost)

    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment": "4.03",
        "title": "Drift over turn count",
        "hypothesis": "Persona drift has a measurable half-life",
        "model": model,
        "max_turns": MAX_TURNS,
        "checkpoints": DRIFT_CHECKPOINTS,
        "curves": [
            {
                "persona_name": c.persona_name,
                "persona_word_count": c.persona_word_count,
                "baseline_overlap": c.baseline_overlap,
                "final_overlap": c.final_overlap,
                "half_life": c.half_life,
                "decay_rate": c.decay_rate,
                "checkpoints": [
                    {
                        "turn": cp.turn,
                        "vocab_overlap": cp.vocab_overlap,
                        "response_snippet": cp.response_snippet,
                    }
                    for cp in c.checkpoints
                ],
            }
            for c in all_curves
        ],
        "total_cost_usd": total_cost,
    }

    results_path = output_dir / "exp_4_03_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_4_03_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
