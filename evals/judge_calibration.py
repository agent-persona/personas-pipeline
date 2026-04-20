"""Experiment 5.13 — Few-shot judge calibration.

Synthesizes personas from the standard tenant, then scores each persona
twice: once with the baseline zero-shot judge ("none") and once with
few-shot calibration anchors ("few_shot"). Compares score distributions
to measure whether anchoring tightens the spread.

Usage:
    python evals/judge_calibration.py
"""

from __future__ import annotations

import asyncio
import json
import math
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
from evaluation.judges import CalibrationMode, JudgeBackend, JudgeScore, LLMJudge
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import SynthesisError, synthesize
from synthesis.models.cluster import ClusterData

TENANT_ID = "tenant_acme_corp"


@dataclass
class JudgingResult:
    cluster_id: str
    persona_name: str
    calibration: str
    overall: float
    dimensions: dict[str, float]
    rationale: str
    cost_note: str = ""


def std_dev(values: list[float]) -> float:
    """Population standard deviation."""
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    return (sum((x - mean) ** 2 for x in clean) / len(clean)) ** 0.5


def print_results(results: list[JudgingResult]) -> None:
    """Print comparison table and distribution analysis."""
    modes = ["none", "few_shot"]
    dims = ["grounded", "distinctive", "coherent", "actionable", "voice_fidelity", "overall"]

    print("\n" + "=" * 110)
    print("EXPERIMENT 5.13 — FEW-SHOT JUDGE CALIBRATION")
    print("=" * 110)

    # Per-persona scores
    print(f"\n{'Cluster':<16} {'Persona':<30} {'Mode':<10} ", end="")
    for d in dims:
        print(f"{d[:6]:>7}", end="")
    print()
    print("-" * 110)

    clusters = sorted(set(r.cluster_id for r in results))
    for cid in clusters:
        cluster_results = [r for r in results if r.cluster_id == cid]
        for mode in modes:
            r = next((x for x in cluster_results if x.calibration == mode), None)
            if not r:
                continue
            label = r.persona_name[:28] if mode == "none" else '  "  (calibrated)'
            print(f"{cid[:14]:<16} {label:<30} {mode:<10} ", end="")
            for d in dims:
                val = r.dimensions.get(d, r.overall) if d != "overall" else r.overall
                if math.isnan(val):
                    print(f"{'NaN':>7}", end="")
                else:
                    print(f"{val:>7.1f}", end="")
            print()
        print()

    # Distribution analysis
    print("=" * 110)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 110)
    print(f"\n{'Mode':<12} {'Dimension':<16} {'Mean':>6} {'StdDev':>7} {'Min':>5} {'Max':>5} {'Range':>6} {'N':>3}")
    print("-" * 70)

    for mode in modes:
        mode_results = [r for r in results if r.calibration == mode]
        for dim in dims:
            if dim == "overall":
                values = [r.overall for r in mode_results]
            else:
                values = [r.dimensions.get(dim, float("nan")) for r in mode_results]
            clean = [v for v in values if not math.isnan(v)]
            if not clean:
                continue
            mean = sum(clean) / len(clean)
            sd = std_dev(clean)
            lo = min(clean)
            hi = max(clean)
            rng = hi - lo
            print(f"{mode:<12} {dim:<16} {mean:>6.2f} {sd:>7.2f} {lo:>5.1f} {hi:>5.1f} {rng:>6.1f} {len(clean):>3}")
        print()

    # Comparison summary
    print("=" * 110)
    print("TIGHTENING SUMMARY (few_shot vs none)")
    print("=" * 110)
    print(f"\n{'Dimension':<16} {'Baseline SD':>11} {'Calibrated SD':>13} {'Change':>8} {'Tighter?':>9}")
    print("-" * 62)

    for dim in dims:
        for mode in modes:
            mode_results = [r for r in results if r.calibration == mode]
            if dim == "overall":
                values = [r.overall for r in mode_results]
            else:
                values = [r.dimensions.get(dim, float("nan")) for r in mode_results]
            clean = [v for v in values if not math.isnan(v)]
            if mode == "none":
                sd_none = std_dev(clean)
            else:
                sd_few = std_dev(clean)

        delta = sd_few - sd_none
        tighter = "YES" if delta < 0 else ("same" if delta == 0 else "no")
        print(f"{dim:<16} {sd_none:>11.3f} {sd_few:>13.3f} {delta:>+8.3f} {tighter:>9}")

    total_none = []
    total_few = []
    for dim in dims:
        for mode in modes:
            mode_results = [r for r in results if r.calibration == mode]
            if dim == "overall":
                values = [r.overall for r in mode_results]
            else:
                values = [r.dimensions.get(dim, float("nan")) for r in mode_results]
            clean = [v for v in values if not math.isnan(v)]
            if mode == "none":
                total_none.extend(clean)
            else:
                total_few.extend(clean)

    sd_all_none = std_dev(total_none)
    sd_all_few = std_dev(total_few)
    delta_all = sd_all_few - sd_all_none
    print("-" * 62)
    print(f"{'ALL SCORES':<16} {sd_all_none:>11.3f} {sd_all_few:>13.3f} {delta_all:>+8.3f} "
          f"{'YES' if delta_all < 0 else 'no':>9}")
    print()

    # Show rationales
    print("=" * 110)
    print("JUDGE RATIONALES")
    print("=" * 110)
    for r in results:
        print(f"\n  [{r.calibration.upper()}] {r.persona_name} ({r.cluster_id[:14]}):")
        print(f"    {r.rationale[:200]}")
    print()


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    print(f"Synthesis model: {settings.default_model}")
    print(f"Judge model: {settings.default_model}")

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
    synth_backend = AnthropicBackend(client=client, model=settings.default_model)

    # Step 1: Synthesize personas (one per cluster, shared across both judge modes)
    print("\n--- Synthesizing personas ---")
    personas: list[tuple[str, dict]] = []  # (cluster_id, persona_dict)
    for cluster in clusters:
        print(f"  {cluster.cluster_id}...", end="", flush=True)
        try:
            result = await synthesize(cluster, synth_backend)
            persona_dict = result.persona.model_dump(mode="json")
            personas.append((cluster.cluster_id, persona_dict))
            print(f" {result.persona.name} (g={result.groundedness.score:.2f})")
        except SynthesisError as e:
            print(f" FAILED: {e}")

    if not personas:
        print("No personas synthesized. Exiting.")
        sys.exit(1)

    # Step 2: Judge each persona in both modes
    print("\n--- Judging personas ---")
    judge_backend = JudgeBackend(client=client, model=settings.default_model)
    modes: list[CalibrationMode] = ["none", "few_shot"]
    results: list[JudgingResult] = []

    for cluster_id, persona_dict in personas:
        persona_name = persona_dict.get("name", "Unknown")
        for mode in modes:
            judge = LLMJudge(
                backend=judge_backend,
                model=settings.default_model,
                calibration=mode,
            )
            print(f"  Scoring {persona_name} [{mode}]...", end="", flush=True)
            t0 = time.monotonic()
            score = await judge.score_persona(persona_dict)
            elapsed = time.monotonic() - t0
            print(f" overall={score.overall:.1f} ({elapsed:.1f}s)")

            results.append(JudgingResult(
                cluster_id=cluster_id,
                persona_name=persona_name,
                calibration=mode,
                overall=score.overall,
                dimensions=score.dimensions,
                rationale=score.rationale,
            ))

    print_results(results)

    # Save results
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "exp_5_13_results.json"
    data = []
    for r in results:
        data.append({
            "cluster_id": r.cluster_id,
            "persona_name": r.persona_name,
            "calibration": r.calibration,
            "overall": r.overall,
            "dimensions": r.dimensions,
            "rationale": r.rationale,
        })
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
