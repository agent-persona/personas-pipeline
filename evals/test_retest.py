"""Experiment 5.16 — Test-retest reliability.

Run the same persona through the same LLM judge multiple times at
different temperatures to measure score consistency.

Usage:
    python evals/test_retest.py
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
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
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import synthesize
from synthesis.models.cluster import ClusterData

DIMS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")

JUDGE_SYSTEM = """\
You are an expert persona evaluator. Score on a 1-5 scale.
Respond with ONLY a JSON object:
{"grounded":<1-5>,"distinctive":<1-5>,"coherent":<1-5>,"actionable":<1-5>,"voice_fidelity":<1-5>,"overall":<1-5>}
"""


async def judge_once(client: AsyncAnthropic, model: str, persona: dict, temperature: float) -> dict[str, float]:
    prompt = "Score this persona:\n" + json.dumps(persona, indent=2, default=str)
    resp = await client.messages.create(
        model=model, max_tokens=256, temperature=temperature,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    scores = {}
    for d in DIMS:
        scores[d] = float(data.get(d, 0))
    scores["overall"] = float(data.get("overall", 0))
    return scores


def std_dev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)
    model = settings.default_model
    print(f"Model: {model}")

    records = fetch_all("tenant_acme_corp")
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]
    clusters_raw = segment(raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool", existing_persona_names=[],
        similarity_threshold=0.15, min_cluster_size=2)
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=model)

    # Synthesize one persona
    print("Synthesizing persona...")
    result = await synthesize(clusters[0], backend)
    persona_dict = result.persona.model_dump(mode="json")
    print(f"  {result.persona.name}")

    # Judge at two temperatures
    n_runs = 5
    results_by_temp: dict[float, list[dict[str, float]]] = {}
    for temp in [0.0, 0.5]:
        print(f"\nJudging {n_runs}x at temperature={temp}...")
        runs = []
        for i in range(n_runs):
            scores = await judge_once(client, model, persona_dict, temp)
            runs.append(scores)
            print(f"  Run {i+1}: overall={scores['overall']}")
        results_by_temp[temp] = runs

    # Report
    print("\n" + "=" * 80)
    print("EXPERIMENT 5.16 — TEST-RETEST RELIABILITY")
    print("=" * 80)

    all_dims = list(DIMS) + ["overall"]
    print(f"\n{'Dim':<16} {'T=0 Mean':>8} {'T=0 SD':>7} {'T=0 Rng':>7} {'T=0.5 Mean':>10} {'T=0.5 SD':>8} {'T=0.5 Rng':>9}")
    print("-" * 75)

    for d in all_dims:
        v0 = [r[d] for r in results_by_temp[0.0]]
        v5 = [r[d] for r in results_by_temp[0.5]]
        m0, s0, r0 = sum(v0)/len(v0), std_dev(v0), max(v0)-min(v0)
        m5, s5, r5 = sum(v5)/len(v5), std_dev(v5), max(v5)-min(v5)
        print(f"{d:<16} {m0:>8.2f} {s0:>7.3f} {r0:>7.1f} {m5:>10.2f} {s5:>8.3f} {r5:>9.1f}")

    total_sd_0 = std_dev([r[d] for r in results_by_temp[0.0] for d in all_dims])
    total_sd_5 = std_dev([r[d] for r in results_by_temp[0.5] for d in all_dims])
    print(f"\nOverall SD: T=0 -> {total_sd_0:.3f}, T=0.5 -> {total_sd_5:.3f}")
    print(f"T=0 is {'MORE' if total_sd_0 < total_sd_5 else 'LESS'} reliable than T=0.5")

    # Save
    out = REPO_ROOT / "output" / "exp_5_16_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"persona": result.persona.name,
        "temp_0": results_by_temp[0.0], "temp_05": results_by_temp[0.5]}, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
