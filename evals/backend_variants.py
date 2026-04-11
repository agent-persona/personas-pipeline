"""Experiment 2.04 — Tool-use vs JSON mode vs free-text+parser.

Compare three approaches to getting valid persona JSON from the LLM:
  A) Tool-use forcing (current baseline)
  B) Plain JSON instruction (no tool, ask for JSON in prompt)
  C) Free-text generation + regex/JSON extraction

Metric: pre-retry schema validity, retry cost, content richness.

Usage:
    python evals/backend_variants.py
"""

from __future__ import annotations

import asyncio
import json
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
from pydantic import ValidationError
from crawler import fetch_all
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.groundedness import check_groundedness
from synthesis.engine.model_backend import AnthropicBackend, LLMResult
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_tool_definition, build_user_message
from synthesis.engine.synthesizer import synthesize
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

JSON_INSTRUCTION = """

IMPORTANT: Respond with ONLY a valid JSON object matching the PersonaV1 schema.
Do not include any text before or after the JSON. No markdown fences.
The JSON must include all required fields: schema_version, name, summary,
demographics, firmographics, goals, pains, motivations, objections, channels,
vocabulary, decision_triggers, sample_quotes, journey_stages, source_evidence.
"""


async def variant_tool_use(cluster: ClusterData, client: AsyncAnthropic, model: str):
    """Variant A: Tool-use forcing (baseline)."""
    backend = AnthropicBackend(client=client, model=model)
    result = await synthesize(cluster, backend)
    return result.persona, result.groundedness.score, result.total_cost_usd, result.attempts


async def variant_json_mode(cluster: ClusterData, client: AsyncAnthropic, model: str):
    """Variant B: Ask for JSON in the prompt, no tool forcing."""
    user_msg = build_user_message(cluster) + JSON_INSTRUCTION
    total_cost = 0.0
    for attempt in range(3):
        resp = await client.messages.create(
            model=model, max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        cost = (resp.usage.input_tokens * 1 + resp.usage.output_tokens * 5) / 1_000_000
        total_cost += cost
        # Try to parse JSON
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            persona = PersonaV1.model_validate(data)
            g = check_groundedness(persona, cluster)
            return persona, g.score, total_cost, attempt + 1
        except (json.JSONDecodeError, ValidationError):
            continue
    return None, 0.0, total_cost, 3


async def variant_freetext(cluster: ClusterData, client: AsyncAnthropic, model: str):
    """Variant C: Free-text generation, extract JSON from response."""
    user_msg = build_user_message(cluster) + (
        "\n\nDescribe this persona in detail, then provide the structured JSON "
        "representation at the end of your response inside ```json fences."
    )
    total_cost = 0.0
    for attempt in range(3):
        resp = await client.messages.create(
            model=model, max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text
        cost = (resp.usage.input_tokens * 1 + resp.usage.output_tokens * 5) / 1_000_000
        total_cost += cost
        # Extract JSON from fences
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"\{[\s\S]*\"schema_version\"[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(1) if match.lastindex else match.group())
                persona = PersonaV1.model_validate(data)
                g = check_groundedness(persona, cluster)
                return persona, g.score, total_cost, attempt + 1
            except (json.JSONDecodeError, ValidationError):
                pass
        continue
    return None, 0.0, total_cost, 3


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = settings.default_model

    records = fetch_all("tenant_acme_corp")
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]
    clusters_raw = segment(raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool", existing_persona_names=[],
        similarity_threshold=0.15, min_cluster_size=2)
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    cluster = clusters[0]  # Use first cluster for all variants
    print(f"Cluster: {cluster.cluster_id} ({len(cluster.sample_records)} records)\n")

    variants = [
        ("tool-use", variant_tool_use),
        ("json-mode", variant_json_mode),
        ("free+parse", variant_freetext),
    ]

    results = []
    for name, fn in variants:
        print(f"--- {name} ---")
        t0 = time.monotonic()
        try:
            persona, g_score, cost, attempts = await fn(cluster, client, model)
            elapsed = time.monotonic() - t0
            if persona:
                n_goals = len(persona.goals)
                avg_goal_len = sum(len(g) for g in persona.goals) / n_goals if n_goals else 0
                print(f"  {persona.name} (g={g_score:.2f}, att={attempts}, ${cost:.4f}, {elapsed:.1f}s)")
                results.append({
                    "variant": name, "persona": persona.name, "groundedness": g_score,
                    "attempts": attempts, "cost": cost, "elapsed_s": elapsed,
                    "n_goals": n_goals, "avg_goal_len": avg_goal_len,
                    "goals": persona.goals[:3], "failed": False,
                })
            else:
                print(f"  FAILED after {attempts} attempts (${cost:.4f}, {elapsed:.1f}s)")
                results.append({
                    "variant": name, "persona": "FAILED", "groundedness": 0,
                    "attempts": attempts, "cost": cost, "elapsed_s": elapsed,
                    "n_goals": 0, "avg_goal_len": 0, "goals": [], "failed": True,
                })
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  ERROR: {e}")
            results.append({
                "variant": name, "persona": "ERROR", "groundedness": 0,
                "attempts": 0, "cost": 0, "elapsed_s": elapsed,
                "n_goals": 0, "avg_goal_len": 0, "goals": [], "failed": True,
            })

    print("\n" + "=" * 80)
    print("EXPERIMENT 2.04 -- TOOL-USE vs JSON-MODE vs FREE+PARSER")
    print("=" * 80)
    print(f"\n{'Variant':<12} {'Persona':<30} {'Ground':>6} {'Att':>3} {'Cost':>8} {'Goals':>5} {'AvgLen':>6}")
    print("-" * 75)
    for r in results:
        if r["failed"]:
            print(f"{r['variant']:<12} {'FAILED':<30} {'--':>6} {r['attempts']:>3} ${r['cost']:>7.4f}")
        else:
            print(f"{r['variant']:<12} {r['persona'][:28]:<30} {r['groundedness']:>6.2f} "
                  f"{r['attempts']:>3} ${r['cost']:>7.4f} {r['n_goals']:>5} {r['avg_goal_len']:>6.0f}")

    out = REPO_ROOT / "output" / "exp_2_04_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
