"""Experiment 2.15 — Distillation: exemplar-primed synthesis.

Since only Haiku is available, tests whether providing a high-quality
exemplar persona in the prompt (simulating distillation from a stronger
model) improves Haiku's output quality vs baseline Haiku with no exemplar.

Usage:
    python evals/distillation.py
"""

from __future__ import annotations

import asyncio
import json
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
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_tool_definition, build_user_message
from synthesis.engine.synthesizer import synthesize, SynthesisError
from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1
from pydantic import ValidationError

# A hand-crafted exemplar (simulating Opus-quality output)
EXEMPLAR = """\

## Quality Exemplar
Here is an example of an excellent persona synthesized from similar data. Use it
as a quality reference — match this level of specificity, grounding, and voice:

{
  "name": "DevOps Dana - The Automation-First Platform Lead",
  "summary": "A senior platform engineer at a 150-person fintech company who treats every manual process as a bug.",
  "goals": ["Automate deployment pipelines to <10min cycles across 4 teams", "Consolidate observability into one Datadog dashboard"],
  "pains": ["5+ hours/week debugging flaky CI tests other teams wrote", "Every team has its own deployment script"],
  "vocabulary": ["toil", "blast radius", "golden path", "SLO", "error budget"],
  "sample_quotes": ["If I can't terraform it, it doesn't exist.", "That's not engineering, that's archaeology."]
}

Now synthesize a persona of THIS quality from the data below. Be equally specific —
quantify where possible, name real tools, and write quotes that could only come from
this one person.
"""


async def run_baseline(cluster, backend):
    """Standard synthesis, no exemplar."""
    result = await synthesize(cluster, backend)
    return result


async def run_with_exemplar(cluster, client, model):
    """Synthesis with exemplar injected into user message."""
    tool = build_tool_definition()
    user_msg = EXEMPLAR + "\n" + build_user_message(cluster)
    for attempt in range(3):
        resp = await client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
        )
        tool_block = next(b for b in resp.content if b.type == "tool_use")
        cost = (resp.usage.input_tokens * 1 + resp.usage.output_tokens * 5) / 1_000_000
        try:
            persona = PersonaV1.model_validate(tool_block.input)
            g = check_groundedness(persona, cluster)
            if g.passed:
                return persona, g.score, cost, attempt + 1
        except ValidationError:
            pass
    return None, 0.0, cost * 3, 3


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = settings.default_model
    backend = AnthropicBackend(client=client, model=model)

    records = fetch_all("tenant_acme_corp")
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]
    clusters_raw = segment(raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool", existing_persona_names=[],
        similarity_threshold=0.15, min_cluster_size=2)
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    print(f"Clusters: {len(clusters)}\n")

    results = []
    for cluster in clusters:
        cid = cluster.cluster_id[:14]
        # Baseline
        print(f"[baseline] {cid}...", end="", flush=True)
        try:
            r = await run_baseline(cluster, backend)
            print(f" {r.persona.name} (g={r.groundedness.score:.2f}, ${r.total_cost_usd:.4f})")
            base = {"mode": "baseline", "cluster": cid, "persona": r.persona.name,
                    "groundedness": r.groundedness.score, "cost": r.total_cost_usd,
                    "attempts": r.attempts, "goals": r.persona.goals[:3],
                    "avg_goal_len": sum(len(g) for g in r.persona.goals)/len(r.persona.goals),
                    "failed": False}
        except SynthesisError:
            print(" FAILED")
            base = {"mode": "baseline", "cluster": cid, "persona": "FAILED",
                    "groundedness": 0, "cost": 0, "attempts": 3, "goals": [],
                    "avg_goal_len": 0, "failed": True}

        # With exemplar
        print(f"[exemplar] {cid}...", end="", flush=True)
        persona, g, cost, att = await run_with_exemplar(cluster, client, model)
        if persona:
            print(f" {persona.name} (g={g:.2f}, ${cost:.4f})")
            exem = {"mode": "exemplar", "cluster": cid, "persona": persona.name,
                    "groundedness": g, "cost": cost, "attempts": att,
                    "goals": persona.goals[:3],
                    "avg_goal_len": sum(len(g) for g in persona.goals)/len(persona.goals),
                    "failed": False}
        else:
            print(f" FAILED (${cost:.4f})")
            exem = {"mode": "exemplar", "cluster": cid, "persona": "FAILED",
                    "groundedness": 0, "cost": cost, "attempts": att, "goals": [],
                    "avg_goal_len": 0, "failed": True}

        results.extend([base, exem])

    print("\n" + "=" * 80)
    print("EXPERIMENT 2.15 -- DISTILLATION (exemplar-primed synthesis)")
    print("=" * 80)
    print(f"\n{'Mode':<12} {'Cluster':<16} {'Persona':<28} {'G':>5} {'Att':>3} {'Cost':>8} {'GoalLen':>7}")
    print("-" * 80)
    for r in results:
        if r["failed"]:
            print(f"{r['mode']:<12} {r['cluster']:<16} {'FAILED':<28}")
        else:
            print(f"{r['mode']:<12} {r['cluster']:<16} {r['persona'][:26]:<28} "
                  f"{r['groundedness']:>5.2f} {r['attempts']:>3} ${r['cost']:>7.4f} {r['avg_goal_len']:>7.0f}")

    out = REPO_ROOT / "output" / "exp_2_15_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
