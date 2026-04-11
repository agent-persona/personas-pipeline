"""Experiment 1.10 — Value-level provenance.

Compare the baseline PersonaV1 (separate source_evidence array) against
PersonaV1Provenance (inline provenance on each value field).

Measures: prompt cost delta, groundedness, and auditability (% of claims
with inline provenance vs requiring cross-reference).

Usage:
    python evals/value_provenance.py
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
from pydantic import ValidationError
from crawler import fetch_all
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_user_message
from synthesis.engine.synthesizer import synthesize, SynthesisError
from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1
from synthesis.models.persona_provenance import PersonaV1Provenance

PROVENANCE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

IMPORTANT SCHEMA CHANGE: In this version, goals, pains, motivations, and
objections are NOT plain strings. Each is an object with:
  {"text": "the claim", "source_record_ids": ["id1", "id2"], "confidence": 0.85, "model_version": "claude-haiku-4-5-20251001"}

You do NOT need a separate source_evidence array. The provenance is inline
on each value. Every goal, pain, motivation, and objection must include at
least one valid source_record_id from the provided data.
"""


def check_provenance_groundedness(persona: PersonaV1Provenance, cluster: ClusterData) -> tuple[float, list[str]]:
    """Walk the provenance tree and check all record IDs are valid."""
    valid_ids = set(cluster.all_record_ids)
    violations = []
    total = 0
    covered = 0

    for field_name in ("goals", "pains", "motivations", "objections"):
        items = getattr(persona, field_name)
        for i, item in enumerate(items):
            total += 1
            bad_ids = [rid for rid in item.source_record_ids if rid not in valid_ids]
            if bad_ids:
                violations.append(f"{field_name}.{i}: invalid record IDs {bad_ids}")
            else:
                covered += 1

    score = covered / total if total > 0 else 1.0
    return score, violations


async def run_baseline(cluster: ClusterData, backend: AnthropicBackend):
    """Standard PersonaV1 synthesis."""
    result = await synthesize(cluster, backend)
    return result.persona, result.groundedness.score, result.total_cost_usd, result.attempts


async def run_provenance(cluster: ClusterData, client: AsyncAnthropic, model: str):
    """Synthesis with PersonaV1Provenance schema."""
    tool = {
        "name": "create_persona",
        "description": "Create a persona with inline provenance on each value field.",
        "input_schema": PersonaV1Provenance.model_json_schema(),
    }
    user_msg = build_user_message(cluster)
    total_cost = 0.0

    for attempt in range(3):
        resp = await client.messages.create(
            model=model, max_tokens=4096,
            system=PROVENANCE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
        )
        tool_block = next(b for b in resp.content if b.type == "tool_use")
        cost = (resp.usage.input_tokens * 1 + resp.usage.output_tokens * 5) / 1_000_000
        total_cost += cost
        try:
            persona = PersonaV1Provenance.model_validate(tool_block.input)
            g_score, violations = check_provenance_groundedness(persona, cluster)
            if g_score >= 0.9:
                return persona, g_score, total_cost, attempt + 1
        except ValidationError:
            continue
    return None, 0.0, total_cost, 3


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
            persona, g, cost, att = await run_baseline(cluster, backend)
            n_evidence = len(persona.source_evidence)
            print(f" {persona.name} (g={g:.2f}, evidence={n_evidence}, ${cost:.4f})")
            results.append({"mode": "baseline", "cluster": cid,
                "persona": persona.name, "groundedness": g, "cost": cost,
                "attempts": att, "n_evidence_entries": n_evidence,
                "n_goals": len(persona.goals), "failed": False})
        except SynthesisError:
            print(" FAILED")
            results.append({"mode": "baseline", "cluster": cid,
                "persona": "FAILED", "groundedness": 0, "cost": 0,
                "attempts": 3, "n_evidence_entries": 0, "n_goals": 0, "failed": True})

        # Provenance
        print(f"[provenance] {cid}...", end="", flush=True)
        persona_p, g_p, cost_p, att_p = await run_provenance(cluster, client, model)
        if persona_p:
            n_inline = sum(len(v.source_record_ids) for field in ("goals","pains","motivations","objections")
                          for v in getattr(persona_p, field))
            print(f" {persona_p.name} (g={g_p:.2f}, inline_refs={n_inline}, ${cost_p:.4f})")
            results.append({"mode": "provenance", "cluster": cid,
                "persona": persona_p.name, "groundedness": g_p, "cost": cost_p,
                "attempts": att_p, "n_inline_refs": n_inline,
                "n_goals": len(persona_p.goals), "failed": False,
                "sample_goal": persona_p.goals[0].model_dump() if persona_p.goals else {}})
        else:
            print(f" FAILED (${cost_p:.4f})")
            results.append({"mode": "provenance", "cluster": cid,
                "persona": "FAILED", "groundedness": 0, "cost": cost_p,
                "attempts": att_p, "n_inline_refs": 0, "n_goals": 0, "failed": True})

    print("\n" + "=" * 80)
    print("EXPERIMENT 1.10 -- VALUE-LEVEL PROVENANCE")
    print("=" * 80)

    print(f"\n{'Mode':<12} {'Cluster':<16} {'Persona':<26} {'G':>5} {'Att':>3} {'Cost':>8} {'Evidence':>8}")
    print("-" * 80)
    for r in results:
        if r["failed"]:
            print(f"{r['mode']:<12} {r['cluster']:<16} {'FAILED':<26}")
        else:
            ev = r.get("n_evidence_entries", r.get("n_inline_refs", 0))
            label = "entries" if r["mode"] == "baseline" else "inline"
            print(f"{r['mode']:<12} {r['cluster']:<16} {r['persona'][:24]:<26} "
                  f"{r['groundedness']:>5.2f} {r['attempts']:>3} ${r['cost']:>7.4f} {ev:>4} {label}")

    # Cost comparison
    base_cost = sum(r["cost"] for r in results if r["mode"] == "baseline")
    prov_cost = sum(r["cost"] for r in results if r["mode"] == "provenance")
    delta = ((prov_cost - base_cost) / base_cost * 100) if base_cost else 0
    print(f"\nCost: baseline=${base_cost:.4f}, provenance=${prov_cost:.4f} ({delta:+.0f}%)")

    # Show sample provenanced goal
    for r in results:
        if r["mode"] == "provenance" and not r["failed"] and r.get("sample_goal"):
            print(f"\nSample provenanced goal:")
            print(f"  {json.dumps(r['sample_goal'], indent=2)}")
            break

    out = REPO_ROOT / "output" / "exp_1_10_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
