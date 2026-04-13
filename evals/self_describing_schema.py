"""Experiment 1.06 — Self-describing schema.

Two-call synthesis: first the LLM proposes a per-tenant schema based on
the data, then fills it. Compare against the global PersonaV1 schema.

Hypothesis: emergent schemas are more grounded but break cross-persona
comparison because each persona has different fields.

Usage:
    python evals/self_describing_schema.py
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
from crawler import fetch_all
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.prompt_builder import build_user_message
from synthesis.engine.synthesizer import synthesize, SynthesisError
from synthesis.models.cluster import ClusterData

SCHEMA_PROPOSAL_SYSTEM = """\
You are a persona schema designer. Given behavioral data from a customer cluster,
propose a JSON schema that best captures the patterns in THIS specific data.

Rules:
- The schema should have 8-15 top-level fields
- Always include: name (string), summary (string)
- Other fields should emerge from the data — if the data shows strong technical
  signals, include tech-specific fields. If it shows budget concerns, include
  pricing-related fields.
- Each field needs a type (string, list[string], object) and a description
- Include a source_evidence field that maps claims to record IDs

Respond with ONLY a JSON object describing the schema:
{
  "schema_name": "descriptive name",
  "fields": {
    "field_name": {"type": "string|list|object", "description": "what this captures"},
    ...
  }
}
"""

FILL_SCHEMA_SYSTEM = """\
You are a persona synthesis expert. Fill in the provided custom schema using
ONLY information from the behavioral data. Every claim must trace to specific
record IDs from the sample records.

Respond with ONLY a valid JSON object matching the provided schema.
"""


async def propose_schema(cluster: ClusterData, client: AsyncAnthropic, model: str) -> dict | None:
    """Call 1: LLM proposes a per-tenant schema."""
    user_msg = build_user_message(cluster) + "\n\nPropose a custom schema for this data."
    resp = await client.messages.create(
        model=model, max_tokens=2048,
        system=SCHEMA_PROPOSAL_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text.strip()
    cost = (resp.usage.input_tokens * 1 + resp.usage.output_tokens * 5) / 1_000_000
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        schema = json.loads(text)
        return schema, cost
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group()), cost
            except json.JSONDecodeError:
                pass
    return None, cost


async def fill_schema(cluster: ClusterData, schema: dict, client: AsyncAnthropic, model: str) -> tuple[dict | None, float]:
    """Call 2: LLM fills the proposed schema with data."""
    user_msg = (
        f"## Custom Schema\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        + build_user_message(cluster)
        + "\n\nFill in the custom schema above using the data provided."
    )
    resp = await client.messages.create(
        model=model, max_tokens=4096,
        system=FILL_SCHEMA_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text.strip()
    cost = (resp.usage.input_tokens * 1 + resp.usage.output_tokens * 5) / 1_000_000
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), cost
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), cost
            except json.JSONDecodeError:
                pass
    return None, cost


def check_self_groundedness(persona: dict, cluster: ClusterData) -> tuple[float, int]:
    """Check how many record IDs in the persona are valid."""
    valid_ids = set(cluster.all_record_ids)
    text = json.dumps(persona)
    # Find all record ID references
    referenced = set(re.findall(r'(?:ga4|hubspot|intercom)_\d{3}', text))
    if not referenced:
        return 0.0, 0
    valid_refs = referenced & valid_ids
    return len(valid_refs) / len(referenced), len(referenced)


def schema_field_overlap(schemas: list[dict]) -> float:
    """Measure field overlap across proposed schemas (comparability)."""
    if len(schemas) < 2:
        return 1.0
    field_sets = []
    for s in schemas:
        fields = set(s.get("fields", {}).keys())
        field_sets.append(fields)
    # Pairwise Jaccard
    overlaps = []
    for i in range(len(field_sets)):
        for j in range(i + 1, len(field_sets)):
            union = field_sets[i] | field_sets[j]
            if union:
                overlaps.append(len(field_sets[i] & field_sets[j]) / len(union))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


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
    proposed_schemas = []

    for cluster in clusters:
        cid = cluster.cluster_id[:14]

        # Baseline: global schema
        print(f"[global] {cid}...", end="", flush=True)
        t0 = time.monotonic()
        try:
            r = await synthesize(cluster, backend)
            elapsed = time.monotonic() - t0
            print(f" {r.persona.name} (g={r.groundedness.score:.2f}, ${r.total_cost_usd:.4f}, {elapsed:.1f}s)")
            results.append({"mode": "global", "cluster": cid,
                "persona": r.persona.name, "groundedness": r.groundedness.score,
                "cost": r.total_cost_usd, "attempts": r.attempts,
                "n_fields": 14, "elapsed": elapsed, "failed": False})
        except SynthesisError:
            print(f" FAILED")
            results.append({"mode": "global", "cluster": cid,
                "persona": "FAILED", "groundedness": 0, "cost": 0,
                "attempts": 3, "n_fields": 14, "elapsed": 0, "failed": True})

        # Self-describing: two calls
        print(f"[self-desc] {cid}...", end="", flush=True)
        t0 = time.monotonic()
        schema, cost1 = await propose_schema(cluster, client, model)
        if schema:
            n_fields = len(schema.get("fields", {}))
            proposed_schemas.append(schema)
            print(f" schema={n_fields} fields...", end="", flush=True)
            persona, cost2 = await fill_schema(cluster, schema, client, model)
            total_cost = cost1 + cost2
            elapsed = time.monotonic() - t0
            if persona:
                g, n_refs = check_self_groundedness(persona, cluster)
                name = persona.get("name", "Unknown")
                print(f" {name} (g={g:.2f}, refs={n_refs}, ${total_cost:.4f}, {elapsed:.1f}s)")
                results.append({"mode": "self-desc", "cluster": cid,
                    "persona": name, "groundedness": g, "cost": total_cost,
                    "attempts": 1, "n_fields": n_fields, "elapsed": elapsed,
                    "schema_name": schema.get("schema_name", ""),
                    "field_names": list(schema.get("fields", {}).keys()),
                    "failed": False})
            else:
                print(f" fill FAILED (${total_cost:.4f})")
                results.append({"mode": "self-desc", "cluster": cid,
                    "persona": "FAILED", "groundedness": 0, "cost": total_cost,
                    "attempts": 1, "n_fields": n_fields, "elapsed": elapsed,
                    "failed": True})
        else:
            print(f" schema proposal FAILED")
            results.append({"mode": "self-desc", "cluster": cid,
                "persona": "FAILED", "groundedness": 0, "cost": cost1,
                "attempts": 1, "n_fields": 0, "elapsed": 0, "failed": True})

    # Report
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.06 -- SELF-DESCRIBING SCHEMA")
    print("=" * 80)

    print(f"\n{'Mode':<12} {'Cluster':<16} {'Persona':<24} {'G':>5} {'Fields':>6} {'Cost':>8} {'Time':>6}")
    print("-" * 80)
    for r in results:
        if r["failed"]:
            print(f"{r['mode']:<12} {r['cluster']:<16} {'FAILED':<24}")
        else:
            print(f"{r['mode']:<12} {r['cluster']:<16} {r['persona'][:22]:<24} "
                  f"{r['groundedness']:>5.2f} {r['n_fields']:>6} ${r['cost']:>7.4f} {r['elapsed']:>5.1f}s")

    # Comparability
    overlap = schema_field_overlap(proposed_schemas)
    print(f"\nCross-persona comparability (field overlap): {overlap:.2f}")
    print("  1.0 = identical schemas, 0.0 = completely different fields")

    if proposed_schemas:
        print("\nProposed schema fields:")
        for i, s in enumerate(proposed_schemas):
            fields = list(s.get("fields", {}).keys())
            print(f"  Cluster {i+1} ({s.get('schema_name', '?')}): {', '.join(fields[:8])}")

    # Cost comparison
    base_cost = sum(r["cost"] for r in results if r["mode"] == "global")
    self_cost = sum(r["cost"] for r in results if r["mode"] == "self-desc")
    delta = ((self_cost - base_cost) / base_cost * 100) if base_cost else 0
    print(f"\nCost: global=${base_cost:.4f}, self-desc=${self_cost:.4f} ({delta:+.0f}%)")

    out = REPO_ROOT / "output" / "exp_1_06_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
