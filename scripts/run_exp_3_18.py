"""exp-3.18 — PII-stripped vs raw.

Measures field-by-field impact of PII stripping on persona quality.

Two conditions:
  - Condition A (raw): synthesize from raw cluster
  - Condition B (stripped): synthesize from PII-stripped cluster

Claude-as-judge rates each field group (demographics, firmographics, goals,
pains, motivations, objections) on 1-5 grounding scale.

Hypothesis: PII stripping degrades demographics by ≥0.5 but leaves
behavioral fields within ≤0.1.

Usage:
    python scripts/run_exp_3_18.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-3.18"

FIELD_GROUPS = ["demographics", "firmographics", "goals", "pains", "motivations", "objections"]


def strip_pii(text: str) -> str:
    """Remove common PII patterns from text."""
    # Email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    # IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    # US phone numbers
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    # ZIP codes (US 5-digit)
    text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '[ZIP]', text)
    # Names (simple heuristic: capitalized word pairs)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    # Ages
    text = re.sub(r'\bage\s+\d+\b', '[AGE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+years?\s+old\b', '[AGE]', text, flags=re.IGNORECASE)
    return text


def strip_cluster_pii(cluster: ClusterData) -> ClusterData:
    """Deep copy a cluster with all string payload values PII-stripped."""
    stripped = cluster.model_copy(deep=True)
    for rec in stripped.sample_records:
        new_payload = {}
        for k, v in rec.payload.items():
            if isinstance(v, str):
                new_payload[k] = strip_pii(v)
            else:
                new_payload[k] = v
        rec.payload = new_payload
    stripped.cluster_id = f"{cluster.cluster_id}_pii_stripped"
    return stripped


async def judge_field_groups(
    persona_dict: dict,
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """Claude-as-judge: rate each field group on grounding quality."""
    record_summary = []
    for rec in cluster.sample_records[:10]:
        record_summary.append(f"- {rec.record_id}: {json.dumps(rec.payload)}")

    field_groups_text = ""
    for group in FIELD_GROUPS:
        val = persona_dict.get(group, {})
        field_groups_text += f"\n**{group}**: {json.dumps(val)}"

    judge_prompt = f"""Rate each field group of this persona on grounding quality (1-5 scale):
5 = strongly supported by source data, 1 = no evidence in records

Persona name: {persona_dict.get('name')}
{field_groups_text}

Source records:
{chr(10).join(record_summary)}

Rate EACH of these groups: demographics, firmographics, goals, pains, motivations, objections

Respond with STRICT JSON only:
{{"demographics": <int 1-5>, "firmographics": <int 1-5>, "goals": <int 1-5>, "pains": <int 1-5>, "motivations": <int 1-5>, "objections": <int 1-5>, "rationale": "<1-2 sentences>"}}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        parsed = {}
    return {group: parsed.get(group) for group in FIELD_GROUPS}


async def main() -> None:
    print("=" * 72)
    print("exp-3.18 — PII-stripped vs raw")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    # ---- Ingest + segment ----
    print("\n[1/3] Fetching and clustering mock records...")
    raw_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = sorted(
        [ClusterData.model_validate(c) for c in cluster_dicts],
        key=lambda c: len(c.sample_records),
        reverse=True,
    )
    print(f"  Got {len(clusters)} clusters, using largest: {clusters[0].cluster_id}")

    raw_cluster = clusters[0]
    stripped_cluster = strip_cluster_pii(raw_cluster)

    print(f"  Raw cluster: {raw_cluster.cluster_id} ({len(raw_cluster.sample_records)} records)")
    print(f"  Stripped cluster: {stripped_cluster.cluster_id}")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # ---- Condition A: raw ----
    print("\n[2/3] Condition A — synthesize from raw cluster...")
    raw_persona_dict = None
    raw_result_data = {}
    try:
        r = await synthesize(raw_cluster, backend)
        raw_persona_dict = r.persona.model_dump(mode="json")
        print(f"  [OK] {raw_persona_dict['name'][:40]}  cost=${r.total_cost_usd:.4f}")
        raw_result_data = {
            "status": "ok",
            "persona": raw_persona_dict,
            "cost_usd": r.total_cost_usd,
            "groundedness": r.groundedness.score,
            "attempts": r.attempts,
        }
    except SynthesisError as e:
        print(f"  [FAIL] {e}")
        raw_result_data = {
            "status": "failed",
            "error": str(e),
            "cost_usd": sum(a.cost_usd for a in e.attempts),
        }

    # ---- Condition B: stripped ----
    print("\n       Condition B — synthesize from PII-stripped cluster...")
    stripped_persona_dict = None
    stripped_result_data = {}
    try:
        r = await synthesize(stripped_cluster, backend)
        stripped_persona_dict = r.persona.model_dump(mode="json")
        print(f"  [OK] {stripped_persona_dict['name'][:40]}  cost=${r.total_cost_usd:.4f}")
        stripped_result_data = {
            "status": "ok",
            "persona": stripped_persona_dict,
            "cost_usd": r.total_cost_usd,
            "groundedness": r.groundedness.score,
            "attempts": r.attempts,
        }
    except SynthesisError as e:
        print(f"  [FAIL] {e}")
        stripped_result_data = {
            "status": "failed",
            "error": str(e),
            "cost_usd": sum(a.cost_usd for a in e.attempts),
        }

    # ---- Judge field groups ----
    print("\n[3/3] Judging field groups...")
    raw_scores = {}
    stripped_scores = {}

    if raw_persona_dict:
        print("  Judging raw persona...")
        raw_scores = await judge_field_groups(raw_persona_dict, raw_cluster, client, settings.default_model)
        print(f"  Raw scores: {raw_scores}")

    if stripped_persona_dict:
        print("  Judging stripped persona...")
        stripped_scores = await judge_field_groups(stripped_persona_dict, stripped_cluster, client, settings.default_model)
        print(f"  Stripped scores: {stripped_scores}")

    # ---- Compute deltas ----
    deltas = {}
    for group in FIELD_GROUPS:
        r_val = raw_scores.get(group)
        s_val = stripped_scores.get(group)
        if r_val is not None and s_val is not None:
            deltas[group] = r_val - s_val
        else:
            deltas[group] = None

    demographics_degradation = deltas.get("demographics")
    behavioral_deltas = [abs(deltas[g]) for g in ["goals", "pains", "motivations", "objections"] if deltas.get(g) is not None]
    max_behavioral_delta = max(behavioral_deltas) if behavioral_deltas else None

    summary = {
        "experiment_id": "3.18",
        "branch": "exp-3.18-pii-stripped",
        "model": settings.default_model,
        "raw_scores": raw_scores,
        "stripped_scores": stripped_scores,
        "deltas": deltas,
        "hypothesis_check": {
            "demographics_degradation": demographics_degradation,
            "max_behavioral_delta": max_behavioral_delta,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "raw_personas.json").write_text(
        json.dumps(raw_result_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "stripped_personas.json").write_text(
        json.dumps(stripped_result_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
