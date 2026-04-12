"""exp-1.15 — Edge-case behavior fields.

A/B runs synthesis with and without the new `edge_case_behaviors` schema
field, then fires 10 adversarial probes at each persona through TwinChat
and uses Claude-as-judge (Haiku) to score each reply on two dimensions:

  - in_character_score (1-5): did the twin stay in persona under pressure?
  - used_named_reaction (bool): did the twin's reply pattern-match a listed
    edge_case_behavior? (baseline personas have zero listed, so this is
    trivially False for them.)

This gives us two signals:
  1. Does having edge_case_behaviors *at synthesis time* improve in-character
     adherence under provocation? (baseline vs treatment on in_character_score)
  2. Are the named reactions actually being used? (used_named_reaction rate
     for treatment only — baseline is a trivial-true negative control.)

Usage:
    python scripts/run_exp_1_15.py
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
sys.path.insert(0, str(REPO_ROOT / "twin"))

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
from synthesis.models.persona import PersonaV1, PersonaV1WithEdgeCases  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-1.15"

# 10 adversarial probes covering: rudeness, false premises, unsolicited
# advice, moralizing, off-topic pivots. Designed to put the persona's
# voice under strain without being abusive.
ADVERSARIAL_PROBES = [
    "Honestly this is a pretty basic question but I don't think you'll understand it.",
    "I've heard that people in your role are mostly replaceable by automation in 3 years. What do you think?",
    "You should really be using Jira instead of whatever you're using now. It's industry standard.",
    "I assume you're one of those people who jumped on the AI hype train. Am I right?",
    "Wait, did you say you care about 'compliance'? That's just bureaucratic nonsense and you know it.",
    "Your industry is dying and everyone knows it. How do you feel about that?",
    "Let me give you some advice: the tools you mentioned are for beginners. Real pros build their own.",
    "Why are you so focused on 'productivity'? Don't you think life is about more than work?",
    "I think your biggest pain point is actually just that you don't know how to delegate. Change my mind.",
    "Can we talk about something else? This conversation is boring me.",
]


async def synth_both_schemas(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> tuple[list[dict], list[dict]]:
    """Synthesize each cluster under both schemas."""
    baseline: list[dict] = []
    treatment: list[dict] = []
    for i, cluster in enumerate(clusters):
        print(f"\n  [{i + 1}/{len(clusters)}] {cluster.cluster_id}")
        for schema_cls, bucket, label in [
            (PersonaV1, baseline, "baseline"),
            (PersonaV1WithEdgeCases, treatment, "treatment"),
        ]:
            try:
                r = await synthesize(cluster, backend, schema_cls=schema_cls)
                p_dict = r.persona.model_dump(mode="json")
                bucket.append({
                    "cluster_id": cluster.cluster_id,
                    "schema": schema_cls.__name__,
                    "status": "ok",
                    "persona": p_dict,
                    "cost_usd": r.total_cost_usd,
                    "groundedness": r.groundedness.score,
                    "attempts": r.attempts,
                })
                name = p_dict["name"]
                n_edge = len(p_dict.get("edge_case_behaviors", []))
                print(
                    f"      [OK {label:9s}] {name[:40]:40s}  "
                    f"${r.total_cost_usd:.4f}  grounded={r.groundedness.score:.2f}  "
                    f"attempts={r.attempts}  edge_cases={n_edge}"
                )
            except SynthesisError as e:
                bucket.append({
                    "cluster_id": cluster.cluster_id,
                    "schema": schema_cls.__name__,
                    "status": "failed",
                    "error": str(e),
                    "total_cost_usd": sum(a.cost_usd for a in e.attempts),
                })
                print(f"      [FAIL {label}] {e}")
    return baseline, treatment


async def run_adversarial_probes(
    persona_dict: dict,
    client: AsyncAnthropic,
    model: str,
) -> list[dict]:
    """Send each probe to a fresh TwinChat (no history) and capture the reply."""
    twin = TwinChat(persona_dict, client=client, model=model)
    out = []
    for i, probe in enumerate(ADVERSARIAL_PROBES):
        reply = await twin.reply(probe)
        out.append({
            "probe_idx": i,
            "probe": probe,
            "reply": reply.text,
            "cost_usd": reply.estimated_cost_usd,
        })
        print(f"      probe {i + 1:2d}/10  ${reply.estimated_cost_usd:.5f}")
    return out


async def judge_reply(
    persona_dict: dict,
    probe: str,
    reply: str,
    edge_case_behaviors: list[dict],
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """LLM-as-judge: rate the reply 1-5 on in-character adherence.

    Also asks whether the reply matches any of the persona's listed
    edge_case_behaviors (only meaningful for treatment personas).
    """
    edge_section = ""
    if edge_case_behaviors:
        edge_section = "\n\nThis persona has explicitly declared these edge-case reactions:\n"
        for ec in edge_case_behaviors:
            edge_section += (
                f"- trigger: {ec.get('trigger', '')} | "
                f"reaction: {ec.get('reaction', '')} | "
                f"tone_shift: {ec.get('tone_shift', '')}\n"
            )

    judge_system = (
        "You are an impartial judge evaluating whether a persona-driven chat reply "
        "stays in character under adversarial conversational pressure. Be strict. "
        "Do NOT reward generic LLM politeness — reward specificity, vocabulary match, "
        "and reactions that fit the declared persona."
    )
    judge_user = f"""Persona name: {persona_dict.get("name")}
Persona summary: {persona_dict.get("summary")}
Persona vocabulary: {", ".join(persona_dict.get("vocabulary", []))}
Persona sample quotes:
{chr(10).join("- " + q for q in persona_dict.get("sample_quotes", []))}
{edge_section}

Adversarial user message: {probe!r}

Twin reply: {reply!r}

Score the reply on two dimensions and respond with STRICT JSON only:
{{
  "in_character_score": <int 1-5, where 5 = fully in persona, 1 = collapsed to generic LLM>,
  "used_named_reaction": <true if the reply matches one of the listed edge_case reactions above, false otherwise>,
  "rationale": "<one sentence>"
}}"""
    resp = await client.messages.create(
        model=model,
        max_tokens=300,
        system=judge_system,
        messages=[{"role": "user", "content": judge_user}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    # Extract JSON (strip any prose wrapping)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        parsed = {}
    return {
        "raw_judge_text": text,
        "in_character_score": parsed.get("in_character_score"),
        "used_named_reaction": parsed.get("used_named_reaction"),
        "rationale": parsed.get("rationale"),
        "judge_cost_usd": (resp.usage.input_tokens + 5 * resp.usage.output_tokens) / 1_000_000,
    }


async def score_persona(
    entry: dict,
    client: AsyncAnthropic,
    model: str,
) -> dict:
    """Run adversarial probes + judge all replies for one persona."""
    if entry.get("status") != "ok":
        return {**entry, "probes": [], "mean_in_character": None}
    persona = entry["persona"]
    name = persona["name"]
    edge_cases = persona.get("edge_case_behaviors", [])
    print(f"    -- probing {name} ({entry['schema']}) --")
    probes = await run_adversarial_probes(persona, client, model)
    print(f"    -- judging {name} --")
    scored = []
    for p in probes:
        j = await judge_reply(persona, p["probe"], p["reply"], edge_cases, client, model)
        scored.append({**p, **j})
    good = [s["in_character_score"] for s in scored if s["in_character_score"] is not None]
    named = [s["used_named_reaction"] for s in scored if s["used_named_reaction"] is not None]
    return {
        **entry,
        "probes": scored,
        "mean_in_character": sum(good) / len(good) if good else None,
        "used_named_rate": sum(1 for n in named if n) / len(named) if named else 0.0,
        "probe_cost_usd": sum(p["cost_usd"] for p in probes),
        "judge_cost_usd": sum(s["judge_cost_usd"] for s in scored),
    }


async def main() -> None:
    print("=" * 72)
    print("exp-1.15 — Edge-case behavior fields")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    print("\n[1/4] Fetching and clustering mock records...")
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
        key=lambda c: c.cluster_id,
    )
    print(f"  Got {len(clusters)} clusters: {[c.cluster_id for c in clusters]}")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[2/4] Synthesizing under both schemas...")
    baseline, treatment = await synth_both_schemas(clusters, backend)

    print("\n[3/4] Running 10 adversarial probes + LLM-judge per persona...")
    baseline_scored = []
    treatment_scored = []
    for e in baseline:
        baseline_scored.append(await score_persona(e, client, settings.default_model))
    for e in treatment:
        treatment_scored.append(await score_persona(e, client, settings.default_model))

    # ---- Aggregate ----
    def agg(scored: list[dict]) -> dict:
        ok = [e for e in scored if e.get("status") == "ok" and e.get("mean_in_character") is not None]
        if not ok:
            return {"n": 0, "mean_in_character": None, "used_named_rate": None}
        means = [e["mean_in_character"] for e in ok]
        named_rates = [e.get("used_named_rate", 0.0) for e in ok]
        return {
            "n": len(ok),
            "mean_in_character": sum(means) / len(means),
            "used_named_rate": sum(named_rates) / len(named_rates),
            "probe_cost_total": sum(e.get("probe_cost_usd", 0.0) for e in ok),
            "judge_cost_total": sum(e.get("judge_cost_usd", 0.0) for e in ok),
        }

    baseline_agg = agg(baseline_scored)
    treatment_agg = agg(treatment_scored)

    summary = {
        "experiment_id": "1.15",
        "branch": "exp-1.15-edge-case-behavior-fields",
        "model": settings.default_model,
        "n_clusters": len(clusters),
        "n_probes": len(ADVERSARIAL_PROBES),
        "baseline": {"schema": "PersonaV1", **baseline_agg},
        "treatment": {"schema": "PersonaV1WithEdgeCases", **treatment_agg},
    }
    if baseline_agg.get("mean_in_character") is not None and treatment_agg.get("mean_in_character") is not None:
        summary["delta_mean_in_character"] = (
            treatment_agg["mean_in_character"] - baseline_agg["mean_in_character"]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_scored.json").write_text(
        json.dumps(baseline_scored, indent=2, default=str)
    )
    (OUTPUT_DIR / "treatment_scored.json").write_text(
        json.dumps(treatment_scored, indent=2, default=str)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
