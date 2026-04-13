"""Experiment 6.18 — Cross-domain transfer.

Run same cluster through two synthesizer prompt variants (structured vs
narrative) and compare output similarity.

Usage:
    python evals/synthesizer_compare.py
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
from synthesis.engine.groundedness import check_groundedness
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_messages, build_tool_definition
from synthesis.engine.synthesizer import synthesize
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

NARRATIVE_SYSTEM_PROMPT = """\
You are a persona synthesis expert. Your job is to analyze behavioral data from a \
customer cluster and produce a single, richly detailed persona as a NARRATIVE \
CHARACTER SKETCH that tells a story.

Quality criteria:
- **Grounded**: Every claim must trace back to specific source records.
- **Storytelling**: Write goals and pains as mini-narratives. Instead of bullet-point \
facts, paint a picture of a day in this person's life. Make the reader feel what \
this person experiences.
- **Vivid voice**: Sample quotes should be colorful and memorable. Use metaphors, \
humor, or frustration that feels authentic.
- **Consistent**: All fields must describe the same coherent person.

Evidence rules:
- Each entry in source_evidence must reference at least one record_id from the \
provided sample records.
- The field_path must use dot notation (e.g. "goals.0", "pains.2").
- Every item in goals, pains, motivations, and objections MUST have a corresponding \
source_evidence entry.
- Confidence: 1.0 = verbatim, 0.5 = reasonable inference.
"""


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


async def synthesize_with_prompt(cluster, backend, system_prompt):
    """Run synthesis with a custom system prompt (up to 3 attempts)."""
    tool = build_tool_definition()
    messages = build_messages(cluster)
    for attempt in range(3):
        result = await backend.generate(system=system_prompt, messages=messages, tool=tool)
        try:
            persona = PersonaV1.model_validate(result.tool_input)
            g = check_groundedness(persona, cluster)
            return persona, g.score, result.estimated_cost_usd
        except (ValidationError, Exception):
            continue
    return None, 0.0, 0.0


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key"); sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    records = fetch_all("tenant_acme_corp")
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]
    clusters_raw = segment(raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool", existing_persona_names=[],
        similarity_threshold=0.15, min_cluster_size=2)
    clusters = [ClusterData.model_validate(c) for c in clusters_raw]
    print(f"Clusters: {len(clusters)}")

    results = []
    for cluster in clusters:
        print(f"\nCluster: {cluster.cluster_id}")

        # Variant A: default structured
        print("  [structured]...", end="", flush=True)
        p_a, g_a, c_a = await synthesize_with_prompt(cluster, backend, SYSTEM_PROMPT)
        if p_a:
            print(f" {p_a.name} (g={g_a:.2f})")
        else:
            print(" FAILED")

        # Variant B: narrative
        print("  [narrative]...", end="", flush=True)
        p_b, g_b, c_b = await synthesize_with_prompt(cluster, backend, NARRATIVE_SYSTEM_PROMPT)
        if p_b:
            print(f" {p_b.name} (g={g_b:.2f})")
        else:
            print(" FAILED")

        if p_a and p_b:
            vocab_a = set(v.lower() for v in p_a.vocabulary)
            vocab_b = set(v.lower() for v in p_b.vocabulary)
            goal_words_a = set(w.lower() for g in p_a.goals for w in g.split())
            goal_words_b = set(w.lower() for g in p_b.goals for w in g.split())

            results.append({
                "cluster_id": cluster.cluster_id,
                "structured": {"name": p_a.name, "groundedness": g_a, "cost": c_a,
                    "goals": p_a.goals, "vocabulary": p_a.vocabulary[:8]},
                "narrative": {"name": p_b.name, "groundedness": g_b, "cost": c_b,
                    "goals": p_b.goals, "vocabulary": p_b.vocabulary[:8]},
                "vocab_overlap": jaccard(vocab_a, vocab_b),
                "goal_word_overlap": jaccard(goal_words_a, goal_words_b),
            })

    print("\n" + "=" * 80)
    print("EXPERIMENT 6.18 -- CROSS-DOMAIN TRANSFER")
    print("=" * 80)
    print(f"\n{'Cluster':<16} {'Vocab Overlap':>13} {'Goal Overlap':>12} {'Struct G':>8} {'Narr G':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['cluster_id'][:14]:<16} {r['vocab_overlap']:>13.3f} {r['goal_word_overlap']:>12.3f} "
              f"{r['structured']['groundedness']:>8.2f} {r['narrative']['groundedness']:>7.2f}")

    for r in results:
        print(f"\n  {r['cluster_id'][:14]}:")
        print(f"    Structured: {r['structured']['name']}")
        print(f"      Goals: {r['structured']['goals'][:2]}")
        print(f"    Narrative:  {r['narrative']['name']}")
        print(f"      Goals: {r['narrative']['goals'][:2]}")

    out = REPO_ROOT / "output" / "exp_6_18_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
