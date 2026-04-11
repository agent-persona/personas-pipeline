"""Experiment 6.07 — Counterfactual probes.

Per persona, craft questions whose source-data answer is unambiguous.
Query the twin and measure factual accuracy.

Usage:
    python evals/counterfactual_probes.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
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
from twin import TwinChat


@dataclass
class Probe:
    question: str
    field: str  # persona field this tests
    expected_keywords: list[str]  # any match = pass


def build_probes(persona: dict) -> list[Probe]:
    """Generate probes from known persona attributes."""
    probes = []
    firmo = persona.get("firmographics", {})
    if firmo.get("industry"):
        probes.append(Probe(
            question="What industry do you work in?",
            field="firmographics.industry",
            expected_keywords=[firmo["industry"].lower()],
        ))
    if firmo.get("company_size"):
        probes.append(Probe(
            question="How big is your company?",
            field="firmographics.company_size",
            expected_keywords=[firmo["company_size"].lower()],
        ))
    goals = persona.get("goals", [])
    if goals:
        # Extract key words from first goal
        words = [w.lower() for w in goals[0].split() if len(w) > 4][:3]
        probes.append(Probe(
            question="What are you trying to achieve right now?",
            field="goals",
            expected_keywords=words,
        ))
    pains = persona.get("pains", [])
    if pains:
        words = [w.lower() for w in pains[0].split() if len(w) > 4][:3]
        probes.append(Probe(
            question="What's your biggest frustration at work?",
            field="pains",
            expected_keywords=words,
        ))
    vocab = persona.get("vocabulary", [])
    if vocab:
        probes.append(Probe(
            question="Tell me about the tools and concepts you work with daily.",
            field="vocabulary",
            expected_keywords=[v.lower() for v in vocab[:5]],
        ))
    roles = firmo.get("role_titles", [])
    if roles:
        probes.append(Probe(
            question="What's your job title?",
            field="firmographics.role_titles",
            expected_keywords=[r.lower() for r in roles],
        ))
    return probes


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

    all_results = []
    total_probes = 0
    total_hits = 0

    for cluster in clusters:
        print(f"\nSynthesizing {cluster.cluster_id}...")
        result = await synthesize(cluster, backend)
        persona_dict = result.persona.model_dump(mode="json")
        print(f"  {result.persona.name}")

        twin = TwinChat(persona_dict, client=client, model=settings.default_model)
        probes = build_probes(persona_dict)
        print(f"  Probing with {len(probes)} questions...")

        persona_results = {"persona": result.persona.name, "probes": []}
        hits = 0
        for probe in probes:
            reply = await twin.reply(probe.question)
            text_lower = reply.text.lower()
            hit = any(kw in text_lower for kw in probe.expected_keywords)
            hits += 1 if hit else 0
            total_probes += 1
            total_hits += 1 if hit else 0
            status = "HIT" if hit else "MISS"
            print(f"    [{status}] {probe.field}: {probe.question}")
            persona_results["probes"].append({
                "question": probe.question, "field": probe.field,
                "expected": probe.expected_keywords, "hit": hit,
                "response_snippet": reply.text[:150],
            })
        accuracy = hits / len(probes) if probes else 0
        persona_results["accuracy"] = accuracy
        print(f"  Accuracy: {hits}/{len(probes)} ({accuracy:.0%})")
        all_results.append(persona_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT 6.07 -- COUNTERFACTUAL PROBES")
    print("=" * 80)
    pop_accuracy = total_hits / total_probes if total_probes else 0
    print(f"\nPopulation-level accuracy: {total_hits}/{total_probes} ({pop_accuracy:.0%})")
    for r in all_results:
        print(f"  {r['persona']}: {r['accuracy']:.0%}")

    out = REPO_ROOT / "output" / "exp_6_07_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
