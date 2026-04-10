"""Experiment 1.20 — Behavioral verbs vs adjectives.

Runs synthesis on each cluster in three prompt modes (default, verbs,
adjectives) and compares the persona output for agentic content and
stereotype rate.

Usage:
    python evals/verbs_vs_adjectives.py
"""

from __future__ import annotations

import asyncio
import json
import re
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
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.config import Settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.prompt_builder import SynthesisStyle, build_system_prompt
from synthesis.engine.synthesizer import SynthesisError, synthesize
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

TENANT_ID = "tenant_acme_corp"

# Common stereotype adjectives that are generic and could describe anyone
STEREOTYPE_TERMS = [
    "tech-savvy", "detail-oriented", "results-driven", "data-driven",
    "innovative", "passionate", "forward-thinking", "proactive",
    "collaborative", "strategic", "dynamic", "motivated",
    "budget-conscious", "quality-focused", "customer-centric",
    "growth-minded", "efficiency-focused", "self-starter",
    "team player", "thought leader",
]

# Verb indicators — active verbs that signal behavioral/agentic content
ACTION_VERB_PATTERNS = [
    r"\b(automates?|consolidates?|reduces?|monitors?|builds?|deploys?)\b",
    r"\b(reviews?|configures?|integrates?|exports?|imports?|tests?)\b",
    r"\b(schedules?|tracks?|measures?|analyzes?|debugs?|scripts?)\b",
    r"\b(migrates?|refactors?|customizes?|optimizes?|validates?)\b",
    r"\b(switches?|compares?|evaluates?|documents?|maintains?)\b",
    r"\b(runs?|checks?|sets up|logs?|pushes?|pulls?|merges?)\b",
    r"\b(spends?\s+\d+|waits?\s+\d+|takes?\s+\d+)\b",  # time-specific actions
]

# Adjective/identity indicators
TRAIT_PATTERNS = [
    r"\b(is\s+(?:a\s+)?(?:\w+-)?(?:oriented|driven|focused|conscious|minded))\b",
    r"\b(is\s+(?:a\s+)?(?:analytical|collaborative|creative|pragmatic|methodical))\b",
    r"\b(is\s+(?:a\s+)?(?:cautious|adventurous|risk-averse|skeptical|trusting))\b",
    r"\b(tends\s+to\s+be|naturally|inherently|fundamentally)\b",
    r"\b(personality|disposition|temperament|mindset|attitude)\b",
]


@dataclass
class PersonaAnalysis:
    style: str
    cluster_id: str
    persona_name: str
    groundedness_score: float
    attempts: int
    cost_usd: float
    # Content analysis
    goals: list[str]
    pains: list[str]
    sample_quotes: list[str]
    # Scores
    verb_count: int = 0
    trait_count: int = 0
    stereotype_count: int = 0
    agentic_ratio: float = 0.0  # verbs / (verbs + traits)
    failed: bool = False
    error: str = ""


def count_patterns(texts: list[str], patterns: list[str]) -> int:
    """Count total regex matches across all texts."""
    total = 0
    combined = " ".join(texts).lower()
    for pattern in patterns:
        total += len(re.findall(pattern, combined, re.IGNORECASE))
    return total


def count_stereotypes(texts: list[str]) -> int:
    """Count stereotype terms in combined text."""
    combined = " ".join(texts).lower()
    total = 0
    for term in STEREOTYPE_TERMS:
        total += combined.count(term.lower())
    return total


def analyze_persona(
    persona: PersonaV1,
    style: str,
    cluster_id: str,
    groundedness_score: float,
    attempts: int,
    cost: float,
) -> PersonaAnalysis:
    """Analyze a synthesized persona for verb/adjective content."""
    all_text = persona.goals + persona.pains + persona.motivations + persona.objections + persona.sample_quotes

    verb_count = count_patterns(all_text, ACTION_VERB_PATTERNS)
    trait_count = count_patterns(all_text, TRAIT_PATTERNS)
    stereotype_count = count_stereotypes(all_text)
    total = verb_count + trait_count
    agentic_ratio = verb_count / total if total > 0 else 0.5

    return PersonaAnalysis(
        style=style,
        cluster_id=cluster_id,
        persona_name=persona.name,
        groundedness_score=groundedness_score,
        attempts=attempts,
        cost_usd=cost,
        goals=persona.goals,
        pains=persona.pains,
        sample_quotes=persona.sample_quotes,
        verb_count=verb_count,
        trait_count=trait_count,
        stereotype_count=stereotype_count,
        agentic_ratio=agentic_ratio,
    )


async def run_synthesis(
    cluster: ClusterData,
    backend: AnthropicBackend,
    style: SynthesisStyle,
) -> PersonaAnalysis:
    """Run synthesis with the given style and return analysis."""
    try:
        result = await synthesize(cluster, backend, style=style)
        return analyze_persona(
            result.persona,
            style=style,
            cluster_id=cluster.cluster_id,
            groundedness_score=result.groundedness.score,
            attempts=result.attempts,
            cost=result.total_cost_usd,
        )
    except SynthesisError as e:
        return PersonaAnalysis(
            style=style,
            cluster_id=cluster.cluster_id,
            persona_name="FAILED",
            groundedness_score=0.0,
            attempts=len(e.attempts),
            cost_usd=sum(a.cost_usd for a in e.attempts),
            goals=[], pains=[], sample_quotes=[],
            failed=True,
            error=str(e),
        )


def print_results(results: list[PersonaAnalysis]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 1.20 — BEHAVIORAL VERBS vs ADJECTIVES")
    print("=" * 100)

    styles = ["default", "verbs", "adjectives"]

    for cluster_id in sorted(set(r.cluster_id for r in results)):
        cluster_results = [r for r in results if r.cluster_id == cluster_id]
        print(f"\n--- Cluster: {cluster_id} ---")
        print(
            f"  {'Style':<12} {'Persona':<28} {'Ground':>6} {'Att':>3} "
            f"{'Verbs':>5} {'Traits':>6} {'Stereo':>6} {'AgentRatio':>10} {'Cost':>8}"
        )
        print("  " + "-" * 92)

        for style in styles:
            r = next((x for x in cluster_results if x.style == style), None)
            if not r:
                continue
            if r.failed:
                print(f"  {r.style:<12} {'FAILED':<28} {'—':>6} {r.attempts:>3} "
                      f"{'—':>5} {'—':>6} {'—':>6} {'—':>10} ${r.cost_usd:>7.4f}")
            else:
                print(
                    f"  {r.style:<12} {r.persona_name:<28} {r.groundedness_score:>6.2f} {r.attempts:>3} "
                    f"{r.verb_count:>5} {r.trait_count:>6} {r.stereotype_count:>6} "
                    f"{r.agentic_ratio:>10.2f} ${r.cost_usd:>7.4f}"
                )

    # Aggregate by style
    print("\n" + "=" * 100)
    print("AGGREGATE BY STYLE")
    print("=" * 100)
    print(
        f"  {'Style':<12} {'Success':>7} {'Avg Ground':>10} {'Avg Verbs':>9} "
        f"{'Avg Traits':>10} {'Avg Stereo':>10} {'Avg Agent%':>10} {'Total Cost':>10}"
    )
    print("  " + "-" * 80)

    for style in styles:
        style_results = [r for r in results if r.style == style]
        ok = [r for r in style_results if not r.failed]
        n_total = len(style_results)
        n_ok = len(ok)

        if not ok:
            print(f"  {style:<12} {n_ok}/{n_total:>4} {'—':>10} {'—':>9} {'—':>10} {'—':>10} {'—':>10} "
                  f"${sum(r.cost_usd for r in style_results):>9.4f}")
            continue

        avg_g = sum(r.groundedness_score for r in ok) / len(ok)
        avg_v = sum(r.verb_count for r in ok) / len(ok)
        avg_t = sum(r.trait_count for r in ok) / len(ok)
        avg_s = sum(r.stereotype_count for r in ok) / len(ok)
        avg_a = sum(r.agentic_ratio for r in ok) / len(ok)
        total_c = sum(r.cost_usd for r in style_results)

        print(
            f"  {style:<12} {n_ok}/{n_total:>4} {avg_g:>10.2f} {avg_v:>9.1f} "
            f"{avg_t:>10.1f} {avg_s:>10.1f} {avg_a:>10.2f} ${total_c:>9.4f}"
        )

    print()
    print("KEY:")
    print("  Verbs    = action verb matches in goals/pains/quotes (higher = more behavioral)")
    print("  Traits   = identity/adjective matches (higher = more trait-focused)")
    print("  Stereo   = generic stereotype terms (lower = better)")
    print("  Agent%   = verbs / (verbs + traits) — 1.0 = fully agentic, 0.0 = fully trait-based")
    print()

    # Show sample goals side by side
    print("=" * 100)
    print("SAMPLE GOALS COMPARISON (first cluster)")
    print("=" * 100)
    first_cluster = sorted(set(r.cluster_id for r in results))[0]
    for style in styles:
        r = next((x for x in results if x.cluster_id == first_cluster and x.style == style and not x.failed), None)
        if r:
            print(f"\n  [{r.style.upper()}] {r.persona_name}:")
            for i, g in enumerate(r.goals):
                print(f"    goal {i}: {g}")
            for i, p in enumerate(r.pains[:2]):
                print(f"    pain {i}: {p}")
            for i, q in enumerate(r.sample_quotes[:2]):
                print(f"    quote {i}: \"{q}\"")
    print()


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    print(f"Model: {settings.default_model}")
    print(f"Styles: default, verbs, adjectives")

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
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Run all 3 styles x all clusters
    styles: list[SynthesisStyle] = ["default", "verbs", "adjectives"]
    results: list[PersonaAnalysis] = []

    for cluster in clusters:
        for style in styles:
            print(f"\n  Synthesizing {cluster.cluster_id} [{style}]...", end="", flush=True)
            t0 = time.monotonic()
            analysis = await run_synthesis(cluster, backend, style)
            elapsed = time.monotonic() - t0
            if analysis.failed:
                print(f" FAILED ({elapsed:.1f}s)")
            else:
                print(f" {analysis.persona_name} (g={analysis.groundedness_score:.2f}, {elapsed:.1f}s)")
            results.append(analysis)

    print_results(results)

    # Save raw results
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "exp_1_20_results.json"
    data = []
    for r in results:
        data.append({
            "style": r.style,
            "cluster_id": r.cluster_id,
            "persona_name": r.persona_name,
            "groundedness_score": r.groundedness_score,
            "attempts": r.attempts,
            "cost_usd": r.cost_usd,
            "goals": r.goals,
            "pains": r.pains,
            "sample_quotes": r.sample_quotes,
            "verb_count": r.verb_count,
            "trait_count": r.trait_count,
            "stereotype_count": r.stereotype_count,
            "agentic_ratio": r.agentic_ratio,
            "failed": r.failed,
            "error": r.error,
        })
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
