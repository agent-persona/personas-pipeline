"""Experiment 6.04 — Cross-persona contrast prompting.

Synthesizes personas in two modes:
  - "independent" (baseline): each cluster synthesized in isolation
  - "contrast": persona N+1 sees personas 1..N in the prompt with
    instructions to differ on goals, pains, vocabulary, and voice

Compares distinctiveness (vocabulary overlap, goal overlap) and
groundedness between the two modes.

Usage:
    python evals/contrast_prompting.py
"""

from __future__ import annotations

import asyncio
import json
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
from synthesis.engine.synthesizer import SynthesisError, synthesize
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

TENANT_ID = "tenant_acme_corp"


@dataclass
class PersonaResult:
    mode: str
    cluster_id: str
    persona_name: str
    groundedness_score: float
    attempts: int
    cost_usd: float
    goals: list[str]
    pains: list[str]
    vocabulary: list[str]
    sample_quotes: list[str]
    failed: bool = False
    error: str = ""


def vocabulary_overlap(personas: list[PersonaResult]) -> float:
    """Mean pairwise Jaccard similarity of vocabulary sets across personas."""
    ok = [p for p in personas if not p.failed]
    if len(ok) < 2:
        return 0.0
    similarities = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a = set(w.lower() for w in ok[i].vocabulary)
            b = set(w.lower() for w in ok[j].vocabulary)
            union = a | b
            if not union:
                continue
            similarities.append(len(a & b) / len(union))
    return sum(similarities) / len(similarities) if similarities else 0.0


def goal_overlap(personas: list[PersonaResult]) -> float:
    """Mean pairwise word-level Jaccard similarity of goals."""
    ok = [p for p in personas if not p.failed]
    if len(ok) < 2:
        return 0.0
    similarities = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            words_a = set(w.lower() for g in ok[i].goals for w in g.split())
            words_b = set(w.lower() for g in ok[j].goals for w in g.split())
            union = words_a | words_b
            if not union:
                continue
            similarities.append(len(words_a & words_b) / len(union))
    return sum(similarities) / len(similarities) if similarities else 0.0


def quote_overlap(personas: list[PersonaResult]) -> float:
    """Mean pairwise word-level Jaccard similarity of sample quotes."""
    ok = [p for p in personas if not p.failed]
    if len(ok) < 2:
        return 0.0
    similarities = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            words_a = set(w.lower() for q in ok[i].sample_quotes for w in q.split())
            words_b = set(w.lower() for q in ok[j].sample_quotes for w in q.split())
            union = words_a | words_b
            if not union:
                continue
            similarities.append(len(words_a & words_b) / len(union))
    return sum(similarities) / len(similarities) if similarities else 0.0


async def synthesize_independent(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> list[PersonaResult]:
    """Baseline: synthesize each cluster independently (no contrast)."""
    results = []
    for cluster in clusters:
        print(f"  [independent] {cluster.cluster_id}...", end="", flush=True)
        t0 = time.monotonic()
        try:
            result = await synthesize(cluster, backend)
            elapsed = time.monotonic() - t0
            print(f" {result.persona.name} (g={result.groundedness.score:.2f}, {elapsed:.1f}s)")
            results.append(PersonaResult(
                mode="independent",
                cluster_id=cluster.cluster_id,
                persona_name=result.persona.name,
                groundedness_score=result.groundedness.score,
                attempts=result.attempts,
                cost_usd=result.total_cost_usd,
                goals=result.persona.goals,
                pains=result.persona.pains,
                vocabulary=result.persona.vocabulary,
                sample_quotes=result.persona.sample_quotes,
            ))
        except SynthesisError as e:
            elapsed = time.monotonic() - t0
            print(f" FAILED ({elapsed:.1f}s)")
            results.append(PersonaResult(
                mode="independent",
                cluster_id=cluster.cluster_id,
                persona_name="FAILED",
                groundedness_score=0.0,
                attempts=len(e.attempts),
                cost_usd=sum(a.cost_usd for a in e.attempts),
                goals=[], pains=[], vocabulary=[], sample_quotes=[],
                failed=True, error=str(e),
            ))
    return results


async def synthesize_with_contrast(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> list[PersonaResult]:
    """Experiment: synthesize sequentially, passing prior personas as contrast."""
    results = []
    existing_personas: list[dict] = []

    for cluster in clusters:
        print(
            f"  [contrast] {cluster.cluster_id} "
            f"(seeing {len(existing_personas)} prior persona(s))...",
            end="", flush=True,
        )
        t0 = time.monotonic()
        try:
            result = await synthesize(
                cluster,
                backend,
                existing_personas=existing_personas if existing_personas else None,
            )
            elapsed = time.monotonic() - t0
            print(f" {result.persona.name} (g={result.groundedness.score:.2f}, {elapsed:.1f}s)")

            persona_dict = result.persona.model_dump(mode="json")
            results.append(PersonaResult(
                mode="contrast",
                cluster_id=cluster.cluster_id,
                persona_name=result.persona.name,
                groundedness_score=result.groundedness.score,
                attempts=result.attempts,
                cost_usd=result.total_cost_usd,
                goals=result.persona.goals,
                pains=result.persona.pains,
                vocabulary=result.persona.vocabulary,
                sample_quotes=result.persona.sample_quotes,
            ))

            # Add to existing personas for the next cluster
            existing_personas.append({
                "name": persona_dict.get("name", ""),
                "summary": persona_dict.get("summary", ""),
                "goals": persona_dict.get("goals", []),
                "pains": persona_dict.get("pains", []),
                "vocabulary": persona_dict.get("vocabulary", []),
            })

        except SynthesisError as e:
            elapsed = time.monotonic() - t0
            print(f" FAILED ({elapsed:.1f}s)")
            results.append(PersonaResult(
                mode="contrast",
                cluster_id=cluster.cluster_id,
                persona_name="FAILED",
                groundedness_score=0.0,
                attempts=len(e.attempts),
                cost_usd=sum(a.cost_usd for a in e.attempts),
                goals=[], pains=[], vocabulary=[], sample_quotes=[],
                failed=True, error=str(e),
            ))
    return results


def print_results(
    independent: list[PersonaResult],
    contrast: list[PersonaResult],
) -> None:
    """Print comparison."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 6.04 — CROSS-PERSONA CONTRAST PROMPTING")
    print("=" * 100)

    for mode_label, results in [("INDEPENDENT (baseline)", independent), ("CONTRAST (exp-6.04)", contrast)]:
        print(f"\n--- {mode_label} ---")
        print(f"  {'Cluster':<16} {'Persona':<35} {'Ground':>6} {'Att':>3} {'Cost':>8}")
        print("  " + "-" * 75)
        for r in results:
            if r.failed:
                print(f"  {r.cluster_id[:14]:<16} {'FAILED':<35} {'—':>6} {r.attempts:>3} ${r.cost_usd:>7.4f}")
            else:
                print(
                    f"  {r.cluster_id[:14]:<16} {r.persona_name[:33]:<35} "
                    f"{r.groundedness_score:>6.2f} {r.attempts:>3} ${r.cost_usd:>7.4f}"
                )

    # Distinctiveness metrics
    print("\n" + "=" * 100)
    print("DISTINCTIVENESS COMPARISON")
    print("=" * 100)

    vocab_ind = vocabulary_overlap(independent)
    vocab_con = vocabulary_overlap(contrast)
    goal_ind = goal_overlap(independent)
    goal_con = goal_overlap(contrast)
    quote_ind = quote_overlap(independent)
    quote_con = quote_overlap(contrast)

    print(f"\n  {'Metric':<28} {'Independent':>12} {'Contrast':>12} {'Delta':>8} {'Better?':>8}")
    print("  " + "-" * 72)
    print(f"  {'Vocab overlap (Jaccard)':<28} {vocab_ind:>12.3f} {vocab_con:>12.3f} {vocab_con - vocab_ind:>+8.3f} "
          f"{'YES' if vocab_con < vocab_ind else 'no':>8}")
    print(f"  {'Goal word overlap':<28} {goal_ind:>12.3f} {goal_con:>12.3f} {goal_con - goal_ind:>+8.3f} "
          f"{'YES' if goal_con < goal_ind else 'no':>8}")
    print(f"  {'Quote word overlap':<28} {quote_ind:>12.3f} {quote_con:>12.3f} {quote_con - quote_ind:>+8.3f} "
          f"{'YES' if quote_con < quote_ind else 'no':>8}")

    print(f"\n  (Lower overlap = more distinctive. 'YES' = contrast mode produced more distinct personas)")

    # Groundedness comparison
    print("\n" + "=" * 100)
    print("GROUNDEDNESS COMPARISON")
    print("=" * 100)

    ok_ind = [r for r in independent if not r.failed]
    ok_con = [r for r in contrast if not r.failed]
    avg_g_ind = sum(r.groundedness_score for r in ok_ind) / len(ok_ind) if ok_ind else 0
    avg_g_con = sum(r.groundedness_score for r in ok_con) / len(ok_con) if ok_con else 0
    avg_att_ind = sum(r.attempts for r in ok_ind) / len(ok_ind) if ok_ind else 0
    avg_att_con = sum(r.attempts for r in ok_con) / len(ok_con) if ok_con else 0
    total_cost_ind = sum(r.cost_usd for r in independent)
    total_cost_con = sum(r.cost_usd for r in contrast)

    print(f"\n  {'Metric':<28} {'Independent':>12} {'Contrast':>12} {'Delta':>8}")
    print("  " + "-" * 64)
    print(f"  {'Avg groundedness':<28} {avg_g_ind:>12.2f} {avg_g_con:>12.2f} {avg_g_con - avg_g_ind:>+8.2f}")
    print(f"  {'Avg attempts':<28} {avg_att_ind:>12.1f} {avg_att_con:>12.1f} {avg_att_con - avg_att_ind:>+8.1f}")
    print(f"  {'Total cost':<28} ${total_cost_ind:>11.4f} ${total_cost_con:>11.4f} ${total_cost_con - total_cost_ind:>+7.4f}")
    print(f"  {'Success rate':<28} {len(ok_ind)}/{len(independent):>9} {len(ok_con)}/{len(contrast):>9}")

    # Show vocabulary side-by-side
    print("\n" + "=" * 100)
    print("VOCABULARY COMPARISON")
    print("=" * 100)
    for i, (ind, con) in enumerate(zip(independent, contrast)):
        if ind.failed or con.failed:
            continue
        shared = set(w.lower() for w in ind.vocabulary) & set(w.lower() for w in con.vocabulary)
        only_ind = set(w.lower() for w in ind.vocabulary) - set(w.lower() for w in con.vocabulary)
        only_con = set(w.lower() for w in con.vocabulary) - set(w.lower() for w in ind.vocabulary)
        print(f"\n  Cluster {i + 1} ({ind.cluster_id[:14]}):")
        print(f"    Independent: {', '.join(ind.vocabulary[:10])}")
        print(f"    Contrast:    {', '.join(con.vocabulary[:10])}")
        print(f"    Shared:      {', '.join(sorted(shared)) if shared else '(none)'}")
        print(f"    Only in contrast: {', '.join(sorted(only_con)) if only_con else '(none)'}")
    print()


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    print(f"Model: {settings.default_model}")

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

    # Run both modes
    print("\n--- Synthesizing: INDEPENDENT (baseline) ---")
    independent = await synthesize_independent(clusters, backend)

    print("\n--- Synthesizing: CONTRAST (with prior personas) ---")
    contrast = await synthesize_with_contrast(clusters, backend)

    print_results(independent, contrast)

    # Save results
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "exp_6_04_results.json"
    data = {
        "independent": [
            {
                "cluster_id": r.cluster_id, "persona_name": r.persona_name,
                "groundedness_score": r.groundedness_score, "attempts": r.attempts,
                "cost_usd": r.cost_usd, "goals": r.goals, "pains": r.pains,
                "vocabulary": r.vocabulary, "sample_quotes": r.sample_quotes,
                "failed": r.failed,
            }
            for r in independent
        ],
        "contrast": [
            {
                "cluster_id": r.cluster_id, "persona_name": r.persona_name,
                "groundedness_score": r.groundedness_score, "attempts": r.attempts,
                "cost_usd": r.cost_usd, "goals": r.goals, "pains": r.pains,
                "vocabulary": r.vocabulary, "sample_quotes": r.sample_quotes,
                "failed": r.failed,
            }
            for r in contrast
        ],
        "metrics": {
            "vocab_overlap_independent": vocabulary_overlap(independent),
            "vocab_overlap_contrast": vocabulary_overlap(contrast),
            "goal_overlap_independent": goal_overlap(independent),
            "goal_overlap_contrast": goal_overlap(contrast),
        },
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
