"""Experiment: Humanization A/B.

Hypothesis: Humanized personas (with backstory, speech patterns, emotional
triggers) produce twin chat replies that read more like real humans than
baseline personas, without sacrificing persona quality scores.

Approach:
  1. Shared ingest + segmentation
  2. Baseline: synthesize v1, score, twin chat, score replies
  3. Humanized: synthesize v2, score with humanness dim, humanized twin chat, score replies
  4. Compare all scores

Usage:
    python scripts/experiment_humanization_ab.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))

import os

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from openai import AsyncOpenAI  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import OpenAIBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize, synthesize_v2  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from twin.chat import OpenAITwinChat, build_persona_system_prompt, build_humanized_system_prompt  # noqa: E402

from evals.judge_helper_1_07 import (  # noqa: E402
    HUMANIZED_DIMENSIONS,
    OpenAIJudgeBackend,
    LLMJudge,
    _HUMANIZED_JUDGE_SYSTEM_PROMPT,
)
from evals.humanness_judge import TwinReplyJudge  # noqa: E402
from evals.ab_comparison import (  # noqa: E402
    StageComparison,
    compare_scores,
    format_comparison_table,
    format_findings_md,
)

# ── Gemma4 config ────────────────────────────────────────────────────
GEMMA4_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://gemma4.maxpetrusenko.com/v1")
GEMMA4_API_KEY = os.getenv("GEMMA4_API_KEY", "sk-gemma4-874e8ba21ef8b61830681077634c54d9fd134fa4236525b0")
GEMMA4_MODEL = "gemma4-32k:latest"  # 32k context for persona synthesis
GEMMA4_JUDGE_MODEL = "gemma4-32k:latest"

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-humanization-ab"

TWIN_QUESTIONS = [
    "What's the single biggest frustration you have with your current tools?",
    "A colleague just recommended a new project management tool. What's your first reaction?",
    "Walk me through a typical Monday morning.",
]


# ── Pipeline helpers ──────────────────────────────────────────────────

def get_clusters() -> list[ClusterData]:
    """Run ingest + segmentation."""
    crawler_records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(c) for c in cluster_dicts]


async def run_twin_chat(
    persona_dict: dict,
    client: AsyncOpenAI,
    questions: list[str],
    system_prompt: str | None = None,
    model: str = GEMMA4_MODEL,
) -> list[dict]:
    """Run twin chat with a persona and return question/reply pairs."""
    twin = OpenAITwinChat(
        persona=persona_dict,
        client=client,
        model=model,
        system_prompt=system_prompt,
    )
    results = []
    for q in questions:
        reply = await twin.reply(q)
        results.append({
            "question": q,
            "reply": reply.text,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "model": reply.model,
        })
    return results


async def score_twin_replies(
    twin_results: list[dict],
    persona_name: str,
    reply_judge: TwinReplyJudge,
) -> list[dict]:
    """Score each twin reply on humanness dimensions."""
    scored = []
    for tr in twin_results:
        score = await reply_judge.score_reply(
            reply=tr["reply"],
            persona_name=persona_name,
            question=tr["question"],
        )
        scored.append({
            **tr,
            "humanness_score": asdict(score),
        })
    return scored


def avg_twin_scores(scored_replies: list[dict]) -> dict[str, float]:
    """Average humanness sub-dimensions across replies."""
    if not scored_replies:
        return {}
    dims = [
        "discourse_markers", "hedging", "specificity",
        "sentence_variety", "emotional_authenticity", "overall",
    ]
    avgs = {}
    for dim in dims:
        vals = [
            r["humanness_score"].get(dim, float("nan"))
            for r in scored_replies
        ]
        valid = [v for v in vals if v == v]  # filter NaN
        avgs[dim] = round(sum(valid) / len(valid), 3) if valid else float("nan")
    return avgs


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT: Humanization A/B")
    print("Hypothesis: Humanized personas produce more human-sounding twin replies")
    print("=" * 72)

    client = AsyncOpenAI(
        base_url=GEMMA4_BASE_URL,
        api_key=GEMMA4_API_KEY,
    )
    backend = OpenAIBackend(client=client, model=GEMMA4_MODEL)

    judge_backend = OpenAIJudgeBackend(client=client, model=GEMMA4_JUDGE_MODEL)
    baseline_judge = LLMJudge(
        backend=judge_backend,
        model=GEMMA4_JUDGE_MODEL,
        calibration="few_shot",
    )
    humanized_judge = LLMJudge(
        backend=judge_backend,
        model=GEMMA4_JUDGE_MODEL,
        calibration="few_shot",
        dimensions=HUMANIZED_DIMENSIONS,
        system_prompt=_HUMANIZED_JUDGE_SYSTEM_PROMPT,
    )
    reply_judge = TwinReplyJudge(
        client=client,
        model=GEMMA4_JUDGE_MODEL,
        backend_type="openai",
    )
    print(f"Using model: {GEMMA4_MODEL} at {GEMMA4_BASE_URL}")

    # Create output directories
    for subdir in ["shared", "baseline", "humanized"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Shared ingest + segmentation ─────────────────────────
    print("\n[1/4] Running ingest + segmentation...")
    t0 = time.monotonic()
    clusters = get_clusters()
    seg_time = time.monotonic() - t0
    print(f"      Got {len(clusters)} clusters ({seg_time:.1f}s)")

    if not clusters:
        print("ERROR: No clusters found")
        sys.exit(1)

    # Save shared data
    clusters_data = [c.model_dump(mode="json") for c in clusters]
    (OUTPUT_DIR / "shared" / "clusters.json").write_text(
        json.dumps(clusters_data, indent=2, default=str)
    )

    # ── Step 2: Baseline run ─────────────────────────────────────────
    print(f"\n[2/4] Baseline run ({len(clusters)} clusters)...")
    baseline_results = []

    for i, cluster in enumerate(clusters):
        print(f"      Cluster {i}: synthesize v1...")
        t0 = time.monotonic()
        result = await synthesize(cluster, backend, groundedness_threshold=0.0, max_retries=4)
        persona_dict = result.persona.model_dump(mode="json")
        synth_time = time.monotonic() - t0
        name = persona_dict.get("name", f"persona_{i:02d}")
        print(f"        -> {name} ({synth_time:.1f}s)")

        # Save persona
        (OUTPUT_DIR / "baseline" / f"persona_{i:02d}.json").write_text(
            json.dumps(persona_dict, indent=2, default=str)
        )

        # Score persona
        persona_score = await baseline_judge.score_persona(persona_dict)
        score_data = {
            "overall": persona_score.overall,
            "dimensions": persona_score.dimensions,
            "rationale": persona_score.rationale,
        }
        (OUTPUT_DIR / "baseline" / f"persona_{i:02d}_scores.json").write_text(
            json.dumps(score_data, indent=2)
        )
        print(f"        Persona score: {persona_score.overall:.1f}")

        # Twin chat (baseline uses standard system prompt)
        baseline_sys = build_persona_system_prompt(persona_dict)
        twin_results = await run_twin_chat(persona_dict, client, TWIN_QUESTIONS, system_prompt=baseline_sys)
        scored_twins = await score_twin_replies(twin_results, name, reply_judge)
        (OUTPUT_DIR / "baseline" / f"persona_{i:02d}_twin.json").write_text(
            json.dumps(scored_twins, indent=2, default=str)
        )
        twin_avgs = avg_twin_scores(scored_twins)
        print(f"        Twin humanness avg: {twin_avgs.get('overall', 'n/a')}")

        baseline_results.append({
            "index": i,
            "name": name,
            "persona_scores": score_data,
            "twin_avgs": twin_avgs,
        })

    # ── Step 3: Humanized run ────────────────────────────────────────
    print(f"\n[3/4] Humanized run ({len(clusters)} clusters)...")
    humanized_results = []

    for i, cluster in enumerate(clusters):
        print(f"      Cluster {i}: synthesize v2...")
        t0 = time.monotonic()
        result = await synthesize_v2(cluster, backend, groundedness_threshold=0.0, max_retries=4)
        persona_dict = result.persona.model_dump(mode="json")
        synth_time = time.monotonic() - t0
        name = persona_dict.get("name", f"persona_{i:02d}")
        print(f"        -> {name} ({synth_time:.1f}s)")

        # Save persona
        (OUTPUT_DIR / "humanized" / f"persona_{i:02d}.json").write_text(
            json.dumps(persona_dict, indent=2, default=str)
        )

        # Score persona (with humanness dimension)
        persona_score = await humanized_judge.score_persona(persona_dict)
        score_data = {
            "overall": persona_score.overall,
            "dimensions": persona_score.dimensions,
            "rationale": persona_score.rationale,
        }
        (OUTPUT_DIR / "humanized" / f"persona_{i:02d}_scores.json").write_text(
            json.dumps(score_data, indent=2)
        )
        print(f"        Persona score: {persona_score.overall:.1f}")

        # Twin chat with humanized system prompt
        h_system = build_humanized_system_prompt(persona_dict)
        twin_results = await run_twin_chat(
            persona_dict, client, TWIN_QUESTIONS, system_prompt=h_system
        )
        scored_twins = await score_twin_replies(twin_results, name, reply_judge)
        (OUTPUT_DIR / "humanized" / f"persona_{i:02d}_twin.json").write_text(
            json.dumps(scored_twins, indent=2, default=str)
        )
        twin_avgs = avg_twin_scores(scored_twins)
        print(f"        Twin humanness avg: {twin_avgs.get('overall', 'n/a')}")

        humanized_results.append({
            "index": i,
            "name": name,
            "persona_scores": score_data,
            "twin_avgs": twin_avgs,
        })

    # ── Step 4: Compare ──────────────────────────────────────────────
    print("\n[4/4] Comparing results...")

    # Persona score comparison
    persona_comparisons = []
    for b, h in zip(baseline_results, humanized_results):
        # Use only overlapping dimensions for comparison
        b_dims = b["persona_scores"]["dimensions"]
        h_dims = h["persona_scores"]["dimensions"]
        shared_dims = {k: v for k, v in b_dims.items() if k in h_dims}
        shared_h = {k: h_dims[k] for k in shared_dims}
        deltas = compare_scores(shared_dims, shared_h)
        persona_comparisons.append(StageComparison(
            stage_name=f"Persona {b['index']}",
            baseline_scores=shared_dims,
            humanized_scores=shared_h,
            deltas=deltas,
        ))

    persona_table = format_comparison_table(persona_comparisons)
    print("\nPersona Score Comparison:")
    print(persona_table)

    # Twin reply comparison
    twin_comparisons = []
    for b, h in zip(baseline_results, humanized_results):
        deltas = compare_scores(b["twin_avgs"], h["twin_avgs"])
        twin_comparisons.append(StageComparison(
            stage_name=f"Twin {b['index']}",
            baseline_scores=b["twin_avgs"],
            humanized_scores=h["twin_avgs"],
            deltas=deltas,
        ))

    twin_table = format_comparison_table(twin_comparisons)
    print("\nTwin Reply Humanness Comparison:")
    print(twin_table)

    # Build per-persona result summaries
    persona_result_details = []
    for b, h in zip(baseline_results, humanized_results):
        persona_result_details.append({
            "name": b["name"],
            "baseline_overall": b["persona_scores"]["overall"],
            "humanized_overall": h["persona_scores"]["overall"],
            "baseline_twin_overall": b["twin_avgs"].get("overall", float("nan")),
            "humanized_twin_overall": h["twin_avgs"].get("overall", float("nan")),
        })

    # Save comparison.json
    comparison_data = {
        "persona_comparisons": [asdict(c) for c in persona_comparisons],
        "twin_comparisons": [asdict(c) for c in twin_comparisons],
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(comparison_data, indent=2, default=str)
    )

    # Save results.json
    results_data = {
        "experiment": "humanization-ab",
        "title": "Humanization A/B",
        "hypothesis": "Humanized personas produce more human-sounding twin replies",
        "model_synthesis": settings.default_model,
        "model_judge": "claude-sonnet-4-20250514",
        "tenant_id": TENANT_ID,
        "num_clusters": len(clusters),
        "twin_questions": TWIN_QUESTIONS,
        "baseline_results": baseline_results,
        "humanized_results": humanized_results,
        "comparison_table": persona_table,
        "twin_comparison_table": twin_table,
        "persona_results": persona_result_details,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    # Save FINDINGS.md
    findings = format_findings_md(results_data)
    (OUTPUT_DIR / "FINDINGS.md").write_text(findings)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
