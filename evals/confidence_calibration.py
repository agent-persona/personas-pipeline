"""Experiment 5.14 -- Confidence calibration.

Synthesize personas, judge them with confidence scores, and compute
Brier calibration to measure judge reliability.

Brier score = mean((confidence - accuracy)^2)
where accuracy = 1 if score >= 4 else 0  (on the 1-5 scale)

A perfectly calibrated judge has Brier score 0.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# -- path setup ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler.pipeline import fetch_all  # noqa: E402
from evaluation.judges import LLMJudge  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("exp-5.14")

TENANT_ID = "tenant_acme_corp"
NUM_PERSONAS = 2
JUDGE_MODEL = os.getenv("default_model", "claude-haiku-4-5-20251001")
SYNTH_MODEL = os.getenv("default_model", "claude-haiku-4-5-20251001")


def _crawler_to_segmentation(records: list) -> list[RawRecord]:
    """Convert crawler Records to segmentation RawRecords."""
    return [
        RawRecord(
            record_id=r.record_id,
            tenant_id=r.tenant_id,
            source=r.source,
            timestamp=r.timestamp,
            user_id=r.user_id,
            behaviors=r.behaviors,
            pages=r.pages,
            payload=r.payload,
            schema_version=r.schema_version,
        )
        for r in records
    ]


def compute_brier_score(
    scores_1to5: list[float],
    confidences: list[float],
    threshold: int = 4,
) -> float:
    """Compute Brier score: mean((confidence - accuracy)^2).

    accuracy = 1.0 if score >= threshold else 0.0
    """
    if not scores_1to5:
        return float("nan")
    total = 0.0
    for score, conf in zip(scores_1to5, confidences):
        accuracy = 1.0 if score >= threshold else 0.0
        total += (conf - accuracy) ** 2
    return total / len(scores_1to5)


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Check synthesis/.env")
        sys.exit(1)

    client = AsyncAnthropic(api_key=api_key)

    # -- Step 1: Crawl ----------------------------------------------------------
    logger.info("Crawling records for %s", TENANT_ID)
    crawler_records = fetch_all(TENANT_ID)
    logger.info("Fetched %d records", len(crawler_records))

    # -- Step 2: Segment ---------------------------------------------------------
    logger.info("Segmenting records")
    raw_records = _crawler_to_segmentation(crawler_records)
    cluster_dicts = segment(raw_records)
    logger.info("Found %d clusters", len(cluster_dicts))

    if len(cluster_dicts) < NUM_PERSONAS:
        logger.warning(
            "Only %d clusters found, need %d personas",
            len(cluster_dicts),
            NUM_PERSONAS,
        )

    # -- Step 3: Synthesize personas --------------------------------------------
    backend = AnthropicBackend(client=client, model=SYNTH_MODEL)
    personas: list[dict] = []
    for i, cd in enumerate(cluster_dicts[:NUM_PERSONAS]):
        cluster = ClusterData.model_validate(cd)
        logger.info("Synthesizing persona %d/%d from cluster %s", i + 1, NUM_PERSONAS, cluster.cluster_id)
        result = await synthesize(cluster, backend)
        persona_dict = result.persona.model_dump()
        personas.append(persona_dict)
        logger.info("  -> %s (cost=$%.4f)", result.persona.name, result.total_cost_usd)

    # -- Step 4: Judge with confidence ------------------------------------------
    judge = LLMJudge(model=JUDGE_MODEL, client=client)
    all_scores: list[dict] = []

    for i, persona in enumerate(personas):
        logger.info("Judging persona %d/%d: %s", i + 1, len(personas), persona.get("name", "?"))
        judge_result = await judge.score_persona(persona)
        all_scores.append({
            "persona_name": persona.get("name", f"persona_{i}"),
            "dimensions": judge_result.dimensions,
            "confidences": judge_result.confidences,
            "overall": judge_result.overall,
            "rationale": judge_result.rationale,
        })

    # -- Step 5: Compute Brier calibration --------------------------------------
    dimensions = LLMJudge.DEFAULT_DIMENSIONS
    per_dim_scores_raw: dict[str, list[float]] = {d: [] for d in dimensions}
    per_dim_confidences: dict[str, list[float]] = {d: [] for d in dimensions}

    for entry in all_scores:
        for dim in dimensions:
            # dimensions are stored as 0-1 in JudgeScore; convert back to 1-5
            normed = entry["dimensions"].get(dim, float("nan"))
            raw_score = normed * 5.0
            per_dim_scores_raw[dim].append(raw_score)
            per_dim_confidences[dim].append(entry["confidences"].get(dim, 0.5))

    # Flatten for overall Brier
    all_raw_scores = []
    all_confidences = []
    for dim in dimensions:
        all_raw_scores.extend(per_dim_scores_raw[dim])
        all_confidences.extend(per_dim_confidences[dim])

    overall_brier = compute_brier_score(all_raw_scores, all_confidences)

    # -- Report -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 5.14 -- Confidence Calibration Report")
    print("=" * 70)

    for entry in all_scores:
        print(f"\nPersona: {entry['persona_name']}")
        print(f"  Overall score (0-1): {entry['overall']:.3f}")
        print(f"  Rationale: {entry['rationale'][:120]}...")
        print(f"  {'Dimension':<18} {'Score (1-5)':>11} {'Confidence':>11}")
        print(f"  {'-'*18} {'-'*11} {'-'*11}")
        for dim in dimensions:
            raw = entry["dimensions"].get(dim, float("nan")) * 5.0
            conf = entry["confidences"].get(dim, float("nan"))
            print(f"  {dim:<18} {raw:>11.1f} {conf:>11.2f}")

    print(f"\n{'='*70}")
    print("Per-dimension Brier scores (lower = better calibrated):")
    print(f"  {'Dimension':<18} {'Brier':>8} {'Avg Score':>10} {'Avg Conf':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*10} {'-'*10}")
    for dim in dimensions:
        brier = compute_brier_score(per_dim_scores_raw[dim], per_dim_confidences[dim])
        avg_score = sum(per_dim_scores_raw[dim]) / len(per_dim_scores_raw[dim])
        avg_conf = sum(per_dim_confidences[dim]) / len(per_dim_confidences[dim])
        print(f"  {dim:<18} {brier:>8.4f} {avg_score:>10.2f} {avg_conf:>10.2f}")

    print(f"\nOverall Brier score: {overall_brier:.4f}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Synthesis model: {SYNTH_MODEL}")
    print(f"Personas evaluated: {len(personas)}")
    print("=" * 70)

    # Save raw results
    output_path = REPO_ROOT / "evals" / "confidence_calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "experiment": "5.14",
                "judge_model": JUDGE_MODEL,
                "synth_model": SYNTH_MODEL,
                "overall_brier_score": overall_brier,
                "persona_scores": all_scores,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
