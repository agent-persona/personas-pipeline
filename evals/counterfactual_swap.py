"""Experiment 3.15 -- Counterfactual swap test.

Swaps cluster data between two tenant contexts at synthesis time.
A grounded pipeline should follow the DATA (produce a persona matching the
swapped cluster), while a prompt-anchored pipeline would follow the tenant
context in the prompt.

Method
------
1. Ingest records and segment for tenant_acme_corp (engineers + designers).
2. Synthesize each cluster normally (control).
3. Swap: keep the engineer cluster's records but change tenant context to a
   different industry ("Healthcare" / "Hospital management software").
4. Compare: does the persona follow the data (still looks like an engineer)
   or the prompt (looks like healthcare)?
5. Measure: data-following rate by checking if the persona's goals/vocabulary
   match the original data vs the swapped tenant context.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (standard eval pattern)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic

from crawler.pipeline import fetch_all
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment
from synthesis.models.cluster import ClusterData
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import SynthesisResult, synthesize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TENANT_ID = "tenant_acme_corp"
ORIGINAL_INDUSTRY = "B2B SaaS"
ORIGINAL_PRODUCT = "Project management and developer tools platform"

SWAPPED_INDUSTRY = "Healthcare"
SWAPPED_PRODUCT = "Hospital management software"

MODEL = "claude-haiku-4-5-20251001"

# Keywords that signal the persona is following the *data* (engineering cluster)
ENGINEERING_KEYWORDS = [
    "api", "webhook", "integration", "github", "terraform", "developer",
    "engineer", "sdk", "pipeline", "deployment", "ci/cd", "infrastructure",
    "code", "devops", "automation", "technical", "dashboard", "slack",
]

# Keywords that signal the persona is following the *swapped prompt* (healthcare)
HEALTHCARE_KEYWORDS = [
    "patient", "hospital", "clinical", "nurse", "doctor", "physician",
    "ehr", "medical", "healthcare", "hipaa", "diagnosis", "treatment",
    "pharmacy", "care", "health", "compliance", "EMR",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("exp_3.15")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_count(text: str, keywords: list[str]) -> int:
    """Count how many distinct keywords appear in text (case-insensitive)."""
    lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def _persona_text(result: SynthesisResult) -> str:
    """Flatten a persona into a single string for keyword analysis."""
    p = result.persona
    parts = [
        p.name,
        p.summary,
        " ".join(p.goals),
        " ".join(p.pains),
        " ".join(p.motivations),
        " ".join(p.objections),
        " ".join(p.vocabulary),
        " ".join(p.sample_quotes),
        " ".join(p.channels),
    ]
    if p.firmographics.industry:
        parts.append(p.firmographics.industry)
    parts.extend(p.firmographics.role_titles)
    parts.extend(p.firmographics.tech_stack_signals)
    for stage in p.journey_stages:
        parts.append(stage.mindset)
        parts.extend(stage.key_actions)
        parts.extend(stage.content_preferences)
    return " ".join(parts)


def _pick_engineer_cluster(clusters: list[dict]) -> dict | None:
    """Heuristically pick the engineer cluster from segment output."""
    for c in clusters:
        behaviors = c.get("summary", {}).get("top_behaviors", [])
        behavior_str = " ".join(behaviors).lower()
        if "api" in behavior_str or "webhook" in behavior_str or "github" in behavior_str:
            return c
    return clusters[0] if clusters else None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def run_experiment() -> dict:
    """Execute the counterfactual swap experiment and return results."""

    # ----- Step 1: Ingest + segment -----
    logger.info("Step 1: Fetching records for %s", TENANT_ID)
    raw_records = fetch_all(TENANT_ID)
    logger.info("Fetched %d records", len(raw_records))

    seg_records = [RawRecord(**r.model_dump()) for r in raw_records]
    clusters_raw = segment(
        seg_records,
        tenant_industry=ORIGINAL_INDUSTRY,
        tenant_product=ORIGINAL_PRODUCT,
    )
    logger.info("Segmentation produced %d clusters", len(clusters_raw))

    if not clusters_raw:
        raise RuntimeError("Segmentation produced zero clusters -- cannot proceed")

    # Find the engineer cluster
    engineer_raw = _pick_engineer_cluster(clusters_raw)
    if engineer_raw is None:
        raise RuntimeError("Could not identify engineer cluster")

    engineer_cluster = ClusterData.model_validate(engineer_raw)
    logger.info(
        "Selected engineer cluster %s with %d sample records; top behaviors: %s",
        engineer_cluster.cluster_id,
        len(engineer_cluster.sample_records),
        engineer_cluster.summary.top_behaviors[:5],
    )

    # ----- Step 2: Control synthesis (original tenant context) -----
    client = AsyncAnthropic()
    backend = AnthropicBackend(client=client, model=MODEL)

    logger.info("Step 2: Control synthesis (original tenant context)")
    control_result = await synthesize(engineer_cluster, backend)
    logger.info("Control persona: %s", control_result.persona.name)

    # ----- Step 3: Swapped synthesis (healthcare tenant context) -----
    swapped_cluster = engineer_cluster.model_copy(deep=True)
    swapped_cluster.tenant.industry = SWAPPED_INDUSTRY
    swapped_cluster.tenant.product_description = SWAPPED_PRODUCT
    swapped_cluster.tenant.tenant_id = "tenant_healthcare_demo"

    logger.info("Step 3: Swapped synthesis (healthcare tenant context)")
    swapped_result = await synthesize(swapped_cluster, backend)
    logger.info("Swapped persona: %s", swapped_result.persona.name)

    # ----- Step 4 & 5: Compare and measure -----
    logger.info("Step 4-5: Comparing control vs swapped personas")

    control_text = _persona_text(control_result)
    swapped_text = _persona_text(swapped_result)

    control_eng_hits = _keyword_count(control_text, ENGINEERING_KEYWORDS)
    control_hc_hits = _keyword_count(control_text, HEALTHCARE_KEYWORDS)

    swapped_eng_hits = _keyword_count(swapped_text, ENGINEERING_KEYWORDS)
    swapped_hc_hits = _keyword_count(swapped_text, HEALTHCARE_KEYWORDS)

    # Data-following rate: does the swapped persona still have more engineering
    # keywords than healthcare keywords?
    swapped_follows_data = swapped_eng_hits > swapped_hc_hits
    swapped_follows_prompt = swapped_hc_hits > swapped_eng_hits

    if swapped_eng_hits + swapped_hc_hits > 0:
        data_following_rate = swapped_eng_hits / (swapped_eng_hits + swapped_hc_hits)
    else:
        data_following_rate = 0.5  # ambiguous

    results = {
        "experiment": "3.15 - Counterfactual swap test",
        "model": MODEL,
        "engineer_cluster_id": engineer_cluster.cluster_id,
        "num_sample_records": len(engineer_cluster.sample_records),
        "control": {
            "persona_name": control_result.persona.name,
            "persona_summary": control_result.persona.summary,
            "engineering_keyword_hits": control_eng_hits,
            "healthcare_keyword_hits": control_hc_hits,
            "groundedness_score": control_result.groundedness.score,
            "cost_usd": control_result.total_cost_usd,
            "attempts": control_result.attempts,
            "vocabulary": control_result.persona.vocabulary,
            "goals": control_result.persona.goals,
            "firmographic_industry": control_result.persona.firmographics.industry,
            "role_titles": control_result.persona.firmographics.role_titles,
        },
        "swapped": {
            "persona_name": swapped_result.persona.name,
            "persona_summary": swapped_result.persona.summary,
            "engineering_keyword_hits": swapped_eng_hits,
            "healthcare_keyword_hits": swapped_hc_hits,
            "groundedness_score": swapped_result.groundedness.score,
            "cost_usd": swapped_result.total_cost_usd,
            "attempts": swapped_result.attempts,
            "vocabulary": swapped_result.persona.vocabulary,
            "goals": swapped_result.persona.goals,
            "firmographic_industry": swapped_result.persona.firmographics.industry,
            "role_titles": swapped_result.persona.firmographics.role_titles,
        },
        "analysis": {
            "data_following_rate": round(data_following_rate, 3),
            "swapped_follows_data": swapped_follows_data,
            "swapped_follows_prompt": swapped_follows_prompt,
            "verdict": (
                "DATA-GROUNDED: Pipeline follows the actual behavioral data"
                if swapped_follows_data
                else (
                    "PROMPT-ANCHORED: Pipeline follows the tenant context prompt"
                    if swapped_follows_prompt
                    else "AMBIGUOUS: No clear signal dominance"
                )
            ),
        },
        "total_cost_usd": round(
            control_result.total_cost_usd + swapped_result.total_cost_usd, 4
        ),
    }

    return results


def main() -> None:
    results = asyncio.run(run_experiment())

    print("\n" + "=" * 72)
    print("EXPERIMENT 3.15 -- COUNTERFACTUAL SWAP TEST RESULTS")
    print("=" * 72)
    print(json.dumps(results, indent=2))
    print("=" * 72)

    # Summary
    analysis = results["analysis"]
    print(f"\nVerdict: {analysis['verdict']}")
    print(f"Data-following rate: {analysis['data_following_rate']:.1%}")
    print(
        f"  Swapped persona engineering keywords: "
        f"{results['swapped']['engineering_keyword_hits']}"
    )
    print(
        f"  Swapped persona healthcare keywords:  "
        f"{results['swapped']['healthcare_keyword_hits']}"
    )
    print(f"\nControl persona: {results['control']['persona_name']}")
    print(f"Swapped persona: {results['swapped']['persona_name']}")
    print(f"Total cost: ${results['total_cost_usd']:.4f}")
    print()


if __name__ == "__main__":
    main()
