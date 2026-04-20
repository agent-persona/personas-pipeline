"""Experiment 1.09: Hierarchical vs flat schema

Self-contained runner. Imports only canonical personas-pipeline modules.

Run from repo root:
    python output/experiments/exp-1.09-hierarchical-vs-flat/run_experiment.py

Or via the unified runner:
    python -m scripts.experiments.run_yash_experiments 1.09

Hypothesis:
    Hierarchical schemas with semantic grouping produce higher coherence than flat schemas because the LLM can structure related fields together.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add pipeline subpackages to path so canonical imports work
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
for pkg in ["synthesis", "evaluation", "twin", "orchestration", "segmentation", "crawler"]:
    sys.path.insert(0, str(PIPELINE_ROOT / pkg))

import anthropic
from synthesis.engine.synthesizer import synthesize
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.models.cluster import (
    ClusterData, ClusterSummary, EnrichmentPayload, SampleRecord, TenantContext,
)
from evaluation.judges import LLMJudge
from evaluation.golden_set import load_golden_set


def make_test_cluster() -> ClusterData:
    tenants = load_golden_set()
    tenant = tenants[0]
    return ClusterData(
        cluster_id="cluster_001",
        tenant=TenantContext(
            tenant_id=tenant.tenant_id,
            industry=tenant.industry,
            product_description=tenant.product_description,
            existing_persona_names=[],
        ),
        summary=ClusterSummary(
            cluster_size=150,
            top_behaviors=["views pricing page", "downloads whitepaper", "requests demo"],
            top_pages=["/pricing", "/features", "/case-studies"],
            conversion_rate=0.12,
            avg_session_duration_seconds=340,
            top_referrers=["google", "linkedin", "direct"],
        ),
        sample_records=[
            SampleRecord(record_id="rec_001", source="ga4", timestamp="2026-03-15",
                         payload={"page": "/pricing", "duration": 120}),
            SampleRecord(record_id="rec_002", source="hubspot", timestamp="2026-03-16",
                         payload={"type": "form_submit", "form": "demo_request"}),
            SampleRecord(record_id="rec_003", source="intercom", timestamp="2026-03-17",
                         payload={"message": "How does your tool integrate with Jira?"}),
            SampleRecord(record_id="rec_004", source="ga4", timestamp="2026-03-18",
                         payload={"page": "/case-studies/enterprise", "duration": 95}),
            SampleRecord(record_id="rec_005", source="hubspot", timestamp="2026-03-19",
                         payload={"type": "email_open", "campaign": "q1_nurture"}),
        ],
        enrichment=EnrichmentPayload(
            firmographic={"median_company_size": "50-200"},
            intent_signals=["evaluating project management tools"],
            technographic={"tools": ["Jira", "Slack", "GitHub"]},
        ),
    )


async def main() -> None:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    backend = AnthropicBackend(client=client, model="claude-haiku-4-5-20251001")
    judge = LLMJudge(model="gpt-4o")

    cluster = make_test_cluster()
    result = await synthesize(cluster, backend)
    persona = result.persona.model_dump()

    score = await judge.score_persona(persona)

    output = {
        "experiment_id": "1.09",
        "persona": persona,
        "judge_score": {
            "overall": score.overall,
            "dimensions": score.dimensions,
            "rationale": score.rationale,
        },
        "synthesis_cost_usd": result.total_cost_usd,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
