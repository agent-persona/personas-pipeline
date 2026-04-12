"""exp-3.14 — Negative evidence handling.

Tests whether an explicit "say unknown" instruction reduces hallucination
of fields that have no supporting evidence in the source data.

Two conditions (3 trials each):
  - Condition A (baseline): standard system prompt
  - Condition B (treatment): system prompt + "leave gaps null" instruction

Audits 4 gap fields: demographics.income_bracket, demographics.education_level,
firmographics.industry, firmographics.company_size.

Hypothesis: baseline hallucinates ≥60% of missing fields; treatment drops to ≤20%.

Usage:
    python scripts/run_exp_3_14.py
"""

from __future__ import annotations

import asyncio
import json
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

from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.prompt_builder import SYSTEM_PROMPT  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import (  # noqa: E402
    ClusterData,
    ClusterSummary,
    EnrichmentPayload,
    SampleRecord,
    TenantContext,
)

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-3.14"
N_TRIALS = 3

TREATMENT_ADDENDUM = (
    "\n\nIMPORTANT: If no source evidence supports a field, leave it null or "
    "empty rather than guessing. Honest gaps are better than plausible inventions."
)


def build_gap_cluster() -> ClusterData:
    """Build a cluster from the sparse_gap_cluster fixture with only behavioral data."""
    fixture_path = (
        REPO_ROOT / "synthesis" / "synthesis" / "fixtures"
        / "sparse_gap_cluster" / "records.json"
    )
    with open(fixture_path) as f:
        raw_records = json.load(f)

    sample_records = [
        SampleRecord(
            record_id=r["record_id"],
            source=r["source"],
            timestamp=r.get("timestamp"),
            payload=r.get("payload", {}),
        )
        for r in raw_records
    ]

    return ClusterData(
        cluster_id="gap_cluster_behavioral_only",
        tenant=TenantContext(
            tenant_id="tenant_gap_test",
            industry="B2B SaaS",
            product_description="Workflow automation platform",
        ),
        summary=ClusterSummary(
            cluster_size=len(sample_records),
            top_behaviors=["page_view", "click", "cta_start_trial", "btn_see_demo", "cta_book_demo"],
            top_pages=["/pricing", "/features/integrations", "/blog/automation-guide", "/enterprise", "/case-studies"],
        ),
        sample_records=sample_records,
        enrichment=EnrichmentPayload(),  # deliberately empty
    )


def audit_gap_fields(persona_dict: dict) -> dict:
    """Check whether gap fields are null (acknowledged) or filled (hallucinated)."""
    demo = persona_dict.get("demographics", {})
    firmo = persona_dict.get("firmographics", {})

    gap_fields = {
        "demographics.income_bracket": demo.get("income_bracket"),
        "demographics.education_level": demo.get("education_level"),
        "firmographics.industry": firmo.get("industry"),
        "firmographics.company_size": firmo.get("company_size"),
    }

    results = {}
    for field_path, value in gap_fields.items():
        is_gap = (
            value is None
            or value == ""
            or (isinstance(value, str) and value.lower() in {"unknown", "n/a", "not specified", "not available"})
        )
        results[field_path] = {
            "value": value,
            "status": "acknowledged_gap" if is_gap else "hallucinated",
        }

    n_hallucinated = sum(1 for v in results.values() if v["status"] == "hallucinated")
    n_total = len(results)

    return {
        "field_audits": results,
        "hallucination_count": n_hallucinated,
        "hallucination_rate": n_hallucinated / n_total,
    }


async def run_trials(
    cluster: ClusterData,
    backend: AnthropicBackend,
    system_prompt: str,
    label: str,
    n_trials: int,
) -> list[dict]:
    """Run n_trials synthesis attempts and audit each."""
    results = []
    for trial in range(1, n_trials + 1):
        print(f"  [{label}] trial {trial}/{n_trials}")
        try:
            r = await synthesize(cluster, backend, system_prompt=system_prompt)
            p_dict = r.persona.model_dump(mode="json")
            audit = audit_gap_fields(p_dict)
            print(
                f"    [OK] {p_dict['name'][:40]}  "
                f"hallucination_rate={audit['hallucination_rate']:.0%}  "
                f"cost=${r.total_cost_usd:.4f}"
            )
            results.append({
                "trial": trial,
                "status": "ok",
                "persona": p_dict,
                "cost_usd": r.total_cost_usd,
                "groundedness": r.groundedness.score,
                "attempts": r.attempts,
                **audit,
            })
        except SynthesisError as e:
            print(f"    [FAIL] {e}")
            results.append({
                "trial": trial,
                "status": "failed",
                "error": str(e),
                "cost_usd": sum(a.cost_usd for a in e.attempts),
            })
    return results


async def main() -> None:
    print("=" * 72)
    print("exp-3.14 — Negative evidence handling")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    cluster = build_gap_cluster()
    print(f"Gap cluster: {cluster.cluster_id} ({len(cluster.sample_records)} records)")
    print("Note: cluster has ONLY behavioral events — no demographic/firmographic signals")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Condition A: baseline
    print("\n[1/2] Condition A — baseline (standard prompt)...")
    baseline_results = await run_trials(
        cluster, backend, SYSTEM_PROMPT, "baseline", N_TRIALS
    )

    # Condition B: treatment
    treatment_prompt = SYSTEM_PROMPT + TREATMENT_ADDENDUM
    print("\n[2/2] Condition B — treatment (say-unknown instruction)...")
    treatment_results = await run_trials(
        cluster, backend, treatment_prompt, "treatment", N_TRIALS
    )

    # Aggregate
    def agg(results: list[dict]) -> dict:
        ok = [r for r in results if r.get("status") == "ok"]
        rates = [r["hallucination_rate"] for r in ok]
        return {
            "n_trials": len(results),
            "n_ok": len(ok),
            "mean_hallucination_rate": sum(rates) / len(rates) if rates else None,
            "per_trial": [
                {
                    "trial": r["trial"],
                    "hallucination_rate": r.get("hallucination_rate"),
                    "hallucination_count": r.get("hallucination_count"),
                    "field_audits": r.get("field_audits"),
                }
                for r in results
                if r.get("status") == "ok"
            ],
        }

    baseline_agg = agg(baseline_results)
    treatment_agg = agg(treatment_results)

    b_rate = baseline_agg.get("mean_hallucination_rate")
    t_rate = treatment_agg.get("mean_hallucination_rate")

    summary = {
        "experiment_id": "3.14",
        "branch": "exp-3.14-negative-evidence",
        "model": settings.default_model,
        "n_trials": N_TRIALS,
        "gap_fields_audited": 4,
        "baseline": baseline_agg,
        "treatment": treatment_agg,
        "delta_hallucination_rate": (t_rate - b_rate) if t_rate is not None and b_rate is not None else None,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_personas.json").write_text(
        json.dumps(baseline_results, indent=2, default=str)
    )
    (OUTPUT_DIR / "treatment_personas.json").write_text(
        json.dumps(treatment_results, indent=2, default=str)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
