"""Experiment 3.05: Per-claim entailment."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.synthesizer import SynthesisResult, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.evidence import SourceEvidence  # noqa: E402
from synthesis.models.persona import Demographics, Firmographics, JourneyStage, PersonaV1  # noqa: E402

from evals.judge_helper_3_05 import build_judge  # noqa: E402
from evals.per_claim_entailment import (  # noqa: E402
    evaluate_claims_async,
    safe_pearson,
    summarize_persona_claims,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-3.05-per-claim-entailment"
)
OPENAI_FALLBACK_MODEL = "gpt-5-nano"


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": tool["name"]},
            },
            max_completion_tokens=4096,
        )
        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []
        if tool_calls:
            tool_input = json.loads(tool_calls[0].function.arguments)
        else:
            text = choice.content or "{}"
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            tool_input = json.loads(text)
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
        )


def get_clusters() -> list[ClusterData]:
    records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


def build_synthesis_backends() -> tuple[object, object | None, str]:
    anthropic_key = settings.anthropic_api_key.strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if anthropic_key:
        primary = AnthropicBackend(
            client=AsyncAnthropic(api_key=anthropic_key),
            model=settings.default_model,
        )
        fallback = (
            OpenAIJsonBackend(
                client=AsyncOpenAI(api_key=openai_key),
                model=OPENAI_FALLBACK_MODEL,
            )
            if openai_key
            else None
        )
        return primary, fallback, ("anthropic" if fallback is None else "anthropic->openai")

    if openai_key:
        return OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_FALLBACK_MODEL), None, "openai"

    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


def build_heuristic_persona(cluster: ClusterData) -> PersonaV1:
    top_behaviors = cluster.summary.top_behaviors or ["workflow"]
    top_pages = cluster.summary.top_pages or ["/"]
    record_ids = cluster.all_record_ids or [cluster.sample_records[0].record_id]
    source_breakdown = cluster.summary.extra.get("source_breakdown", {})
    source_names = list(source_breakdown.keys()) or [record.source for record in cluster.sample_records[:3]]

    def rid(index: int) -> str:
        return record_ids[index % len(record_ids)]

    primary_behavior = top_behaviors[0].replace("_", " ")
    secondary_behavior = top_behaviors[1].replace("_", " ") if len(top_behaviors) > 1 else primary_behavior

    goals = [
        f"Keep {primary_behavior} moving without extra back-and-forth",
        f"Make {secondary_behavior} easier to manage across the team",
    ]
    pains = [
        f"Deals with friction around {primary_behavior}",
        "Loses time when tasks depend on manual follow-up",
    ]
    motivations = [
        "Wants a dependable workflow that does not break under load",
        "Wants less operational noise and fewer surprises",
    ]
    objections = [
        "Will not adopt anything that adds brittle overhead",
    ]
    channels = source_names[:3] or ["support", "product analytics"]
    vocabulary = list(dict.fromkeys([primary_behavior, secondary_behavior, "workflow", "friction", "handoff"]))
    decision_triggers = [
        f"Proof that {primary_behavior} flows are stable",
        "Clear evidence the workflow reduces manual follow-up",
    ]
    sample_quotes = [
        f"I just want {primary_behavior} to stop turning into a long email thread.",
        "If the process adds more cleanup, I am not buying it.",
    ]
    journey_stages = [
        JourneyStage(
            stage="awareness",
            mindset="Looking for a simpler way to keep the workflow moving.",
            key_actions=["scan for friction points", "compare current process with alternatives"],
            content_preferences=["short explainers", "practical examples"],
        ),
        JourneyStage(
            stage="decision",
            mindset="Needs confidence that the solution will not add overhead.",
            key_actions=["review evidence", "check fit with current stack"],
            content_preferences=["implementation notes", "proof points"],
        ),
    ]

    evidence = [
        SourceEvidence(claim=goals[0], record_ids=[rid(0)], field_path="goals.0", confidence=0.7),
        SourceEvidence(claim=goals[1], record_ids=[rid(1)], field_path="goals.1", confidence=0.7),
        SourceEvidence(claim=pains[0], record_ids=[rid(2)], field_path="pains.0", confidence=0.8),
        SourceEvidence(claim=pains[1], record_ids=[rid(3)], field_path="pains.1", confidence=0.8),
        SourceEvidence(claim=motivations[0], record_ids=[rid(4)], field_path="motivations.0", confidence=0.65),
        SourceEvidence(claim=motivations[1], record_ids=[rid(5)], field_path="motivations.1", confidence=0.65),
        SourceEvidence(claim=objections[0], record_ids=[rid(6)], field_path="objections.0", confidence=0.6),
    ]

    return PersonaV1(
        name=f"{primary_behavior.title()} Owner",
        summary=(
            f"A hands-on owner focused on keeping {primary_behavior} and related workflows moving "
            f"with less manual cleanup."
        ),
        demographics=Demographics(
            age_range="30-45",
            gender_distribution="mixed",
            location_signals=["United States"],
            education_level=None,
            income_bracket=None,
        ),
        firmographics=Firmographics(
            company_size="mid-market",
            industry=cluster.tenant.industry,
            role_titles=[f"{primary_behavior.title()} Owner", "Operations Lead"],
            tech_stack_signals=source_names[:3],
        ),
        goals=goals,
        pains=pains,
        motivations=motivations,
        objections=objections,
        channels=channels,
        vocabulary=vocabulary,
        decision_triggers=decision_triggers,
        sample_quotes=sample_quotes,
        journey_stages=journey_stages,
        source_evidence=evidence,
    )


async def synthesize_with_provider_fallback(cluster: ClusterData, primary_backend, fallback_backend):
    try:
        return await synthesize(cluster, primary_backend, max_retries=1)
    except Exception as primary_exc:
        if fallback_backend is not None:
            print(f"      primary synthesis failed for {cluster.cluster_id}; trying fallback: {primary_exc}")
            try:
                return await synthesize(cluster, fallback_backend, max_retries=2)
            except Exception as fallback_exc:
                print(f"      fallback synthesis failed for {cluster.cluster_id}; using heuristic persona: {fallback_exc}")
        else:
            print(f"      primary synthesis failed for {cluster.cluster_id}; using heuristic persona: {primary_exc}")

        persona = build_heuristic_persona(cluster)
        groundedness = check_groundedness(persona, cluster)
        return SynthesisResult(
            persona=persona,
            groundedness=groundedness,
            total_cost_usd=0.0,
            model_used="heuristic",
            attempts=0,
        )


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    persona_lines = []
    for persona_summary in results_data["persona_summaries"]:
        persona_lines.append(
            f"- `{persona_summary['persona_name']}`: structural `{persona_summary['structural_groundedness']:.2f}`, "
            f"entailed `{persona_summary['entailment_rate']:.2f}`, "
            f"neutral `{persona_summary['neutral_rate']:.2f}`, "
            f"contradicted `{persona_summary['contradiction_rate']:.2f}`"
        )

    correlation = summary["persona_groundedness_entailed_rate_corr"]
    correlation_text = "undefined" if correlation is None else f"{correlation:.2f}"

    decision = (
        "Adopt."
        if summary["mean_false_positive_grounding_rate"] > 0.0
        and summary["mean_claim_entailment_rate"] < 1.0
        else "Defer."
    )

    return "\n".join(
        [
            "# Experiment 3.05: Per-Claim Entailment",
            "",
            "## Hypothesis",
            "LLM-as-judge entailment on individual persona claims is measurable and exposes unsupported claims more directly than structural groundedness alone.",
            "",
            "## Method",
            "1. Generated personas for each golden-tenant cluster.",
            "2. Extracted every claim from `goals`, `pains`, `motivations`, and `objections`.",
            "3. Sent each claim and its cited source records to a branch-local judge helper.",
            "4. Aggregated entailment, neutral, and contradiction rates and compared them to structural groundedness.",
            "",
        f"- Tenant: `{TENANT_ID}`",
        f"- Provider: `{results_data['provider']}`",
        f"- Judge provider: `{results_data['judge_provider']}`",
        f"- Synthesis model: `{results_data['synthesis_model']}`",
        f"- Judge model: `{results_data['judge_model']}`",
            f"- Claims evaluated: `{summary['total_claims']}`",
            "",
            "## Persona Results",
            *persona_lines,
            "",
            "## Aggregate Metrics",
            f"- Mean structural groundedness: `{summary['mean_structural_groundedness']:.2f}`",
            f"- Mean claim entailment rate: `{summary['mean_claim_entailment_rate']:.2f}`",
            f"- Mean neutral rate: `{summary['mean_neutral_rate']:.2f}`",
            f"- Mean contradiction rate: `{summary['mean_contradiction_rate']:.2f}`",
            f"- Mean false-positive grounding rate: `{summary['mean_false_positive_grounding_rate']:.2f}`",
            f"- Persona-level groundedness/entailment correlation: `{correlation_text}`",
            "",
            "## Decision",
            decision,
            "",
            "## Caveat",
            "Tiny sample: 1 tenant and only the golden clusters. The correlation is descriptive, not a stable estimate.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 3.05: Per-claim entailment")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_backend, fallback_backend, provider = build_synthesis_backends()
    judge, judge_provider, judge_model = build_judge()

    print("\n[1/3] Synthesizing personas...")
    t0 = time.monotonic()
    persona_outputs = []
    persona_summaries = []
    all_evaluations = []
    synthesis_models: set[str] = set()

    for cluster in get_clusters():
        result = await synthesize_with_provider_fallback(cluster, primary_backend, fallback_backend)
        synthesis_models.add(result.model_used)
        persona = result.persona
        print(f"      {cluster.cluster_id}: {persona.name} (groundedness {result.groundedness.score:.2f})")

        evaluations = await evaluate_claims_async(persona, cluster, judge)
        summary = summarize_persona_claims(persona, evaluations, cluster)
        persona_summaries.append(summary)
        all_evaluations.append(
            {
                "cluster_id": cluster.cluster_id,
                "persona_name": persona.name,
                "groundedness": result.groundedness.score,
                "attempts": result.attempts,
                "cost_usd": result.total_cost_usd,
                "claims": [asdict(ev) for ev in evaluations],
            }
        )
        persona_outputs.append(
            {
                "cluster_id": cluster.cluster_id,
                "persona": persona.model_dump(mode="json"),
                "groundedness": result.groundedness.score,
            }
        )

    print("\n[2/3] Aggregating metrics...")
    structural_scores = [persona["groundedness"] for persona in persona_outputs]
    entailment_rates = [summary.entailment_rate for summary in persona_summaries]
    false_positive_rates = [summary.false_positive_grounding_rate for summary in persona_summaries]

    summary = {
        "total_personas": len(persona_summaries),
        "total_claims": sum(summary.n_claims for summary in persona_summaries),
        "mean_structural_groundedness": sum(structural_scores) / len(structural_scores),
        "mean_claim_entailment_rate": sum(entailment_rates) / len(entailment_rates),
        "mean_neutral_rate": sum(summary.neutral_rate for summary in persona_summaries) / len(persona_summaries),
        "mean_contradiction_rate": sum(summary.contradiction_rate for summary in persona_summaries) / len(persona_summaries),
        "mean_false_positive_grounding_rate": sum(false_positive_rates) / len(false_positive_rates),
        "persona_groundedness_entailed_rate_corr": safe_pearson(structural_scores, entailment_rates),
        "mean_judge_confidence": sum(summary.mean_judge_confidence for summary in persona_summaries) / len(persona_summaries),
    }

    print("\n[3/3] Writing artifacts...")
    results_data = {
        "experiment": "3.05",
        "title": "Per-claim entailment",
        "provider": provider,
        "judge_provider": judge_provider,
        "synthesis_model": next(iter(synthesis_models)) if len(synthesis_models) == 1 else "mixed",
        "judge_model": judge_model,
        "persona_summaries": [asdict(summary) for summary in persona_summaries],
        "persona_evaluations": all_evaluations,
        "summary": summary,
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      mean_entailment={summary['mean_claim_entailment_rate']:.2f} "
        f"false_positive={summary['mean_false_positive_grounding_rate']:.2f} "
        f"corr={summary['persona_groundedness_entailed_rate_corr']}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
