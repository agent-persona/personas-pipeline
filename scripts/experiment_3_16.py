"""Experiment 3.16: synthetic ground-truth injection."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from crawler import fetch_all  # noqa: E402
from evals.ground_truth_injection import (  # noqa: E402
    FACT_SPECS,
    AppliedFact,
    ClusterComparison,
    ExperimentSummary,
    PersonaRun,
    assess_fact,
    applied_facts_to_dict,
    inject_synthetic_facts,
)
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-3.16-synthetic-ground-truth-injection"
)
OPENAI_MODEL = "gpt-5-nano"


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        response = await self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=messages,
            max_output_tokens=8192,
            reasoning={"effort": "low"},
            text_format=PersonaV1,
        )
        text = response.output_text or "{}"
        tool_input = PersonaV1.model_validate_json(text).model_dump(mode="json")
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model,
        )


class FallbackGenerateBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        try:
            return await self.primary.generate(system=system, messages=messages, tool=tool)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.generate(system=system, messages=messages, tool=tool)


def load_records() -> list[RawRecord]:
    crawler_records = fetch_all(TENANT_ID)
    return [RawRecord.model_validate(r.model_dump()) for r in crawler_records]


def build_clusters(records: list[RawRecord]) -> list[ClusterData]:
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


def infer_cluster_label(cluster: ClusterData) -> str:
    behaviors = {behavior.lower() for behavior in cluster.summary.top_behaviors}
    designer_markers = {
        "template_browsing",
        "color_picker",
        "asset_export",
        "client_share",
        "comment_threading",
        "brand_kit_creation",
    }
    if behaviors & designer_markers:
        return "designer"
    return "engineer"


def openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_backend() -> tuple[FallbackGenerateBackend | AnthropicBackend | OpenAIJsonBackend, str, str]:
    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(openai_key())
    if has_anthropic and has_openai:
        primary = AnthropicBackend(
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=settings.default_model,
        )
        fallback = OpenAIJsonBackend(
            client=AsyncOpenAI(api_key=openai_key()),
            model=OPENAI_MODEL,
        )
        return FallbackGenerateBackend(primary=primary, fallback=fallback), "anthropic->openai", f"{settings.default_model}->{OPENAI_MODEL}"
    if has_anthropic:
        return AnthropicBackend(client=AsyncAnthropic(api_key=settings.anthropic_api_key), model=settings.default_model), "anthropic", settings.default_model
    if has_openai:
        return OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key()), model=OPENAI_MODEL), "openai", OPENAI_MODEL
    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


async def synthesize_with_retry(cluster: ClusterData, backend, max_attempts: int = 4):
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return await synthesize(cluster, backend, max_retries=4)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"synthesis failed for {cluster.cluster_id}") from last_error


async def run_pipeline(records: list[RawRecord], backend) -> dict[str, PersonaRun]:
    cluster_runs: dict[str, PersonaRun] = {}
    for cluster in build_clusters(records):
        cluster_label = infer_cluster_label(cluster)
        result = await synthesize_with_retry(cluster, backend)
        cluster_runs[cluster_label] = PersonaRun(
            cluster_id=cluster.cluster_id,
            persona_name=result.persona.name,
            groundedness=result.groundedness.score,
            cost_usd=result.total_cost_usd,
            attempts=result.attempts,
            persona=result.persona.model_dump(mode="json"),
        )
    return cluster_runs


def write_findings(results: dict) -> str:
    summary = results["summary"]
    cluster_lines = []
    for cluster in results["clusters"]:
        cluster_lines.append(
            f"- `{cluster['cluster_id']}`: control `{cluster['control']['groundedness']:.2f}` "
            f"-> injected `{cluster['injected']['groundedness']:.2f}`, "
            f"survival `{cluster['injected_fact_survival']:.2f}`"
        )

    fact_lines = []
    for fact in results["facts"]:
        fact_lines.append(
            f"- `{fact['fact_id']}` on `{fact['record_id']}`: "
            f"control `{fact['control_surface_hit']}` / injected `{fact['injected_surface_hit']}` "
            f"(evidence `{fact['injected_evidence_hit']}`)"
        )

    decision = (
        "Adopt. The injected facts survived into the synthesized personas often enough to show the pipeline is carrying planted ground truth through to outputs."
        if summary["injected_survival_rate"] >= 0.5 and summary["injected_survival_rate"] > summary["control_survival_rate"]
        else "Defer. The injected facts were not retained strongly enough to justify calling the pipeline robust."
    )

    return "\n".join(
        [
            "# Experiment 3.16: Synthetic Ground Truth Injection",
            "",
            "## Hypothesis",
            "Known planted facts should survive synthesis if the pipeline is truly grounded.",
            "",
            "## Method",
            f"1. Loaded `{summary['n_records']}` mock tenant records for `tenant_acme_corp`.",
            f"2. Injected `{summary['n_facts']}` distinctive facts into payloads without changing clustering features.",
            "3. Ran the pipeline on control and injected fixtures with the same cluster topology.",
            "4. Scored survival by matching planted fact signals against persona text and source evidence.",
            "",
            f"- Provider: `{results['provider']}`",
            f"- Synthesis model: `{results['synthesis_model']}`",
            "",
            "## Cluster Comparison",
            *cluster_lines,
            "",
            "## Fact Survival",
            *fact_lines,
            "",
            "## Summary",
            f"- Control survival rate: `{summary['control_survival_rate']:.2f}`",
            f"- Injected survival rate: `{summary['injected_survival_rate']:.2f}`",
            f"- Mean control groundedness: `{summary['mean_control_groundedness']:.2f}`",
            f"- Mean injected groundedness: `{summary['mean_injected_groundedness']:.2f}`",
            f"- Mean control cost: `${summary['mean_control_cost_usd']:.4f}`",
            f"- Mean injected cost: `${summary['mean_injected_cost_usd']:.4f}`",
            "",
            "## Decision",
            decision,
            "",
            "## Caveat",
            "Small sample: 1 tenant, 2 clusters, and synthetic facts were intentionally distinctive. This is a retention probe, not a general generalization benchmark.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 3.16: Synthetic ground truth injection")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    backend, provider, synthesis_model = build_backend()

    print("\n[1/4] Loading control fixture...")
    records = load_records()
    control_clusters = build_clusters(records)
    print(f"      records={len(records)} clusters={len(control_clusters)}")

    print("[2/4] Injecting planted facts...")
    injected_records, applied_facts = inject_synthetic_facts(records)
    print(f"      facts={len(applied_facts)}")

    print("[3/4] Synthesizing control and injected personas...")
    t0 = time.monotonic()
    control_runs = await run_pipeline(records, backend)
    injected_runs = await run_pipeline(injected_records, backend)

    print("[4/4] Analyzing survival...")
    cluster_entries = []
    for cluster_label, injected_run in injected_runs.items():
        control_run = control_runs[cluster_label]
        fact_outcomes = []
        for fact in applied_facts:
            if fact.cluster_label != cluster_label:
                continue
            control_score, control_surface, control_evidence, control_excerpt = assess_fact(control_run.persona, fact)
            injected_score, injected_surface, injected_evidence, injected_excerpt = assess_fact(injected_run.persona, fact)
            fact_outcomes.append(
                {
                    "fact_id": fact.fact_id,
                    "cluster_label": fact.cluster_label,
                    "record_id": fact.record_id,
                    "payload_value": fact.payload_value,
                    "signals": list(fact.signals),
                    "control_signal_score": control_score,
                    "injected_signal_score": injected_score,
                    "control_surface_hit": control_surface,
                    "injected_surface_hit": injected_surface,
                    "control_evidence_hit": control_evidence,
                    "injected_evidence_hit": injected_evidence,
                    "control_excerpt": control_excerpt,
                    "injected_excerpt": injected_excerpt,
                }
            )

        cluster_entries.append(
            {
                "cluster_id": injected_run.cluster_id,
                "cluster_label": cluster_label,
                "control": {
                    "persona_name": control_run.persona_name,
                    "groundedness": control_run.groundedness,
                    "cost_usd": control_run.cost_usd,
                    "attempts": control_run.attempts,
                },
                "injected": {
                    "persona_name": injected_run.persona_name,
                    "groundedness": injected_run.groundedness,
                    "cost_usd": injected_run.cost_usd,
                    "attempts": injected_run.attempts,
                },
                "fact_outcomes": fact_outcomes,
                "injected_fact_survival": (
                    sum(1 for fact in fact_outcomes if fact["injected_surface_hit"]) / len(fact_outcomes)
                    if fact_outcomes
                    else 0.0
                ),
            }
        )

    fact_outcomes_flat = []
    for fact in applied_facts:
        control_persona = control_runs[fact.cluster_label].persona
        injected_persona = injected_runs[fact.cluster_label].persona
        control_score, control_surface, control_evidence, control_excerpt = assess_fact(control_persona, fact)
        injected_score, injected_surface, injected_evidence, injected_excerpt = assess_fact(injected_persona, fact)
        fact_outcomes_flat.append(
            {
                "fact_id": fact.fact_id,
                "cluster_label": fact.cluster_label,
                "record_id": fact.record_id,
                "payload_value": fact.payload_value,
                "signals": list(fact.signals),
                "control_signal_score": control_score,
                "injected_signal_score": injected_score,
                "control_surface_hit": control_surface,
                "injected_surface_hit": injected_surface,
                "control_evidence_hit": control_evidence,
                "injected_evidence_hit": injected_evidence,
                "control_excerpt": control_excerpt,
                "injected_excerpt": injected_excerpt,
            }
        )
    summary = {
        "tenant_id": TENANT_ID,
        "n_records": len(records),
        "n_clusters": len(control_runs),
        "n_facts": len(applied_facts),
        "mean_control_groundedness": sum(run.groundedness for run in control_runs.values()) / len(control_runs),
        "mean_injected_groundedness": sum(run.groundedness for run in injected_runs.values()) / len(injected_runs),
        "mean_control_cost_usd": sum(run.cost_usd for run in control_runs.values()) / len(control_runs),
        "mean_injected_cost_usd": sum(run.cost_usd for run in injected_runs.values()) / len(injected_runs),
        "mean_control_fact_survival": sum(1.0 if fact["control_surface_hit"] else 0.0 for fact in fact_outcomes_flat) / len(fact_outcomes_flat),
        "mean_injected_fact_survival": sum(1.0 if fact["injected_surface_hit"] else 0.0 for fact in fact_outcomes_flat) / len(fact_outcomes_flat),
        "control_survival_rate": sum(1.0 if fact["control_surface_hit"] else 0.0 for fact in fact_outcomes_flat) / len(fact_outcomes_flat),
        "injected_survival_rate": sum(1.0 if fact["injected_surface_hit"] else 0.0 for fact in fact_outcomes_flat) / len(fact_outcomes_flat),
        "provider": provider,
        "synthesis_model": synthesis_model,
        "duration_seconds": time.monotonic() - t0,
    }

    results = {
        "experiment": "3.16",
        "title": "Synthetic ground truth injection",
        "provider": provider,
        "synthesis_model": synthesis_model,
        "records": len(records),
        "facts": fact_outcomes_flat,
        "clusters": cluster_entries,
        "summary": summary,
        "applied_facts": applied_facts_to_dict(applied_facts),
        "duration_seconds": summary["duration_seconds"],
    }

    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(write_findings(results))

    print(
        f"      control_survival={summary['control_survival_rate']:.2f} "
        f"injected_survival={summary['injected_survival_rate']:.2f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
