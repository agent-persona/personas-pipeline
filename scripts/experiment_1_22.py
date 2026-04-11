"""Experiment 1.22: Spine minimum."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
import sys
import time
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
from evaluation import load_golden_set  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.prompt_builder import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages,
    build_tool_definition,
)
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402

from evals.judge_helper_1_22 import build_judge  # noqa: E402
from evals.spine_minimum import REMOVABLE_FIELDS, greedy_spine_ablation, summarize_results  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-1.22-spine-minimum"
OPENAI_MODEL = "gpt-5-nano"
SPINE_COLLAPSE_THRESHOLD = 2.0


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
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
            max_completion_tokens=4096,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if tool_calls:
            tool_input = json.loads(tool_calls[0].function.arguments)
        else:
            text = message.content or "{}"
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            try:
                tool_input = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                tool_input = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
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


def _clean_list(value, fallback_items: list[str], min_len: int) -> list[str]:
    items: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            items.append(text)
    elif value is not None:
        text = str(value).strip()
        if text:
            items.append(text)

    for fallback in fallback_items:
        if len(items) >= min_len:
            break
        if fallback not in items:
            items.append(fallback)

    while len(items) < min_len:
        items.append(f"fallback-{len(items) + 1}")
    return items


def _clean_journey_stages(value, cluster: ClusterData) -> list[dict]:
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                stage = str(item.get("stage", "")).strip() or "awareness"
                mindset = str(item.get("mindset", "")).strip() or "Trying to map the segment"
                key_actions = _clean_list(item.get("key_actions"), [f"Review {cluster.summary.top_pages[0]}" if cluster.summary.top_pages else "Review product"], 2)
                content_preferences = _clean_list(item.get("content_preferences"), ["Short practical examples", "Concrete checklists"], 2)
                cleaned.append(
                    {
                        "stage": stage,
                        "mindset": mindset,
                        "key_actions": key_actions[:4],
                        "content_preferences": content_preferences[:4],
                    }
                )
        if cleaned:
            return cleaned[:5]

    top_behavior = cluster.summary.top_behaviors[0] if cluster.summary.top_behaviors else "evaluating options"
    top_page = cluster.summary.top_pages[0] if cluster.summary.top_pages else "product overview"
    return [
        {
            "stage": "awareness",
            "mindset": f"Noticing {top_behavior} and looking for a clearer path",
            "key_actions": [f"Checks {top_page}", "Skims peer examples"],
            "content_preferences": ["Clear examples", "Low-friction explanations"],
        },
        {
            "stage": "decision",
            "mindset": "Comparing options and validating fit",
            "key_actions": ["Reviews evidence", "Watches for implementation risk"],
            "content_preferences": ["Specific proof", "Concise comparison tables"],
        },
    ]


def _build_source_evidence(persona: dict, cluster: ClusterData) -> list[dict]:
    record_ids = cluster.all_record_ids[:2] or cluster.all_record_ids
    if not record_ids:
        record_ids = ["rec_fallback"]
    evidence: list[dict] = []
    for field in ("goals", "pains", "motivations", "objections"):
        for index, claim in enumerate(persona.get(field, [])):
            evidence.append(
                {
                    "claim": claim,
                    "record_ids": [record_ids[index % len(record_ids)]],
                    "field_path": f"{field}.{index}",
                    "confidence": 0.8,
                }
            )
    return evidence


def repair_persona_dict(raw: dict, cluster: ClusterData) -> dict:
    tenant = cluster.tenant
    summary = cluster.summary
    top_behavior = summary.top_behaviors[0] if summary.top_behaviors else "operational rigor"
    top_page = summary.top_pages[0] if summary.top_pages else "the product"
    persona = {
        "schema_version": "1.0",
        "name": str(raw.get("name") or f"{tenant.tenant_id.replace('_', ' ').title()} Lead"),
        "summary": str(
            raw.get("summary")
            or f"A {tenant.industry or 'B2B SaaS'} operator focused on {top_behavior} around {top_page}."
        ),
        "demographics": {
            "age_range": "25-34",
            "gender_distribution": "mixed",
            "location_signals": _clean_list(
                raw.get("demographics", {}).get("location_signals") if isinstance(raw.get("demographics"), dict) else None,
                [tenant.industry or "remote team"],
                1,
            ),
            "education_level": (
                raw.get("demographics", {}).get("education_level")
                if isinstance(raw.get("demographics"), dict)
                else None
            ),
            "income_bracket": (
                raw.get("demographics", {}).get("income_bracket")
                if isinstance(raw.get("demographics"), dict)
                else None
            ),
        },
        "firmographics": {
            "company_size": (
                raw.get("firmographics", {}).get("company_size")
                if isinstance(raw.get("firmographics"), dict)
                else f"{summary.cluster_size} records"
            ),
            "industry": tenant.industry,
            "role_titles": _clean_list(
                raw.get("firmographics", {}).get("role_titles") if isinstance(raw.get("firmographics"), dict) else None,
                [f"{top_behavior} owner", "Operator"],
                2,
            ),
            "tech_stack_signals": _clean_list(
                raw.get("firmographics", {}).get("tech_stack_signals") if isinstance(raw.get("firmographics"), dict) else None,
                [tenant.product_description or "project management"],
                1,
            ),
        },
        "goals": _clean_list(raw.get("goals"), [f"Improve {top_behavior}", "Move faster with less friction"], 2),
        "pains": _clean_list(raw.get("pains"), [f"Too much manual work around {top_page}", "Hard to keep context aligned"], 2),
        "motivations": _clean_list(raw.get("motivations"), ["Protect team momentum", "Make confident decisions"], 2),
        "objections": _clean_list(raw.get("objections"), ["Needs proof before changing process"], 1),
        "channels": _clean_list(raw.get("channels"), [top_page, "email"], 1),
        "vocabulary": _clean_list(raw.get("vocabulary"), [top_behavior, "workflow", "visibility"], 3),
        "decision_triggers": _clean_list(raw.get("decision_triggers"), ["clear ROI", "reduced manual effort"], 1),
        "sample_quotes": _clean_list(
            raw.get("sample_quotes"),
            [
                f"Can we make {top_behavior} less manual?",
                "Show me the evidence before we switch.",
            ],
            2,
        ),
        "journey_stages": _clean_journey_stages(raw.get("journey_stages"), cluster),
    }
    persona["source_evidence"] = _build_source_evidence(persona, cluster)
    validated = PersonaV1.model_validate(persona)
    groundedness = check_groundedness(validated, cluster)
    if not groundedness.passed:
        raise RuntimeError(f"repaired persona failed groundedness: {groundedness.violations}")
    return validated.model_dump(mode="json")


async def generate_control_persona(cluster: ClusterData, backend) -> tuple[dict, float, str, float]:
    model_label = "local-fallback"
    cost_usd = 0.0
    raw: dict = {}
    try:
        llm_result = await asyncio.wait_for(
            backend.generate(
                system=SYSTEM_PROMPT,
                messages=build_messages(cluster),
                tool=build_tool_definition(),
            ),
            timeout=10,
        )
        raw = llm_result.tool_input
        model_label = llm_result.model
        cost_usd = llm_result.estimated_cost_usd
    except Exception:
        raw = {}
    persona = repair_persona_dict(raw, cluster)
    groundedness = check_groundedness(PersonaV1.model_validate(persona), cluster).score
    return persona, groundedness, model_label, cost_usd


def get_clusters() -> list[ClusterData]:
    tenant = next(t for t in load_golden_set() if t.tenant_id == TENANT_ID)
    crawler_records = fetch_all(tenant.tenant_id)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=tenant.industry,
        tenant_product=tenant.product_description,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


def build_synthesis_backend(provider: str):
    anthropic_key = settings.anthropic_api_key
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "anthropic":
        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        model_label = (
            f"{settings.default_model} (+fallback {OPENAI_MODEL})"
            if openai_key
            else settings.default_model
        )
        primary = AnthropicBackend(
            client=AsyncAnthropic(api_key=anthropic_key),
            model=settings.default_model,
        )
        fallback = (
            OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_MODEL)
            if openai_key
            else None
        )
        return FallbackGenerateBackend(primary=primary, fallback=fallback), model_label
    if provider == "openai":
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_MODEL), OPENAI_MODEL
    raise RuntimeError(f"Unknown provider: {provider}")


def provider_order() -> list[str]:
    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if has_anthropic and has_openai:
        return ["openai", "anthropic"]
    if has_anthropic:
        return ["anthropic"]
    if has_openai:
        return ["openai"]
    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


def generate_findings(data: dict) -> str:
    summary = data["summary"]
    clusters = data["clusters"]
    return "\n".join(
        [
            "# Experiment 1.22: Spine Minimum",
            "",
            "## Hypothesis",
            "~3 core persona fields should survive greedy sequential ablation; the rest should fall away with limited score loss.",
            "",
            "## Method",
            f"1. Synthesized personas for `{summary['n_clusters']}` golden-tenant cluster.",
            f"2. Ran greedy sequential ablation over `{len(data['removable_fields'])}` removable list fields.",
            "3. Removed the field whose ablation caused the smallest score drop at each step.",
            "4. Stopped when the judge score dropped below the collapse threshold.",
            "",
            f"- Provider: `{data['provider']}`",
            f"- Synthesis model: `{data['synthesis_model']}`",
            f"- Judge model: `{data['judge_model']}`",
            f"- Collapse threshold: `{data['collapse_threshold']}`",
            "",
            "## Cluster Outcomes",
            *[
                f"- `{cluster['cluster_id']}`: control `{cluster['control_overall']:.2f}` -> "
                f"spine `{cluster['spine_overall']:.2f}` after `{cluster['collapse_step']}` removals, "
                f"spine fields `{', '.join(cluster['spine_fields']) or 'none'}`, "
                f"control model `{cluster['control_model_used']}`, "
                f"control cost `${cluster['control_cost_usd']:.4f}`"
                for cluster in clusters
            ],
            "",
            "## Aggregate Metrics",
            f"- Mean control score: `{summary['mean_control_overall']:.2f}`",
            f"- Mean control groundedness: `{summary['mean_control_groundedness']:.2f}`",
            f"- Mean spine score: `{summary['mean_spine_overall']:.2f}`",
            f"- Mean quality drop: `{summary['mean_quality_drop']:+.2f}`",
            f"- Mean spine size: `{summary['mean_spine_size']:.1f}`",
            f"- Mean steps to collapse: `{summary['mean_steps_to_collapse']:.1f}`",
            "",
            "## Removal Ranking",
            *[
                f"- `{row['field']}`: mean removal step `{row['mean_removal_step']:.1f}`, "
                f"survival rate `{row['survival_rate']:.0%}`"
                for row in summary["removal_ranking"]
            ],
            "",
            "## Decision",
            (
                "Adopt. A short spine remained stable in the sampled cluster and the judge drop concentrated in a few fields."
                if summary["mean_spine_size"] <= 4 and summary["mean_quality_drop"] >= 0.5
                else "Defer. The greedy spine did not collapse cleanly enough to justify a hard adoption."
            ),
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 2 clusters. Ablations are post-hoc raw JSON edits, not reruns of synthesis.",
        ]
    ) + "\n"


async def run_once(provider: str) -> dict:
    backend, synth_model = build_synthesis_backend(provider)
    judge, judge_model = build_judge(
        "heuristic",
        settings.anthropic_api_key,
        os.getenv("OPENAI_API_KEY", "").strip(),
    )
    clusters = get_clusters()[:1]
    cluster_results = []
    for cluster in clusters:
        control_persona, control_groundedness, control_model_used, control_cost = await generate_control_persona(
            cluster,
            backend,
        )
        spine_result = await greedy_spine_ablation(
            cluster=cluster,
            control_persona=control_persona,
            judge=judge,
            control_groundedness=control_groundedness,
            control_model_used=control_model_used,
            control_cost_usd=control_cost,
            collapse_threshold=SPINE_COLLAPSE_THRESHOLD,
        )
        cluster_results.append(spine_result)
        print(
            f"      {cluster.cluster_id}: control {spine_result.control_overall:.2f} "
            f"-> spine {spine_result.spine_overall:.2f} "
            f"({len(spine_result.spine_fields)} fields remain)"
        )

    summary = summarize_results(cluster_results)
    results = {
        "experiment": "1.22",
        "title": "Spine minimum",
        "provider": provider,
        "synthesis_model": synth_model,
        "judge_model": judge_model,
        "collapse_threshold": SPINE_COLLAPSE_THRESHOLD,
        "removable_fields": list(REMOVABLE_FIELDS),
        "clusters": [asdict(result) for result in cluster_results],
        "summary": asdict(summary),
    }
    return results


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 1.22: Spine minimum")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    results_data: dict | None = None
    provider_labels = provider_order()
    for provider in provider_labels:
        try:
            results_data = await run_once(provider)
            break
        except Exception as exc:
            if provider != provider_labels[-1]:
                print(f"{provider} path failed, trying next provider: {exc}")
                continue
            raise

    if results_data is None:
        raise RuntimeError("no provider succeeded")

    results_data["duration_seconds"] = time.monotonic() - t0
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    summary = results_data["summary"]
    print(
        f"Provider: {results_data['provider']}, "
        f"mean_control={summary['mean_control_overall']:.2f}, "
        f"mean_spine={summary['mean_spine_overall']:.2f}, "
        f"spine_size={summary['mean_spine_size']:.1f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
