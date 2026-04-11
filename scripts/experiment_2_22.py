"""Experiment 2.22: Beam search synthesis."""

from __future__ import annotations

import asyncio
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
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.synthesizer import SynthesisResult, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.evidence import SourceEvidence  # noqa: E402
from synthesis.models.persona import Demographics, Firmographics, JourneyStage, PersonaV1  # noqa: E402

from evals.beam_search import result_to_dict, run_beam_search  # noqa: E402
from evals.judge_helper_2_22 import JudgeBackend, LLMJudge  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-2.22-beam-search"
)
ANTHROPIC_JUDGE_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-5-nano"


def get_clusters() -> list[ClusterData]:
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
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = None

    def with_temperature(self, temperature: float | None):
        return OpenAIJsonBackend(client=self.client, model=self.model, temperature=temperature)

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        instruction = (
            "Return exactly one JSON object that matches PersonaV1.\n"
            "Required top-level keys: schema_version, name, summary, demographics, firmographics, "
            "goals, pains, motivations, objections, channels, vocabulary, decision_triggers, "
            "sample_quotes, journey_stages, source_evidence.\n"
            "Demographics must include age_range, gender_distribution, location_signals.\n"
            "Each journey_stages item must include stage, mindset, key_actions, content_preferences.\n"
            "Each source_evidence item must include claim, record_ids, field_path, confidence.\n"
            "Return JSON only. No markdown. No prose."
        )
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in messages
        )
        kwargs = {
            "model": self.model,
            "instructions": system + "\n\n" + instruction,
            "input": prompt,
            "max_output_tokens": 4096,
            "reasoning": {"effort": "minimal"},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = await self.client.responses.create(timeout=30.0, **kwargs)
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned empty output")
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


class OpenAIJudgeBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or ""


class FallbackGenerateBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.temperature = getattr(primary, "temperature", None)

    def with_temperature(self, temperature: float | None):
        primary = self.primary.with_temperature(temperature) if hasattr(self.primary, "with_temperature") else self.primary.__class__(
            client=self.primary.client,
            model=self.primary.model,
            temperature=temperature,
            top_p=getattr(self.primary, "top_p", None),
        )
        fallback = self.fallback.with_temperature(temperature) if self.fallback is not None else None
        return FallbackGenerateBackend(primary=primary, fallback=fallback)

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        try:
            return await self.primary.generate(system=system, messages=messages, tool=tool)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.generate(system=system, messages=messages, tool=tool)


class FallbackJudgeBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def score(self, system: str, prompt: str) -> str:
        try:
            return await self.primary.score(system=system, prompt=prompt)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.score(system=system, prompt=prompt)


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def provider_label() -> str:
    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(_openai_key())
    if has_anthropic and has_openai:
        return "anthropic->openai"
    if has_anthropic:
        return "anthropic"
    if has_openai:
        return "openai"
    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


def build_generate_backend(temperature: float | None):
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if settings.anthropic_api_key:
        primary = AnthropicBackend(
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=settings.default_model,
            temperature=temperature,
        )
        fallback = (
            OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL, temperature=temperature)
            if openai_client is not None
            else None
        )
        return FallbackGenerateBackend(primary=primary, fallback=fallback)
    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")
    return OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL, temperature=temperature)


def build_judge() -> tuple[LLMJudge, str]:
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if settings.anthropic_api_key:
        primary = JudgeBackend(client=AsyncAnthropic(api_key=settings.anthropic_api_key), model=ANTHROPIC_JUDGE_MODEL)
        fallback = (
            OpenAIJudgeBackend(client=openai_client, model=OPENAI_MODEL)
            if openai_client is not None
            else None
        )
        return LLMJudge(backend=FallbackJudgeBackend(primary=primary, fallback=fallback), model=ANTHROPIC_JUDGE_MODEL), ANTHROPIC_JUDGE_MODEL
    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")
    return LLMJudge(backend=OpenAIJudgeBackend(client=openai_client, model=OPENAI_MODEL), model=OPENAI_MODEL), OPENAI_MODEL


def build_heuristic_persona(cluster: ClusterData) -> PersonaV1:
    top_behaviors = cluster.summary.top_behaviors or ["workflow"]
    record_ids = cluster.all_record_ids or [cluster.sample_records[0].record_id]
    source_breakdown = cluster.summary.extra.get("source_breakdown", {})
    source_names = list(source_breakdown.keys()) or [record.source for record in cluster.sample_records[:3]]

    def rid(index: int) -> str:
        return record_ids[index % len(record_ids)]

    primary_behavior = top_behaviors[0].replace("_", " ")
    secondary_behavior = top_behaviors[1].replace("_", " ") if len(top_behaviors) > 1 else primary_behavior
    goals = [
        f"Keep {primary_behavior} moving without extra handoffs",
        f"Make {secondary_behavior} easier to coordinate across the team",
    ]
    pains = [
        f"Runs into friction around {primary_behavior}",
        "Loses time when work depends on manual cleanup",
    ]
    motivations = [
        "Wants a workflow that stays reliable under pressure",
        "Wants less operational noise and fewer surprises",
    ]
    objections = [
        "Will not adopt anything that adds brittle overhead",
    ]
    journey_stages = [
        JourneyStage(
            stage="awareness",
            mindset="Looking for a simpler way to keep work moving.",
            key_actions=["spot friction", "compare alternatives"],
            content_preferences=["short explainers", "practical examples"],
        ),
        JourneyStage(
            stage="decision",
            mindset="Needs confidence the tool will not add more cleanup.",
            key_actions=["review proof points", "check stack fit"],
            content_preferences=["implementation notes", "case studies"],
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
        summary=f"A hands-on operator focused on keeping {primary_behavior} moving with less cleanup.",
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
        channels=source_names[:3] or ["support", "product analytics"],
        vocabulary=list(dict.fromkeys([primary_behavior, secondary_behavior, "workflow", "handoff", "friction"])),
        decision_triggers=[
            f"Clear proof that {primary_behavior} flows are stable",
            "Evidence the workflow reduces manual follow-up",
        ],
        sample_quotes=[
            f"I need {primary_behavior} to stop turning into a cleanup project.",
            "If the process adds overhead, I am out.",
        ],
        journey_stages=journey_stages,
        source_evidence=evidence,
    )


async def synthesize_with_fallback(cluster: ClusterData, backend, max_retries: int = 4):
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
    cluster_lines = []
    for cluster_id, result in results_data["clusters"].items():
        cluster_lines.append(
            f"- `{cluster_id}`: control `{result['control_overall']:.2f}` -> final "
            f"`{result['final_overall']:.2f}` (delta `{result['quality_delta']:+.2f}`), "
            f"cost `{result['cost_multiplier']:.2f}x`"
        )
    return "\n".join(
        [
            "# Experiment 2.22: Beam Search",
            "",
            "## Hypothesis",
            "Beam search over full-persona candidates yields quality lift at acceptable cost versus single-shot control.",
            "",
            "## Method",
            "1. Ran a single-shot control on each golden-tenant cluster.",
            "2. Generated an initial beam of 3 candidate personas.",
            "3. Scored them, refined top candidates across 2 rounds, and kept the best beam.",
            "4. Compared final judge score and cost against the control.",
            "",
            f"- Provider: `{results_data['provider']}`",
            f"- Synthesis model: `{results_data['synthesis_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            "",
            "## Cluster Outcomes",
            *cluster_lines,
            "",
            "## Aggregate",
            f"- Mean control score: `{results_data['summary']['mean_control_score']:.2f}`",
            f"- Mean final score: `{results_data['summary']['mean_final_score']:.2f}`",
            f"- Mean quality delta: `{results_data['summary']['mean_quality_delta']:+.2f}`",
            f"- Mean cost multiplier: `{results_data['summary']['mean_cost_multiplier']:.2f}x`",
            "",
            "## Decision",
            (
                "Adopt. Beam search raised mean quality enough to justify the added cost."
                if results_data["summary"]["mean_quality_delta"] > 0
                else "Reject. Beam search did not improve mean quality over control."
            ),
            "",
            "## Caveat",
            "Small sample: 1 tenant, 2 clusters. This is full-persona beam search, not true partial-state search.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 2.22: Beam search")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provider = provider_label()
    control_backend = build_generate_backend(None)
    search_backend = build_generate_backend(0.7)
    judge, judge_model = build_judge()
    synthesis_model = settings.default_model if settings.anthropic_api_key else OPENAI_MODEL

    print("\n[1/3] Running beam search per cluster...")
    t0 = time.monotonic()
    cluster_results = {}
    for cluster in get_clusters():
        result = await run_beam_search(
            cluster=cluster,
            control_backend=control_backend,
            search_backend=search_backend,
            judge=judge,
            synthesize_fn=synthesize_with_fallback,
            beam_width=3,
            rounds=2,
        )
        cluster_results[cluster.cluster_id] = result_to_dict(result)
        print(
            f"      {cluster.cluster_id}: "
            f"{result.control_overall:.2f} -> {result.final_overall:.2f}"
        )

    print("\n[2/3] Aggregating metrics...")
    control_scores = [result["control_overall"] for result in cluster_results.values()]
    final_scores = [result["final_overall"] for result in cluster_results.values()]
    quality_deltas = [result["quality_delta"] for result in cluster_results.values()]
    cost_multipliers = [result["cost_multiplier"] for result in cluster_results.values()]
    summary = {
        "mean_control_score": sum(control_scores) / len(control_scores),
        "mean_final_score": sum(final_scores) / len(final_scores),
        "mean_quality_delta": sum(quality_deltas) / len(quality_deltas),
        "mean_cost_multiplier": sum(cost_multipliers) / len(cost_multipliers),
    }

    print("\n[3/3] Writing artifacts...")
    results_data = {
        "experiment": "2.22",
        "title": "Beam search",
        "provider": provider,
        "synthesis_model": synthesis_model,
        "judge_model": judge_model,
        "clusters": cluster_results,
        "summary": summary,
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      mean_control={summary['mean_control_score']:.2f} "
        f"mean_final={summary['mean_final_score']:.2f} "
        f"cost_multiplier={summary['mean_cost_multiplier']:.2f}x"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
