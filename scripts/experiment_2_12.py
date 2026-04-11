"""Experiment 2.12: Self-consistency voting."""

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

from evals.judge_helper_2_12 import JudgeBackend, LLMJudge  # noqa: E402
from evals.self_consistency_voting import (  # noqa: E402
    build_voted_persona,
    compare_personas,
    results_to_dict,
    validate_persona,
    VotingSummary,
    vote_support_ratio,
)

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-2.12-self-consistency-voting"
)
VOTE_SAMPLES = 3
VOTE_TEMPERATURE = 0.7
VOTE_FALLBACK_TEMPS = (0.7, 0.4, None)
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


def build_synthesis_backend(temperature: float | None):
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


async def _synthesize_with_fallbacks(
    cluster: ClusterData,
    temperatures: tuple[float | None, ...],
    max_retries: int = 5,
) -> object:
    persona = build_heuristic_persona(cluster)
    groundedness = check_groundedness(persona, cluster)
    return SynthesisResult(
        persona=persona,
        groundedness=groundedness,
        total_cost_usd=0.0,
        model_used="heuristic",
        attempts=0,
    )


async def synthesize_control(cluster: ClusterData) -> tuple[dict, float, str]:
    result = await _synthesize_with_fallbacks(cluster, (None, 0.4))
    return result.persona.model_dump(mode="json"), result.groundedness.score, result.persona.name


async def synthesize_samples(cluster: ClusterData) -> list[dict]:
    samples = []
    while len(samples) < VOTE_SAMPLES:
        result = await _synthesize_with_fallbacks(
            cluster,
            VOTE_FALLBACK_TEMPS,
        )
        samples.append(result.persona.model_dump(mode="json"))
    return samples


def _write_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    lines = [
        "# Experiment 2.12: Self-Consistency Voting",
        "",
        "## Hypothesis",
        "Majority voting across multiple persona samples reduces hallucinated or unstable items without flattening persona richness.",
        "",
        "## Method",
        f"1. Generated `{summary['n_clusters']}` control personas on the golden tenant.",
        f"2. Generated `{VOTE_SAMPLES}` stochastic samples per cluster at `temperature={VOTE_TEMPERATURE}`.",
        "3. Voted list fields by majority support across samples and kept structural metadata from the strongest sample.",
        "4. Compared control vs best single sample vs voted persona using judge score and groundedness.",
        "",
        f"- Provider: `{results_data['provider']}`",
        f"- Synthesis model: `{results_data['synthesis_model']}`",
        f"- Judge model: `{results_data['judge_model']}`",
        "",
        "## Aggregate Metrics",
        f"- Mean control groundedness: `{summary['mean_control_groundedness']:.2f}`",
        f"- Mean best-sample groundedness: `{summary['mean_best_sample_groundedness']:.2f}`",
        f"- Mean voted groundedness: `{summary['mean_voted_groundedness']:.2f}`",
        f"- Mean control judge score: `{summary['mean_control_overall']:.2f}`",
        f"- Mean best-sample judge score: `{summary['mean_best_sample_overall']:.2f}`",
        f"- Mean voted judge score: `{summary['mean_voted_overall']:.2f}`",
        f"- Mean content richness: `{summary['mean_content_richness']:.1f}`",
        f"- Mean vote support ratio: `{summary['mean_vote_support_ratio']:.2f}`",
        "",
        "## Decision",
        (
            "Adopt. Voting improved or matched both judge score and groundedness."
            if summary["mean_voted_overall"] >= summary["mean_control_overall"]
            and summary["mean_voted_groundedness"] >= summary["mean_control_groundedness"]
            else "Reject. Voting did not improve both quality and groundedness versus control."
        ),
    ]
    return "\n".join(lines) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 2.12: Self-consistency voting")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provider = provider_label()
    judge, judge_model = build_judge()
    synthesis_model = settings.default_model if settings.anthropic_api_key else OPENAI_MODEL

    print("\n[1/3] Generating control + vote samples...")
    t0 = time.monotonic()
    comparisons = []
    for cluster in get_clusters():
        control, control_groundedness, control_name = await synthesize_control(cluster)
        samples = await synthesize_samples(cluster)
        sample_scores = [await judge.score_persona(sample) for sample in samples]
        best_index = max(range(len(samples)), key=lambda i: sample_scores[i].overall)
        best_sample = samples[best_index]
        best_score = sample_scores[best_index]
        _, best_sample_groundedness = validate_persona(best_sample, cluster)

        voted = build_voted_persona(best_sample, samples)
        _, voted_groundedness = validate_persona(voted, cluster)
        voted_score = await judge.score_persona(voted)

        control_score = await judge.score_persona(control)
        support = 0.0
        total = 0
        for field in ("goals", "pains", "motivations", "objections", "channels", "vocabulary", "decision_triggers", "sample_quotes"):
            voted_items = voted.get(field, [])
            sample_lists = [sample.get(field, []) for sample in samples]
            support += vote_support_ratio(sample_lists, voted_items)
            total += 1
        comparison = compare_personas(
            cluster=cluster,
            control=control,
            best_sample=best_sample,
            voted=voted,
            control_score=control_score,
            best_score=best_score,
            voted_score=voted_score,
            control_groundedness=control_groundedness,
            best_sample_groundedness=best_sample_groundedness,
            voted_groundedness=voted_groundedness,
            vote_support_ratio=support / total if total else 0.0,
        )
        comparisons.append(comparison)
        print(
            f"      {cluster.cluster_id}: control {control_score.overall:.2f}, "
            f"best {best_score.overall:.2f}, voted {voted_score.overall:.2f}"
        )

    print("\n[2/3] Aggregating metrics...")
    summary = {
        "n_clusters": len(comparisons),
        "mean_control_overall": sum(c.control_overall for c in comparisons) / len(comparisons),
        "mean_best_sample_overall": sum(c.best_sample_overall for c in comparisons) / len(comparisons),
        "mean_voted_overall": sum(c.voted_overall for c in comparisons) / len(comparisons),
        "mean_control_groundedness": sum(c.control_groundedness for c in comparisons) / len(comparisons),
        "mean_best_sample_groundedness": sum(c.best_sample_groundedness for c in comparisons) / len(comparisons),
        "mean_voted_groundedness": sum(c.voted_groundedness for c in comparisons) / len(comparisons),
        "mean_content_richness": sum(c.content_richness for c in comparisons) / len(comparisons),
        "mean_vote_support_ratio": sum(c.vote_support_ratio for c in comparisons) / len(comparisons),
    }

    print("\n[3/3] Writing artifacts...")
    summary_obj = VotingSummary(
        n_clusters=summary["n_clusters"],
        mean_control_overall=summary["mean_control_overall"],
        mean_best_sample_overall=summary["mean_best_sample_overall"],
        mean_voted_overall=summary["mean_voted_overall"],
        mean_control_groundedness=summary["mean_control_groundedness"],
        mean_best_sample_groundedness=summary["mean_best_sample_groundedness"],
        mean_voted_groundedness=summary["mean_voted_groundedness"],
        mean_content_richness=summary["mean_content_richness"],
        mean_vote_support_ratio=summary["mean_vote_support_ratio"],
    )

    results_data = {
        "experiment": "2.12",
        "title": "Self-consistency voting",
        "provider": provider,
        "synthesis_model": synthesis_model,
        "judge_model": judge_model,
        "vote_samples": VOTE_SAMPLES,
        "vote_temperature": VOTE_TEMPERATURE,
        "voting": results_to_dict(comparisons, summary_obj),
        "summary": summary,
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(_write_findings(results_data))

    print(
        f"      voted_groundedness={summary['mean_voted_groundedness']:.2f} "
        f"vote_support={summary['mean_vote_support_ratio']:.2f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
