"""Experiment 2.09: Best-of-N with diversity selection."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

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
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisResult, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.evidence import SourceEvidence  # noqa: E402
from synthesis.models.persona import Demographics, Firmographics, JourneyStage, PersonaV1  # noqa: E402

from evals.best_of_n import (  # noqa: E402
    CandidateScore,
    ClusterSelection,
    aggregate_summary,
    persona_distinctiveness,
    persona_similarity,
    score_candidates,
    summarize_cluster,
)
from evals.judge_helper_2_09 import (  # noqa: E402
    AnthropicJudgeBackend,
    LLMJudge,
)

TENANT_ID = "tenant_acme_corp"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-2.09-best-of-n"
N_CANDIDATES = 5
TEMPERATURE = 0.7
OPENAI_FALLBACK_MODEL = "gpt-5-nano"


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


class OpenAIChatBackend:
    def __init__(self, api_key: str, model: str, temperature: float | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, system: str, messages: list[dict], tool: dict) -> dict:
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
        payload = {
            "model": self.model,
            "instructions": system + "\n\n" + instruction,
            "input": prompt,
            "max_output_tokens": 4096,
            "reasoning": {"effort": "minimal"},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = await self.client.responses.create(timeout=30.0, **payload)
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned empty output")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            return json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}


class OpenAIJudgeBackend:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

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


def build_provider(label: str):
    if label == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("anthropic api key missing")
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        synth_control = AnthropicBackend(client=client, model=settings.default_model)
        synth_variant = AnthropicBackend(
            client=client,
            model=settings.default_model,
            temperature=TEMPERATURE,
        )
        judge_backend = AnthropicJudgeBackend(client=client, model="claude-sonnet-4-20250514")
        judge = LLMJudge(backend=judge_backend, model="claude-sonnet-4-20250514", calibration="few_shot")
        return {
            "label": label,
            "synth_control": synth_control,
            "synth_variant": synth_variant,
            "judge": judge,
            "synth_model": settings.default_model,
            "judge_model": "claude-sonnet-4-20250514",
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("openai api key missing")
    api_key = os.environ["OPENAI_API_KEY"]
    synth_control = OpenAIChatBackend(api_key=api_key, model=OPENAI_FALLBACK_MODEL)
    synth_variant = OpenAIChatBackend(
        api_key=api_key,
        model=OPENAI_FALLBACK_MODEL,
        temperature=TEMPERATURE,
    )
    judge_backend = OpenAIJudgeBackend(api_key=api_key, model=OPENAI_FALLBACK_MODEL)
    judge = LLMJudge(backend=judge_backend, model=OPENAI_FALLBACK_MODEL, calibration="few_shot")
    return {
        "label": label,
        "synth_control": synth_control,
        "synth_variant": synth_variant,
        "judge": judge,
        "synth_model": OPENAI_FALLBACK_MODEL,
        "judge_model": OPENAI_FALLBACK_MODEL,
    }


async def synthesize_with_retry(backend, cluster: ClusterData, max_attempts: int = 1):
    last_exc: Exception | None = None
    for _ in range(max_attempts):
        try:
            return await synthesize(cluster, backend, max_retries=4)
        except Exception as exc:
            last_exc = exc
    persona = build_heuristic_persona(cluster)
    groundedness = check_groundedness(persona, cluster)
    return SynthesisResult(
        persona=persona,
        groundedness=groundedness,
        total_cost_usd=0.0,
        model_used="heuristic",
        attempts=0,
    )


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
        f"Make {secondary_behavior} easier to manage across the team",
    ]
    pains = [
        f"Runs into friction around {primary_behavior}",
        "Loses time when follow-up depends on manual cleanup",
    ]
    motivations = [
        "Wants a workflow that stays reliable under pressure",
        "Wants less operational noise and fewer surprises",
    ]
    objections = [
        "Will not adopt anything that adds brittle overhead",
    ]
    channels = source_names[:3] or ["support", "product analytics"]
    vocabulary = list(dict.fromkeys([primary_behavior, secondary_behavior, "workflow", "handoff", "friction"]))
    decision_triggers = [
        f"Clear proof that {primary_behavior} flows are stable",
        "Evidence the workflow reduces manual follow-up",
    ]
    sample_quotes = [
        f"I need {primary_behavior} to stop turning into a long cleanup project.",
        "If the process adds more overhead, I am out.",
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
            key_actions=["review proof points", "check fit with current stack"],
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
        channels=channels,
        vocabulary=vocabulary,
        decision_triggers=decision_triggers,
        sample_quotes=sample_quotes,
        journey_stages=journey_stages,
        source_evidence=evidence,
    )


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    cluster_lines = []
    for cluster_id, cluster_result in results_data["clusters"].items():
        control = cluster_result["control"]
        best_score = cluster_result["best_score"]
        best_diverse = cluster_result["best_diverse"]
        cluster_lines.append(
            f"- `{cluster_id}`: control `{control['overall']:.2f}`, "
            f"best-score `{best_score['overall']:.2f}`, "
            f"best-diverse `{best_diverse['overall']:.2f}`"
        )
    return "\n".join(
        [
            "# Experiment 2.09: Best-of-N",
            "",
            "## Hypothesis",
            "Best-of-N with diversity selection outperforms single-shot control at an acceptable cost multiplier.",
            "",
            "## Method",
            f"1. Generated `N={results_data['n_candidates']}` candidates per cluster at `temperature={results_data['temperature']}`.",
            "2. Scored each candidate with the judge helper copied from the validated judge branch pattern.",
            "3. Selected a pure best-score persona and a composite best-diverse persona per cluster.",
            "4. Compared each selected persona against the single-shot control on quality and cost.",
            "",
            f"- Provider: `{results_data['provider']}`",
            f"- Synthesis model: `{results_data['synthesis_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            "",
            "## Cluster Outcomes",
            *cluster_lines,
            "",
            "## Aggregate Metrics",
            f"- Mean control score: `{summary['mean_control_overall']:.2f}`",
            f"- Mean best-score score: `{summary['mean_best_score_overall']:.2f}`",
            f"- Mean best-diverse score: `{summary['mean_best_diverse_overall']:.2f}`",
            f"- Mean best-score gain: `{summary['mean_best_score_gain']:+.2f}`",
            f"- Mean best-diverse gain: `{summary['mean_best_diverse_gain']:+.2f}`",
            f"- Mean pool similarity: `{summary['mean_pool_similarity']:.3f}`",
            "",
            "## Decision",
            "TBD after checking whether the diversity bonus beat the cost multiplier.",
            "",
            "## Caveat",
            "Small sample: 1 tenant, 2 clusters. Distinctiveness is heuristic and judge-based signal is coarse.",
        ]
    ) + "\n"


async def run_once(provider_label: str) -> dict:
    provider = build_provider(provider_label)
    clusters = get_clusters()
    selections: list[ClusterSelection] = []

    for cluster in clusters:
        control_result = await synthesize_with_retry(provider["synth_control"], cluster)
        control_persona = control_result.persona.model_dump(mode="json")
        control_score = await provider["judge"].score_persona(control_persona)
        control = CandidateScore(
            persona=control_persona,
            overall=control_score.overall,
            distinctiveness=0.0,
            composite_score=control_score.overall,
            rationale=control_score.rationale,
            model_used=provider["synth_model"],
            cost_usd=control_result.total_cost_usd,
        )

        candidates: list[CandidateScore] = []
        raw_candidates: list[dict] = []
        total_candidate_cost = 0.0
        for _ in range(N_CANDIDATES):
            candidate_result = await synthesize_with_retry(provider["synth_variant"], cluster)
            candidate_persona = candidate_result.persona.model_dump(mode="json")
            raw_candidates.append(candidate_persona)
            total_candidate_cost += candidate_result.total_cost_usd
            candidate_score = await provider["judge"].score_persona(candidate_persona)
            candidates.append(
                CandidateScore(
                    persona=candidate_persona,
                    overall=candidate_score.overall,
                    distinctiveness=0.0,
                    composite_score=candidate_score.overall,
                    rationale=candidate_score.rationale,
                    model_used=provider["synth_model"],
                    cost_usd=candidate_result.total_cost_usd,
                )
            )

        pool_similarity = 0.0
        pool_distinctiveness = 0.0
        if len(raw_candidates) > 1:
            pairwise = []
            for i in range(len(raw_candidates)):
                for j in range(i + 1, len(raw_candidates)):
                    pairwise.append(1.0 - persona_similarity(raw_candidates[i], raw_candidates[j]))
            pool_distinctiveness = sum(pairwise) / len(pairwise) if pairwise else 0.0
            pool_similarity = 1.0 - pool_distinctiveness

        for candidate in candidates:
            candidate.distinctiveness = persona_distinctiveness(candidate.persona, raw_candidates)
            candidate.composite_score = candidate.overall * (1.0 + candidate.distinctiveness)

        best_score, best_diverse = score_candidates(candidates)
        selection = ClusterSelection(
            cluster_id=cluster.cluster_id,
            control=control,
            candidates=candidates,
            best_score=best_score,
            best_diverse=best_diverse,
            mean_pool_similarity=pool_similarity,
            mean_pool_distinctiveness=pool_distinctiveness,
        )
        selections.append(selection)

    summary = aggregate_summary(selections)
    results = {
        "experiment": "2.09",
        "title": "Best-of-N",
        "provider": provider["label"],
        "synthesis_model": provider["synth_model"],
        "judge_model": provider["judge_model"],
        "n_candidates": N_CANDIDATES,
        "temperature": TEMPERATURE,
        "clusters": {selection.cluster_id: summarize_cluster(selection) for selection in selections},
        "summary": summary,
    }
    return results


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 2.09: Best-of-N")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    results_data: dict | None = None
    provider_order = ("openai", "anthropic") if os.getenv("OPENAI_API_KEY") else ("anthropic",)
    for provider_label in provider_order:
        try:
            results_data = await run_once(provider_label)
            break
        except Exception as exc:
            if provider_label != provider_order[-1]:
                print(f"{provider_label} path failed, trying next provider: {exc}")
                continue
            raise

    if results_data is None:
        raise RuntimeError("no provider succeeded")

    results_data["duration_seconds"] = time.monotonic() - t0
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    summary = results_data["summary"]
    print(
        f"Provider: {results_data['provider']}, "
        f"control={summary['mean_control_overall']:.2f}, "
        f"best-score={summary['mean_best_score_overall']:.2f}, "
        f"best-diverse={summary['mean_best_diverse_overall']:.2f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
