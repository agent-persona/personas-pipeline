"""Experiment 6.11: outlier persona helper utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from segmentation.engine.clusterer import cluster_users, jaccard_similarity
from segmentation.engine.featurizer import featurize_records
from segmentation.engine.summarizer import build_cluster_data
from segmentation.models.features import UserFeatures
from segmentation.models.record import RawRecord
from synthesis.config import settings
from synthesis.engine.model_backend import AnthropicBackend, LLMResult
from synthesis.engine.synthesizer import synthesize
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

ANTHROPIC_SYNTHESIS_MODEL = settings.default_model
ANTHROPIC_JUDGE_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-5-nano"


@dataclass
class OutlierSelection:
    selection_reason: str
    selected_user_ids: list[str]
    selected_record_ids: list[str]
    selected_behaviors: list[str]
    cluster_similarities: list[dict[str, object]]


@dataclass
class CoverageSummary:
    total_records: int
    total_behaviors: int
    baseline_record_coverage: float
    forced_record_coverage: float
    baseline_behavior_coverage: float
    forced_behavior_coverage: float
    baseline_unique_records: int
    forced_unique_records: int
    baseline_unique_behaviors: int
    forced_unique_behaviors: int
    added_record_ids: list[str]
    added_behaviors: list[str]


@dataclass
class PersonaScore:
    persona_name: str
    overall: float
    dimensions: dict[str, float]
    rationale: str
    model: str


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = None

    def with_temperature(self, temperature: float | None):
        return OpenAIJsonBackend(client=self.client, model=self.model, temperature=temperature)

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        prompt = "\n\n".join(f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages)
        instructions = (
            system
            + "\n\nReturn exactly one JSON object that matches PersonaV1.\n"
            + "Required top-level keys: schema_version, name, summary, demographics, firmographics, "
            + "goals, pains, motivations, objections, channels, vocabulary, decision_triggers, "
            + "sample_quotes, journey_stages, source_evidence.\n"
            + "Demographics must include age_range, gender_distribution, location_signals.\n"
            + "Each journey_stages item must include stage, mindset, key_actions, content_preferences.\n"
            + "Each source_evidence item must include claim, record_ids, field_path, confidence.\n"
            + "Return JSON only. No markdown. No prose."
        )
        kwargs: dict[str, object] = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": 4096,
            "reasoning": {"effort": "minimal"},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = await self.client.responses.create(**kwargs)
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned empty output")
        tool_input = _extract_json(text)
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model,
        )


class OpenAIJudgeBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            instructions=system + "\n\nReturn JSON only.",
            input=prompt,
            max_output_tokens=512,
            reasoning={"effort": "minimal"},
        )
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI judge returned empty output")
        return text


class FallbackGenerateBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    def with_temperature(self, temperature: float | None):
        primary = self.primary.with_temperature(temperature) if hasattr(self.primary, "with_temperature") else self.primary
        fallback = self.fallback.with_temperature(temperature) if self.fallback is not None and hasattr(self.fallback, "with_temperature") else self.fallback
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


def build_backends() -> tuple[object, object, str]:
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    if settings.anthropic_api_key:
        anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        synth_primary = AnthropicBackend(
            client=anthropic_client,
            model=ANTHROPIC_SYNTHESIS_MODEL,
        )
        judge_primary = JudgeBackend(
            client=anthropic_client,
            model=ANTHROPIC_JUDGE_MODEL,
        )
        synth_fallback = (
            OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL)
            if openai_client is not None
            else None
        )
        judge_fallback = (
            OpenAIJudgeBackend(client=openai_client, model=OPENAI_MODEL)
            if openai_client is not None
            else None
        )
        return (
            FallbackGenerateBackend(synth_primary, synth_fallback),
            FallbackJudgeBackend(judge_primary, judge_fallback),
            "anthropic->openai" if openai_client is not None else "anthropic",
        )

    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")

    return (
        OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL),
        OpenAIJudgeBackend(client=openai_client, model=OPENAI_MODEL),
        "openai",
    )


class JudgeBackend:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


_JUDGE_SYSTEM_PROMPT = """\
You are an expert persona evaluator.

Score a customer persona on a 1-5 scale for:
- grounded
- distinctive
- coherent
- actionable
- voice_fidelity

Return JSON only:
{
  "grounded": <1-5>,
  "distinctive": <1-5>,
  "coherent": <1-5>,
  "actionable": <1-5>,
  "voice_fidelity": <1-5>,
  "overall": <1-5>,
  "rationale": "short justification"
}
"""


class LLMJudge:
    def __init__(self, backend, model: str) -> None:
        self.backend = backend
        self.model = model

    async def score_persona(self, persona: dict) -> PersonaScore:
        prompt = "Score this persona.\n\nPERSONA:\n" + json.dumps(persona, indent=2, default=str)
        response = await self.backend.score(_JUDGE_SYSTEM_PROMPT, prompt)
        data = _extract_json(response)
        dimensions = {k: float(data.get(k, float("nan"))) for k in ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")}
        return PersonaScore(
            persona_name=str(persona.get("name", "unknown")),
            overall=float(data.get("overall", float("nan"))),
            dimensions=dimensions,
            rationale=str(data.get("rationale", "")),
            model=self.model,
        )


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise
        data = json.loads(cleaned[start : end + 1])
    if isinstance(data.get("source_evidence"), list):
        return data
    return data


def get_clusters(records: list[RawRecord], tenant_industry: str, tenant_product: str) -> tuple[list[UserFeatures], list[list[UserFeatures]]]:
    features = featurize_records(records)
    clusters = cluster_users(features, threshold=0.15, min_cluster_size=2)
    return features, clusters


def select_outlier_users(features: list[UserFeatures], clusters: list[list[UserFeatures]]) -> OutlierSelection:
    if not clusters:
        raise RuntimeError("No clusters available for outlier selection")

    cluster_similarities: list[dict[str, object]] = []
    selected: list[UserFeatures] = []
    selected_reason_parts: list[str] = []

    for idx, cluster in enumerate(clusters):
        centroid = set().union(*(member.behaviors for member in cluster))
        scored = [
            (member, jaccard_similarity(member.behaviors, centroid))
            for member in cluster
        ]
        scored.sort(key=lambda item: (item[1], len(item[0].record_ids)))
        chosen, sim = scored[0]
        selected.append(chosen)
        cluster_similarities.append(
            {
                "cluster_index": idx,
                "user_id": chosen.user_id,
                "similarity_to_centroid": sim,
                "record_ids": list(chosen.record_ids),
                "behaviors": sorted(chosen.behaviors),
            }
        )
        selected_reason_parts.append(f"{chosen.user_id}@{sim:.3f}")

    selected_record_ids = sorted({rid for member in selected for rid in member.record_ids})
    selected_behaviors = sorted({behavior for member in selected for behavior in member.behaviors})
    return OutlierSelection(
        selection_reason="lowest-similarity user from each natural cluster",
        selected_user_ids=[member.user_id for member in selected],
        selected_record_ids=selected_record_ids,
        selected_behaviors=selected_behaviors,
        cluster_similarities=cluster_similarities,
    )


def build_forced_outlier_cluster(
    selected_users: list[UserFeatures],
    all_records: list[RawRecord],
    tenant_id: str,
    tenant_industry: str,
    tenant_product: str,
    existing_persona_names: list[str],
    selection: OutlierSelection,
) -> ClusterData:
    cluster_dict = build_cluster_data(
        cluster_users=selected_users,
        all_records=all_records,
        tenant_id=tenant_id,
        tenant_industry=tenant_industry,
        tenant_product=tenant_product,
        existing_persona_names=existing_persona_names,
        sample_size=max(4, len(selection.selected_record_ids)),
    )
    cluster_dict["summary"]["extra"]["forced_outlier"] = True
    cluster_dict["summary"]["extra"]["selection_reason"] = selection.selection_reason
    cluster_dict["summary"]["extra"]["selected_user_ids"] = selection.selected_user_ids
    cluster_dict["summary"]["extra"]["selected_record_ids"] = selection.selected_record_ids
    cluster_dict["summary"]["extra"]["selected_behaviors"] = selection.selected_behaviors
    cluster_dict["summary"]["extra"]["cluster_similarities"] = selection.cluster_similarities
    return ClusterData.model_validate(cluster_dict)


def compute_coverage(
    baseline_clusters: list[ClusterData],
    forced_outlier: ClusterData,
    records: list[RawRecord],
) -> CoverageSummary:
    record_by_id = {record.record_id: record for record in records}
    all_record_ids = set(record_by_id)
    all_behaviors = {behavior for record in records for behavior in record.behaviors}

    baseline_record_ids = {
        sample.record_id
        for cluster in baseline_clusters
        for sample in cluster.sample_records
    }
    forced_record_ids = baseline_record_ids | {sample.record_id for sample in forced_outlier.sample_records}

    def behaviors_for(record_ids: set[str]) -> set[str]:
        return {
            behavior
            for record_id in record_ids
            for behavior in record_by_id[record_id].behaviors
        }

    baseline_behaviors = behaviors_for(baseline_record_ids)
    forced_behaviors = behaviors_for(forced_record_ids)

    return CoverageSummary(
        total_records=len(all_record_ids),
        total_behaviors=len(all_behaviors),
        baseline_record_coverage=len(baseline_record_ids) / len(all_record_ids),
        forced_record_coverage=len(forced_record_ids) / len(all_record_ids),
        baseline_behavior_coverage=len(baseline_behaviors) / len(all_behaviors),
        forced_behavior_coverage=len(forced_behaviors) / len(all_behaviors),
        baseline_unique_records=len(baseline_record_ids),
        forced_unique_records=len(forced_record_ids),
        baseline_unique_behaviors=len(baseline_behaviors),
        forced_unique_behaviors=len(forced_behaviors),
        added_record_ids=sorted(forced_record_ids - baseline_record_ids),
        added_behaviors=sorted(forced_behaviors - baseline_behaviors),
    )
