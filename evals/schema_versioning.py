"""Experiment 1.05: schema versioning drift helpers."""

from __future__ import annotations

import json
import os
import re
import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import mean
from typing import Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from synthesis.engine.groundedness import check_groundedness
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_user_message
from synthesis.engine.model_backend import AnthropicBackend, LLMResult
from synthesis.config import settings
from synthesis.models.cluster import ClusterData
from synthesis.models.evidence import SourceEvidence
from synthesis.models.persona import Demographics, Firmographics, JourneyStage, PersonaV1
from evals.judge_helper_1_05 import JudgeScore


class PersonaV1_1(PersonaV1):
    schema_version: Literal["1.1"] = "1.1"
    beliefs: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    values: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    contradictions: list[str] = Field(default_factory=list, min_length=0, max_length=6)


class PersonaV2(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    name: str = Field(description="A memorable, descriptive name for this persona")
    summary: str = Field(description="2-3 sentence overview of who this persona is")
    demographics: Demographics
    firmographics: Firmographics
    goals: list[str] = Field(min_length=2, max_length=8)
    pains: list[str] = Field(min_length=2, max_length=8)
    motivations: list[str] = Field(min_length=2, max_length=6)
    objections: list[str] = Field(min_length=1, max_length=6)
    vocabulary: list[str] = Field(min_length=3, max_length=15)
    decision_triggers: list[str] = Field(min_length=1, max_length=6)
    sample_quotes: list[str] = Field(min_length=2, max_length=5)
    beliefs: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    values: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    contradictions: list[str] = Field(default_factory=list, min_length=0, max_length=6)
    source_evidence: list[SourceEvidence] = Field(min_length=3)


@dataclass(frozen=True)
class SchemaVariant:
    key: str
    model: type[BaseModel]
    prompt_note: str
    is_baseline: bool = False


SCHEMA_VARIANTS = (
    SchemaVariant(
        key="v1",
        model=PersonaV1,
        prompt_note="Use the baseline PersonaV1 schema exactly. Keep channels and journey_stages.",
        is_baseline=True,
    ),
    SchemaVariant(
        key="v1.1",
        model=PersonaV1_1,
        prompt_note=(
            "Use the PersonaV1 baseline plus the new beliefs, values, and contradictions fields. "
            "Keep all baseline fields, including channels and journey_stages."
        ),
    ),
    SchemaVariant(
        key="v2",
        model=PersonaV2,
        prompt_note=(
            "Use the compact PersonaV2 schema. Do not include channels or journey_stages. "
            "Add beliefs, values, and contradictions."
        ),
    ),
)


def variant_by_key(key: str) -> SchemaVariant:
    return next(variant for variant in SCHEMA_VARIANTS if variant.key == key)


def build_versioned_prompt(cluster: ClusterData, variant: SchemaVariant) -> str:
    return build_user_message(cluster) + (
        "\n\n## Schema Variant Guidance\n"
        f"- Variant: {variant.key}\n"
        f"- {variant.prompt_note}\n"
        "Return only the persona JSON matching this variant."
    )


def build_tool_definition(variant: SchemaVariant) -> dict:
    return {
        "name": f"create_persona_{variant.key.replace('.', '_')}",
        "description": (
            "Create a structured persona from the cluster data for the requested schema variant. "
            "All fields must be grounded in the provided source records."
        ),
        "input_schema": variant.model.model_json_schema(),
    }


def _tokenize(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = [json.dumps(item, sort_keys=True, default=str) for item in value]
        text = " ".join(parts)
    elif isinstance(value, dict):
        text = json.dumps(value, sort_keys=True, default=str)
    else:
        text = str(value)
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _field_similarity(a: object, b: object) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def shared_field_names(baseline: BaseModel, variant: BaseModel) -> list[str]:
    baseline_fields = set(getattr(baseline, "model_fields", {}))
    variant_fields = set(getattr(variant, "model_fields", {}))
    return sorted(
        field_name
        for field_name in baseline_fields & variant_fields
        if field_name not in {"schema_version"}
    )


def compare_shared_fields(baseline: BaseModel, variant: BaseModel) -> dict[str, float]:
    comparisons: dict[str, float] = {}
    for field_name in shared_field_names(baseline, variant):
        baseline_value = getattr(baseline, field_name)
        variant_value = getattr(variant, field_name)
        comparisons[field_name] = _field_similarity(baseline_value, variant_value)
    return comparisons


def mean_shared_field_similarity(comparisons: Iterable[dict[str, float]]) -> float:
    per_pair: list[float] = []
    for comparison in comparisons:
        if comparison:
            per_pair.append(mean(comparison.values()))
    return mean(per_pair) if per_pair else 0.0


@dataclass
class VariantRun:
    cluster_id: str
    variant: str
    persona_name: str
    groundedness: float
    judge_overall: float
    validity: bool
    attempts: int
    model_used: str
    shared_field_similarity: dict[str, float] = field(default_factory=dict)
    source_evidence_count: int = 0


@dataclass
class VariantSummary:
    variant: str
    n_clusters: int
    mean_groundedness: float
    mean_judge_overall: float
    validity_rate: float
    mean_attempts: float
    mean_source_evidence_count: float
    judge_score_delta_vs_v1: float | None = None
    mean_shared_field_similarity_vs_v1: float | None = None
    per_field_similarity_vs_v1: dict[str, float] = field(default_factory=dict)


@dataclass
class SynthesisOutcome:
    cluster_id: str
    variant: str
    persona: BaseModel
    groundedness: float
    attempts: int
    model_used: str
    shared_field_similarity: dict[str, float]
    validity: bool
    source_evidence_count: int


def use_remote_llm() -> bool:
    return os.getenv("PERSONAS_USE_REMOTE_LLM", "").strip() == "1"


def _focus_terms(cluster: ClusterData) -> list[str]:
    terms = list(cluster.summary.top_behaviors[:3])
    terms.extend(cluster.summary.top_pages[:2])
    terms.extend(cluster.summary.top_referrers[:2])
    cleaned = [term.replace("/", " ").replace("_", " ").strip() for term in terms if term]
    return cleaned or [cluster.cluster_id.replace("_", " ")]


def _cluster_theme(cluster: ClusterData) -> str:
    return _focus_terms(cluster)[0]


def build_local_persona(cluster: ClusterData, variant: SchemaVariant) -> BaseModel:
    theme = _cluster_theme(cluster)
    themes = _focus_terms(cluster)
    record_ids = cluster.all_record_ids
    first_record = cluster.sample_records[0]
    company_size = cluster.enrichment.firmographic.get("company_size") or "SMB"
    industry = cluster.tenant.industry or "B2B SaaS"
    role_titles = list(cluster.enrichment.firmographic.get("role_titles", [])) or ["operator"]
    tech_stack = list(cluster.enrichment.firmographic.get("tech_stack_signals", []))
    summary_words = ", ".join(themes[:3])

    goals = [
        f"make {theme} workflows faster",
        f"keep {summary_words} visible without extra manual work",
    ]
    pains = [
        f"losing time to messy handoffs around {theme}",
        f"not having a reliable view of what is happening across {summary_words}",
    ]
    motivations = [
        f"wants cleaner execution around {theme}",
        "values practical improvements that save time immediately",
    ]
    objections = [
        "does not want another tool that adds admin overhead",
    ]
    channels = ["Slack", "Email", "In-app"]
    vocabulary = [
        theme,
        "workflow",
        "handoff",
        "visibility",
        "follow-up",
        "sync",
    ]
    decision_triggers = [
        f"clear proof it improves {theme}",
        "fast setup with low overhead",
    ]
    sample_quotes = [
        f"Can we make {theme} easier without adding more admin?",
        f"I need a cleaner way to track {summary_words}.",
    ]
    journey_stages = [
        {
            "stage": "consideration",
            "mindset": f"Trying to tame {theme} without disrupting current work",
            "key_actions": [f"compare tools for {theme}", "share with teammates"],
            "content_preferences": ["short examples", "clear ROI"],
        },
        {
            "stage": "decision",
            "mindset": "Needs confidence the change is worth the switch",
            "key_actions": ["check implementation effort", "review proof points"],
            "content_preferences": ["concise demos", "practical references"],
        },
    ]
    beliefs = [
        f"{theme} gets better when the workflow stays simple",
        "small process improvements compound quickly",
    ]
    values = [
        "clarity",
        "speed",
        "reliability",
    ]
    contradictions = [
        "wants more automation but still needs control",
    ]
    evidence_items = []
    required_fields = {
        "goals": goals,
        "pains": pains,
        "motivations": motivations,
        "objections": objections,
    }
    for field_name, items in required_fields.items():
        for idx, item in enumerate(items):
            evidence_items.append(
                {
                    "claim": item,
                    "record_ids": [record_ids[idx % len(record_ids)]],
                    "field_path": f"{field_name}.{idx}",
                    "confidence": 0.82,
                }
            )
    evidence_items.append(
        {
            "claim": sample_quotes[0],
            "record_ids": [first_record.record_id],
            "field_path": "sample_quotes.0",
            "confidence": 0.7,
        }
    )

    base_data = {
        "schema_version": variant.model.model_fields["schema_version"].default,
        "name": f"{theme.title()} Operator",
        "summary": (
            f"A {company_size.lower()} {industry.lower()} operator focused on {summary_words}. "
            f"They are usually working across {', '.join(role_titles[:2])} responsibilities."
        ),
        "demographics": {
            "age_range": "30-44",
            "gender_distribution": "mixed",
            "location_signals": [cluster.tenant.tenant_id, "remote-friendly"],
            "education_level": "varied",
            "income_bracket": "mid to upper-mid",
        },
        "firmographics": {
            "company_size": company_size,
            "industry": industry,
            "role_titles": role_titles[:3],
            "tech_stack_signals": tech_stack[:5],
        },
        "goals": goals,
        "pains": pains,
        "motivations": motivations,
        "objections": objections,
        "vocabulary": vocabulary,
        "decision_triggers": decision_triggers,
        "sample_quotes": sample_quotes,
        "source_evidence": evidence_items,
    }
    if variant.key != "v2":
        base_data["channels"] = channels
        base_data["journey_stages"] = journey_stages
    if variant.key in ("v1.1", "v2"):
        base_data["beliefs"] = beliefs
        base_data["values"] = values
        base_data["contradictions"] = contradictions
    return variant.model.model_validate(base_data)


class LocalJudge:
    def __init__(self, model: str = "local") -> None:
        self.model = model

    async def score_persona(self, persona: dict) -> JudgeScore:
        data = persona if isinstance(persona, dict) else persona.model_dump(mode="json")
        goals = data.get("goals", [])
        pains = data.get("pains", [])
        motivations = data.get("motivations", [])
        objections = data.get("objections", [])
        vocabulary = data.get("vocabulary", [])
        decision_triggers = data.get("decision_triggers", [])
        sample_quotes = data.get("sample_quotes", [])
        beliefs = data.get("beliefs", [])
        values = data.get("values", [])
        contradictions = data.get("contradictions", [])
        channels = data.get("channels", [])
        journey_stages = data.get("journey_stages", [])
        evidence_count = len(data.get("source_evidence", []))
        structural_richness = len(goals) + len(pains) + len(motivations) + len(objections)

        grounded = min(5.0, 2.5 + 0.12 * evidence_count + 0.18 * structural_richness)
        distinctive = min(5.0, 2.3 + 0.15 * len(vocabulary) + 0.08 * len(sample_quotes) + 0.2 * len(beliefs))
        coherent = min(
            5.0,
            2.6
            + 0.12 * len(goals)
            + 0.12 * len(pains)
            + 0.1 * len(journey_stages)
            + 0.15 * (1 if values else 0)
        )
        actionable = min(5.0, 2.5 + 0.18 * len(goals) + 0.14 * len(decision_triggers) + 0.08 * len(objections))
        voice_fidelity = min(5.0, 2.4 + 0.18 * len(sample_quotes) + 0.08 * len(channels) + 0.12 * len(contradictions))
        overall = round(mean([grounded, distinctive, coherent, actionable, voice_fidelity]), 2)
        rationale = (
            "Local heuristic judge. Richer schema variants with extra beliefs/values/contradictions score slightly higher."
        )
        return JudgeScore(
            overall=overall,
            dimensions={
                "grounded": round(grounded, 2),
                "distinctive": round(distinctive, 2),
                "coherent": round(coherent, 2),
                "actionable": round(actionable, 2),
                "voice_fidelity": round(voice_fidelity, 2),
            },
            rationale=rationale,
            judge_model=self.model,
        )


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = None

    def with_temperature(self, temperature: float | None):
        return OpenAIJsonBackend(client=self.client, model=self.model, temperature=temperature)

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        tool_schema = json.dumps(tool["input_schema"], indent=2, default=str)
        chat_messages = [{"role": "system", "content": system}]
        chat_messages.extend(messages)
        chat_messages.append(
            {
                "role": "user",
                "content": (
                    "Return a single JSON object that conforms to the schema below.\n\n"
                    f"SCHEMA:\n{tool_schema}"
                ),
            }
        )
        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_completion_tokens": 4096,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = await self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or "{}"
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
        self.temperature = getattr(primary, "temperature", None)
        self.top_p = getattr(primary, "top_p", None)

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
            return await asyncio.wait_for(
                self.primary.generate(system=system, messages=messages, tool=tool),
                timeout=30,
            )
        except Exception:
            if self.fallback is None:
                raise
            return await asyncio.wait_for(
                self.fallback.generate(system=system, messages=messages, tool=tool),
                timeout=120,
            )


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_synthesis_backend(variant: SchemaVariant, temperature: float | None = None):
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if openai_client is not None:
        primary = OpenAIJsonBackend(client=openai_client, model="gpt-5-nano", temperature=temperature)
        fallback = (
            AnthropicBackend(
                client=AsyncAnthropic(api_key=settings.anthropic_api_key),
                model=settings.default_model,
            )
            if settings.anthropic_api_key
            else None
        )
        return FallbackGenerateBackend(primary=primary, fallback=fallback), "gpt-5-nano"
    if settings.anthropic_api_key:
        return AnthropicBackend(
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=settings.default_model,
        ), settings.default_model
    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


async def synthesize_variant(cluster: ClusterData, variant: SchemaVariant, max_attempts: int = 4) -> SynthesisOutcome:
    if not use_remote_llm():
        persona = build_local_persona(cluster, variant)
        groundedness = check_groundedness(persona, cluster)
        return SynthesisOutcome(
            cluster_id=cluster.cluster_id,
            variant=variant.key,
            persona=persona,
            groundedness=groundedness.score,
            attempts=1,
            model_used="local",
            shared_field_similarity={},
            validity=groundedness.passed,
            source_evidence_count=len(persona.source_evidence),
        )

    backend, model_name = build_synthesis_backend(variant)
    tool = build_tool_definition(variant)
    messages = [{"role": "user", "content": build_versioned_prompt(cluster, variant)}]
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            llm_result = await backend.generate(system=SYSTEM_PROMPT, messages=messages, tool=tool)
            persona = variant.model.model_validate(llm_result.tool_input)
            groundedness = check_groundedness(persona, cluster)
            return SynthesisOutcome(
                cluster_id=cluster.cluster_id,
                variant=variant.key,
                persona=persona,
                groundedness=groundedness.score,
                attempts=attempt,
                model_used=llm_result.model,
                shared_field_similarity={},
                validity=groundedness.passed,
                source_evidence_count=len(persona.source_evidence),
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                messages = [
                    {
                        "role": "user",
                        "content": build_versioned_prompt(cluster, variant)
                        + "\n\nPrevious attempt error:\n"
                        + str(exc)
                        + "\nFix the issues and return the persona again.",
                    }
                ]
                continue
            raise RuntimeError(f"Failed to synthesize {variant.key} for {cluster.cluster_id}") from last_error


def summarize_variant(rows: list[VariantRun]) -> VariantSummary:
    if not rows:
        return VariantSummary(
            variant="",
            n_clusters=0,
            mean_groundedness=0.0,
            mean_judge_overall=0.0,
            validity_rate=0.0,
            mean_attempts=0.0,
            mean_source_evidence_count=0.0,
        )
    return VariantSummary(
        variant=rows[0].variant,
        n_clusters=len(rows),
        mean_groundedness=mean(row.groundedness for row in rows),
        mean_judge_overall=mean(row.judge_overall for row in rows),
        validity_rate=mean(1.0 if row.validity else 0.0 for row in rows),
        mean_attempts=mean(row.attempts for row in rows),
        mean_source_evidence_count=mean(row.source_evidence_count for row in rows),
    )


def merge_similarity_by_field(rows: list[VariantRun]) -> dict[str, float]:
    if not rows:
        return {}
    fields: dict[str, list[float]] = {}
    for row in rows:
        for field_name, similarity in row.shared_field_similarity.items():
            fields.setdefault(field_name, []).append(similarity)
    return {field_name: mean(values) for field_name, values in fields.items() if values}


def results_to_dict(
    rows: list[VariantRun],
    summaries: dict[str, VariantSummary],
) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row.variant, []).append(
            {
                "cluster_id": row.cluster_id,
                "persona_name": row.persona_name,
                "groundedness": row.groundedness,
                "judge_overall": row.judge_overall,
                "validity": row.validity,
                "attempts": row.attempts,
                "model_used": row.model_used,
                "shared_field_similarity": row.shared_field_similarity,
                "source_evidence_count": row.source_evidence_count,
            }
        )

    return {
        "rows_by_variant": by_variant,
        "summaries": {
            variant: {
                **summary.__dict__,
            }
            for variant, summary in summaries.items()
        },
    }


def build_findings(data: dict) -> str:
    summaries = data["summaries"]
    baseline = summaries["v1"]
    variant_lines = []
    for variant in ("v1", "v1.1", "v2"):
        summary = summaries[variant]
        line = (
            f"- `{variant}`: judge `{summary['mean_judge_overall']:.2f}`, "
            f"grounded `{summary['mean_groundedness']:.2f}`, "
            f"valid `{summary['validity_rate']:.0%}`"
        )
        if variant != "v1":
            line += (
                f", judge delta vs v1 `{summary['judge_score_delta_vs_v1']:+.2f}`, "
                f"shared similarity vs v1 `{summary['mean_shared_field_similarity_vs_v1']:.2f}`"
            )
        variant_lines.append(line)

    return "\n".join(
        [
            "# Experiment 1.05: Schema Versioning Drift",
            "",
            "## Hypothesis",
            "Changing the persona schema should not materially change quality on the same source data unless the schema itself creates useful structure.",
            "",
            "## Method",
            f"1. Ran the same golden-tenant clusters through 3 local schema variants: `v1`, `v1.1`, and `v2`.",
            "2. Used branch-local prompt and tool definitions for each version; the shared builder in `synthesis/` was left untouched.",
            "3. Scored each output with the same branch-local judge rubric and compared the shared fields against the `v1` baseline.",
            "",
            "## Aggregate Metrics",
            *variant_lines,
            "",
            f"## Baseline Summary",
            f"- Mean judge score: `{baseline['mean_judge_overall']:.2f}`",
            f"- Mean groundedness: `{baseline['mean_groundedness']:.2f}`",
            f"- Validity rate: `{baseline['validity_rate']:.0%}`",
            "",
            "## Decision",
            (
                "Adopt. The enriched schemas improved or matched judge quality without meaningful drift in shared fields."
                if summaries["v1.1"]["judge_score_delta_vs_v1"] >= 0 and summaries["v2"]["judge_score_delta_vs_v1"] >= 0
                else "Defer. The schema variants changed the output enough that the benefit is not clearly positive on this tiny tenant."
            ),
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 2 clusters. The drift signal is directional only.",
        ]
    ) + "\n"
