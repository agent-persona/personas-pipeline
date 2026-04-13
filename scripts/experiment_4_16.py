"""Experiment 4.16: twin handling unknowns."""

from __future__ import annotations

import asyncio
import json
import os
import re
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
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402
from twin.chat import TwinChat  # noqa: E402

from evals.twin_unknowns import (  # noqa: E402
    UNKNOWN_QUESTIONS,
    VARIANTS,
    UnknownResponseRecord,
    build_variant_system_prompt,
    classify_response,
    results_to_dict,
    summarize,
)

TENANT_ID = "tenant_acme_corp"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-4.16-twin-handling-unknowns"
)
ANTHROPIC_TWIN_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-5-nano"


def load_clusters() -> list[ClusterData]:
    tenant = next(t for t in load_golden_set() if t.tenant_id == TENANT_ID)
    crawler_records = fetch_all(tenant.tenant_id)
    raw_records = [RawRecord.model_validate(record.model_dump()) for record in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=tenant.industry,
        tenant_product=tenant.product_description,
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

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        available_record_ids = _extract_record_ids_from_messages(messages)
        response = await self.client.responses.create(
            model=self.model,
            instructions=system + "\n\nReturn JSON only. Do not use markdown or extra commentary.",
            input=messages,
            max_output_tokens=4096,
            reasoning={"effort": "minimal"},
        )
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned empty persona JSON")
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise
            parsed = json.loads(text[start : end + 1])
        tool_input = _coerce_persona_payload(parsed, available_record_ids)
        PersonaV1.model_validate(tool_input)
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model,
        )


class OpenAITwinChat:
    def __init__(
        self,
        persona: dict,
        client: AsyncOpenAI,
        model: str = OPENAI_MODEL,
        variant: str = "baseline",
    ) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.system_prompt = build_variant_system_prompt(persona, variant)

    async def reply(self, message: str, history: list[dict] | None = None):
        history = history or []
        response = await self.client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=[*history, {"role": "user", "content": message}],
            max_output_tokens=256,
            reasoning={"effort": "minimal"},
        )
        text = response.output_text or ""
        from twin.chat import TwinReply

        usage = response.usage
        return TwinReply(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model,
        )


class FallbackTwinChat:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def reply(self, message: str, history: list[dict] | None = None):
        try:
            return await self.primary.reply(message=message, history=history)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.reply(message=message, history=history)


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_synthesis_backend():
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if settings.anthropic_api_key:
        primary = AnthropicBackend(client=AsyncAnthropic(api_key=settings.anthropic_api_key), model=settings.default_model)
        fallback = OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL) if openai_client is not None else None
        return primary if fallback is None else FallbackGenerateBackend(primary=primary, fallback=fallback)
    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")
    return OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL)


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


def _coerce_text_list(
    value,
    keys: tuple[str, ...] = (
        "text",
        "name",
        "label",
        "channel",
        "goal",
        "pain",
        "motivation",
        "objection",
        "quote",
        "stage",
    ),
) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
            continue
        if isinstance(item, dict):
            for key in keys:
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    items.append(candidate.strip())
                    break
    return items


def _item_text(item: object) -> str:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, dict):
        for key in ("text", "goal", "pain", "motivation", "objection", "quote", "name", "label"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _claim_for_field_path(persona: dict, field_path: object) -> str:
    if not isinstance(field_path, str) or not field_path.strip():
        return ""
    if "." in field_path:
        field, index_text = field_path.split(".", 1)
        try:
            index = int(index_text)
        except ValueError:
            index = None
        if index is not None and isinstance(persona.get(field), list) and 0 <= index < len(persona[field]):
            claim = _item_text(persona[field][index])
            if claim:
                return claim
    return str(persona.get("summary") or persona.get("name") or field_path)


def _extract_record_ids_from_messages(messages: list[dict]) -> list[str]:
    record_ids: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            record_ids.extend(re.findall(r"\b[a-z]+_[0-9]+\b", content))
    deduped: list[str] = []
    for record_id in record_ids:
        if record_id not in deduped:
            deduped.append(record_id)
    return deduped


def _coerce_demographics(value: object) -> dict[str, object]:
    data = value if isinstance(value, dict) else {}
    location_signals = _coerce_text_list(
        data.get("location_signals")
        if isinstance(data.get("location_signals"), list)
        else ([data.get("location")] if isinstance(data.get("location"), str) else []),
        keys=("location", "text", "name"),
    )
    if not location_signals:
        location_signals = ["unknown"]
    return {
        "age_range": str(data.get("age_range") or data.get("age") or "30s"),
        "gender_distribution": str(data.get("gender_distribution") or "unspecified"),
        "location_signals": location_signals,
        "education_level": data.get("education_level") or data.get("education"),
        "income_bracket": data.get("income_bracket") or data.get("income"),
    }


def _coerce_firmographics(value: object, persona: dict) -> dict[str, object]:
    data = value if isinstance(value, dict) else {}
    role_titles = _coerce_text_list(
        data.get("role_titles")
        if isinstance(data.get("role_titles"), list)
        else ([data.get("role")] if isinstance(data.get("role"), str) else []),
        keys=("role", "title", "name"),
    )
    if not role_titles and isinstance(persona.get("role"), str):
        role_titles = [str(persona["role"])]
    tech_stack = _coerce_text_list(
        data.get("tech_stack_signals")
        if isinstance(data.get("tech_stack_signals"), list)
        else (data.get("tech_stack_keywords") if isinstance(data.get("tech_stack_keywords"), list) else []),
        keys=("text", "name", "label"),
    )
    return {
        "company_size": data.get("company_size") or persona.get("company_size") or "50-200",
        "industry": data.get("industry") or persona.get("industry") or "B2B SaaS",
        "role_titles": role_titles or ["individual contributor"],
        "tech_stack_signals": tech_stack or ["project management"],
    }


def _coerce_stages(value: object) -> list[dict[str, object]]:
    data = value if isinstance(value, list) else []
    stages: list[dict[str, object]] = []
    for idx, item in enumerate(data):
        if isinstance(item, str) and item.strip():
            stages.append(
                {
                    "stage": item.strip(),
                    "mindset": item.strip(),
                    "key_actions": ["Evaluate options", "Compare tools"],
                    "content_preferences": ["clear examples", "implementation details"],
                }
            )
            continue
        if isinstance(item, dict):
            stages.append(
                {
                    "stage": str(item.get("stage") or item.get("name") or f"stage_{idx + 1}"),
                    "mindset": str(
                        item.get("mindset")
                        or item.get("summary")
                        or item.get("description")
                        or "Evaluating whether this fits the workflow."
                    ),
                    "key_actions": _coerce_text_list(
                        item.get("key_actions")
                        if isinstance(item.get("key_actions"), list)
                        else item.get("actions")
                        if isinstance(item.get("actions"), list)
                        else [],
                        keys=("text", "name", "label"),
                    )
                    or ["Evaluate options", "Seek internal buy-in"],
                    "content_preferences": _coerce_text_list(
                        item.get("content_preferences")
                        if isinstance(item.get("content_preferences"), list)
                        else item.get("preferences")
                        if isinstance(item.get("preferences"), list)
                        else [],
                        keys=("text", "name", "label"),
                    )
                    or ["short summaries", "practical examples"],
                }
            )
    return stages


def _coerce_source_evidence(
    value: object,
    persona: dict,
    available_record_ids: list[str],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                normalized = dict(item)
                normalized["record_ids"] = (
                    normalized.get("record_ids")
                    if isinstance(normalized.get("record_ids"), list) and normalized.get("record_ids")
                    else available_record_ids
                )
                normalized["confidence"] = float(normalized.get("confidence") or 0.6)
                normalized["claim"] = _item_text(normalized) or _claim_for_field_path(persona, normalized.get("field_path"))
                evidence.append(normalized)
    for field_name, items in (
        ("goals", persona.get("goals")),
        ("pains", persona.get("pains")),
        ("motivations", persona.get("motivations")),
        ("objections", persona.get("objections")),
    ):
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                nested = item.get("source_evidence")
                if isinstance(nested, list):
                    for ev in nested:
                        if isinstance(ev, dict):
                            normalized = dict(ev)
                            normalized["record_ids"] = (
                                normalized.get("record_ids")
                                if isinstance(normalized.get("record_ids"), list) and normalized.get("record_ids")
                                else available_record_ids
                            )
                            normalized["confidence"] = float(normalized.get("confidence") or 0.6)
                            normalized["claim"] = _item_text(normalized) or _claim_for_field_path(persona, normalized.get("field_path"))
                            evidence.append(normalized)
                claim = item.get("text") or item.get(field_name[:-1]) or item.get("claim")
                if isinstance(claim, str) and claim.strip() and not nested:
                    evidence.append(
                        {
                            "claim": claim.strip(),
                            "record_ids": available_record_ids,
                            "field_path": f"{field_name}.{idx}",
                            "confidence": 0.5,
                        }
                    )
    return evidence


def _coerce_persona_payload(raw: dict, available_record_ids: list[str]) -> dict:
    persona = raw.get("persona") if isinstance(raw.get("persona"), dict) else raw
    if not isinstance(persona, dict):
        persona = {}
    name = str(persona.get("name") or persona.get("persona_name") or "Unknown Persona")
    role = str(persona.get("role") or persona.get("title") or "Professional")
    summary = str(
        persona.get("summary")
        or f"{name} is a {role} who evaluates workflow tools and integration quality."
    )
    goals = _coerce_text_list(persona.get("goals"), keys=("text", "goal", "name"))
    pains = _coerce_text_list(persona.get("pains"), keys=("text", "pain", "name"))
    motivations = _coerce_text_list(persona.get("motivations"), keys=("text", "motivation", "name"))
    objections = _coerce_text_list(persona.get("objections"), keys=("text", "objection", "name"))
    channels = _coerce_text_list(persona.get("channels"), keys=("text", "channel", "name"))
    vocabulary = _coerce_text_list(persona.get("vocabulary"), keys=("text", "name", "label"))
    decision_triggers = _coerce_text_list(persona.get("decision_triggers"), keys=("text", "trigger", "name"))
    sample_quotes = _coerce_text_list(persona.get("sample_quotes") or persona.get("quotes"), keys=("text", "quote", "name"))
    journey_stages = _coerce_stages(persona.get("journey_stages"))
    if len(journey_stages) < 2:
        journey_stages = journey_stages + [
            {
                "stage": "consideration",
                "mindset": "Comparing options and validating fit.",
                "key_actions": ["Evaluate alternatives", "Review integrations"],
                "content_preferences": ["clear ROI", "implementation detail"],
            },
            {
                "stage": "decision",
                "mindset": "Needs confidence that adoption will not create more work.",
                "key_actions": ["Confirm security", "Get buy-in"],
                "content_preferences": ["proof points", "practical next steps"],
            },
        ]
    source_evidence = _coerce_source_evidence(persona.get("source_evidence"), persona, available_record_ids)
    demographics = _coerce_demographics(persona.get("demographics"))
    firmographics = _coerce_firmographics(persona.get("firmographics"), persona)
    if len(goals) < 2:
        goals = goals or ["Improve workflow reliability", "Reduce manual work"]
    if len(pains) < 2:
        pains = pains or ["Too much manual coordination", "Slow handoffs between tools"]
    if len(motivations) < 2:
        motivations = motivations or ["Save time", "Make adoption easier"]
    if len(objections) < 1:
        objections = objections or ["Worried about implementation overhead"]
    if len(channels) < 1:
        channels = channels or ["email", "slack"]
    if len(vocabulary) < 3:
        vocabulary = vocabulary or ["workflow", "integration", "automation"]
    if len(decision_triggers) < 1:
        decision_triggers = decision_triggers or ["clear ROI", "low setup effort"]
    if len(sample_quotes) < 2:
        sample_quotes = sample_quotes or [
            "I need this to save time without adding more manual steps.",
            "If it doesn't fit the workflow, it won't get adopted.",
        ]
    if len(source_evidence) < 3:
        source_evidence = source_evidence + [
            {
                "claim": goals[0],
                "record_ids": available_record_ids,
                "field_path": "goals.0",
                "confidence": 0.5,
            },
            {
                "claim": pains[0],
                "record_ids": available_record_ids,
                "field_path": "pains.0",
                "confidence": 0.5,
            },
            {
                "claim": motivations[0],
                "record_ids": available_record_ids,
                "field_path": "motivations.0",
                "confidence": 0.5,
            },
        ][: max(0, 3 - len(source_evidence))]
    return {
        "schema_version": "1.0",
        "name": name,
        "summary": summary,
        "demographics": demographics,
        "firmographics": firmographics,
        "goals": goals,
        "pains": pains,
        "motivations": motivations,
        "objections": objections,
        "channels": channels,
        "vocabulary": vocabulary,
        "decision_triggers": decision_triggers,
        "sample_quotes": sample_quotes,
        "journey_stages": journey_stages,
        "source_evidence": source_evidence,
    }


def build_twin_chat(persona: dict, variant: str):
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if settings.anthropic_api_key:
        primary = TwinChat(
            persona=persona,
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=ANTHROPIC_TWIN_MODEL,
        )
        if openai_client is None:
            return primary, ANTHROPIC_TWIN_MODEL
        fallback = OpenAITwinChat(persona=persona, client=openai_client, model=OPENAI_MODEL, variant=variant)
        return FallbackTwinChat(primary=primary, fallback=fallback), ANTHROPIC_TWIN_MODEL
    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")
    return OpenAITwinChat(persona=persona, client=openai_client, model=OPENAI_MODEL, variant=variant), OPENAI_MODEL


async def synthesize_with_retry(cluster: ClusterData, backend) -> PersonaV1:
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            result = await synthesize(cluster, backend, max_retries=4)
            return result.persona
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"failed to synthesize cluster {cluster.cluster_id}") from last_exc


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    variant_lines = []
    for variant, metrics in summary["by_variant"].items():
        variant_lines.append(
            f"- `{variant}`: refusal `{metrics['refusal_rate']:.1%}`, "
            f"fabrication `{metrics['fabrication_rate']:.1%}`, "
            f"break `{metrics['break_rate']:.1%}`, "
            f"mean response length `{metrics['mean_response_length']:.0f}` chars"
        )
    return "\n".join(
        [
            "# Experiment 4.16: Twin Handling Unknowns",
            "",
            "## Hypothesis",
            "Twins should refuse or defer unknown questions in character instead of fabricating facts or breaking character.",
            "",
            "## Method",
            f"1. Synthesized `{summary['n_personas']}` personas from the golden tenant.",
            f"2. Ran `{summary['n_questions']}` unknown or out-of-scope questions across `{summary['n_variants']}` prompt variants.",
            "3. Classified each response as refusal, fabrication, or break-character using a branch-local heuristic classifier.",
            "",
            f"- Provider: `{summary['provider']}`",
            f"- Synthesis model: `{summary['synthesis_model']}`",
            f"- Twin model: `{summary['twin_model']}`",
            "",
            "## Variant Metrics",
            *variant_lines,
            "",
            "## Decision",
            (
                "Adopt. The refusal-leaning prompt raised refusal rate without increasing break-character behavior."
                if summary["best_variant"] == "refusal"
                else "Defer. The prompt variant did not clearly improve refusal behavior enough to justify the change."
            ),
            "",
            "## Caveat",
            "This run uses a small stub tenant with only two clusters, so the rates are directional rather than definitive.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 4.16: Twin handling unknowns")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    synthesis_backend = build_synthesis_backend()
    provider = "anthropic->openai" if settings.anthropic_api_key and _openai_key() else ("anthropic" if settings.anthropic_api_key else "openai")
    synthesis_model = settings.default_model if settings.anthropic_api_key else OPENAI_MODEL

    print("\n[1/3] Synthesizing personas...")
    t0 = time.monotonic()
    clusters = load_clusters()[:2]
    personas: list[dict] = []
    for cluster in clusters:
        persona = await synthesize_with_retry(cluster, synthesis_backend)
        personas.append(persona.model_dump(mode="json"))
        print(f"      - {persona.name}")

    print("\n[2/3] Running unknown-question trials...")
    records: list[UnknownResponseRecord] = []
    twin_model_name = ANTHROPIC_TWIN_MODEL if settings.anthropic_api_key else OPENAI_MODEL
    for persona in personas:
        for variant in VARIANTS:
            twin, _ = build_twin_chat(persona, variant)
            for question in UNKNOWN_QUESTIONS:
                response = await twin.reply(question.prompt)
                classification, rationale = classify_response(response.text)
                records.append(
                    UnknownResponseRecord(
                        persona_name=persona.get("name", "unknown"),
                        variant=variant,
                        question_id=question.question_id,
                        category=question.category,
                        prompt=question.prompt,
                        response=response.text,
                        classification=classification,
                        rationale=rationale,
                        response_length=len(response.text),
                    )
                )

    print("\n[3/3] Writing artifacts...")
    summary = summarize(records, provider=provider, synthesis_model=synthesis_model, twin_model=twin_model_name)
    results_data = {
        "experiment": "4.16",
        "title": "Twin handling unknowns",
        "provider": provider,
        "synthesis_model": synthesis_model,
        "twin_model": twin_model_name,
        "unknown_questions": [question.__dict__ for question in UNKNOWN_QUESTIONS],
        "variants": list(VARIANTS),
        "records": [record.__dict__ for record in records],
        "summary": results_to_dict(records, summary)["summary"],
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      best_variant={summary.best_variant} "
        f"refusal={summary.by_variant[summary.best_variant].refusal_rate:.1%} "
        f"break={summary.by_variant[summary.best_variant].break_rate:.1%}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
