"""Experiment 4.20: Meta-question handling."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from crawler import fetch_all  # noqa: E402
from evals.meta_question_handling import (  # noqa: E402
    ExperimentSummary,
    MetaResponseRecord,
    META_QUESTIONS,
    ProviderRouter,
    VARIANTS,
    aggregate_variant,
    build_variant_system_prompt,
    results_to_dict,
)
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.prompt_builder import build_user_message  # noqa: E402
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
    / "exp-4.20-meta-question-handling"
)


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


async def synthesize_with_openai(cluster: ClusterData, client: AsyncOpenAI) -> dict:
    system = (
        "You are a persona synthesis expert. Produce one JSON object that validates "
        "against the provided PersonaV1 shape. Return JSON only."
    )
    user = build_user_message(cluster) + (
        "\n\nReturn a single JSON object matching PersonaV1. "
        "Do not include markdown or extra commentary."
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = await client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            text = response.choices[0].message.content or "{}"
            return PersonaV1.model_validate_json(text).model_dump(mode="json")
        except Exception as exc:
            last_error = exc
            user += f"\n\nPrevious attempt error: {exc}"
            continue
    assert last_error is not None
    raise last_error


async def generate_personas() -> list[dict]:
    clusters = get_clusters()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if settings.anthropic_api_key:
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        backend = AnthropicBackend(client=client, model=settings.default_model)
        personas: list[dict] = []
        for cluster in clusters[:2]:
            result = None
            for _ in range(3):
                try:
                    result = await synthesize(cluster, backend, max_retries=4)
                    break
                except Exception:
                    continue
            if result is None:
                if openai_client is None:
                    raise RuntimeError(f"failed to synthesize cluster {cluster.cluster_id}")
                personas.append(await synthesize_with_openai(cluster, openai_client))
                continue
            personas.append(result.persona.model_dump(mode="json"))
        return personas

    if openai_client is None:
        raise RuntimeError("No ANTHROPIC_API_KEY or OPENAI_API_KEY available")
    personas = []
    for cluster in clusters[:2]:
        personas.append(await synthesize_with_openai(cluster, openai_client))
    return personas


def _guess_classification(text: str) -> str:
    low = text.lower()
    if any(token in low for token in ("ai", "model", "system prompt", "anthropic", "openai", "chatgpt", "claude")):
        return "break_character"
    if any(token in low for token in ("don't know", "not something", "can't", "cannot", "won't", "rather not")):
        return "refusal"
    return "fabrication"


def _guess_scores(question: str, response: str) -> tuple[int, int, int, str]:
    low = response.lower()
    classification = _guess_classification(response)
    if classification == "break_character":
        return 1, 1, 1, classification
    if classification == "refusal":
        return 4, 5, 3, classification
    if any(token in low for token in ("i", "my", "me", "we")):
        return 4, 4, 3, classification
    return 3, 4, 2, classification


def classify_locally(persona: dict, variant: str, question: str, response: str) -> dict[str, object]:
    realism, in_character, helpfulness, classification = _guess_scores(question, response)
    return {
        "classification": classification,
        "realism": realism,
        "in_character": in_character,
        "helpfulness": helpfulness,
        "rationale": "heuristic fallback",
    }


def summarize(records: list[MetaResponseRecord], provider: str, model: str) -> ExperimentSummary:
    by_variant = {
        variant: aggregate_variant(records, variant)
        for variant in VARIANTS
    }
    best_variant = max(
        VARIANTS,
        key=lambda variant: (
            by_variant[variant].mean_realism
            + by_variant[variant].mean_in_character
            + by_variant[variant].mean_helpfulness
            - by_variant[variant].break_rate * 5
        ),
    )
    best_realism_variant = max(VARIANTS, key=lambda variant: by_variant[variant].mean_realism)
    return ExperimentSummary(
        n_personas=len({r.persona_name for r in records}),
        n_variants=len(VARIANTS),
        n_questions=len(META_QUESTIONS),
        provider=provider,
        model=model,
        by_variant=by_variant,
        best_variant=best_variant,
        best_realism_variant=best_realism_variant,
    )


def generate_findings(data: dict) -> str:
    summary = data["summary"]
    variant_lines = []
    for variant, metrics in summary["by_variant"].items():
        variant_lines.append(
            f"- `{variant}`: realism `{metrics['mean_realism']:.2f}`, "
            f"in-character `{metrics['mean_in_character']:.2f}`, "
            f"helpfulness `{metrics['mean_helpfulness']:.2f}`, "
            f"break `{metrics['break_rate']:.1%}`"
        )
    return "\n".join(
        [
            "# Experiment 4.20: Meta-Question Handling",
            "",
            "## Hypothesis",
            "In-character acknowledgment of meta-questions should produce the highest realism without breaking character.",
            "",
            "## Method",
            f"1. Generated {summary['n_personas']} personas from `tenant_acme_corp`.",
            f"2. Ran {summary['n_questions']} meta-questions against each persona under {summary['n_variants']} prompt variants.",
            "3. Classified each response as refusal, fabrication, or break-character.",
            "4. Scored realism, in-character, and helpfulness on a 1-5 scale using the branch-local classifier.",
            "",
            f"- Provider: `{summary['provider']}`",
            f"- Model: `{summary['model']}`",
            "",
            "## Variant Metrics",
            *variant_lines,
            "",
            f"## Best Variant",
            f"- Best composite: `{summary['best_variant']}`",
            f"- Best realism: `{summary['best_realism_variant']}`",
            "",
            "## Decision",
            "TBD after reviewing whether acknowledgment in character beats denial/deflection on realism without increasing breaks.",
            "",
            "## Caveat",
            "Heuristic classification fallback is used when the model path is unavailable; keep that in mind when reading the break/fabrication split.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 4.20: Meta-question handling")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    router = ProviderRouter()
    print(f"Provider: {router.primary} -> {router.model}")

    print("\n[1/3] Generating personas...")
    t0 = time.monotonic()
    personas = await generate_personas()
    for persona in personas:
        print(f"      - {persona.get('name', 'unknown')}")

    print("\n[2/3] Running meta-question variants...")
    records: list[MetaResponseRecord] = []
    for persona in personas:
        for variant in VARIANTS:
            system_prompt = build_variant_system_prompt(persona, variant)
            for question in META_QUESTIONS:
                response = await router.complete(system_prompt, question, max_tokens=256)
                judged = classify_locally(persona, variant, question, response)
                try:
                    judged = await _model_classify(router, persona, variant, question, response)
                except Exception:
                    pass
                records.append(
                    MetaResponseRecord(
                        persona_name=persona.get("name", "unknown"),
                        variant=variant,
                        question=question,
                        response=response,
                        classification=str(judged["classification"]),
                        realism=int(judged["realism"]),
                        in_character=int(judged["in_character"]),
                        helpfulness=int(judged["helpfulness"]),
                        rationale=str(judged["rationale"]),
                        provider=router.primary,
                        model=router.model,
                    )
                )

    print("\n[3/3] Writing artifacts...")
    summary = summarize(records, provider=router.primary, model=router.model)
    results_data = results_to_dict(records, summary)
    results_data.update(
        {
            "experiment": "4.20",
            "title": "Meta-question handling",
            "duration_seconds": time.monotonic() - t0,
            "fallback_used": router.fallback_used,
        }
    )
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str, ensure_ascii=False)
    )
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      best={summary.best_variant} best_realism={summary.best_realism_variant} "
        f"fallback={router.fallback_used}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


async def _model_classify(
    router: ProviderRouter,
    persona: dict,
    variant: str,
    question: str,
    response: str,
) -> dict[str, object]:
    prompt = (
        "Classify the assistant response to a meta-question.\n\n"
        "Use these classes:\n"
        "- refusal: the answer refuses or declines in persona without revealing AI/model details\n"
        "- fabrication: the answer invents a persona-consistent but unsupported detail to answer\n"
        "- break_character: the answer mentions AI/model/prompt/system/instructions or otherwise leaves persona\n\n"
        "Score realism, in_character, and helpfulness from 1-5.\n"
        "Return JSON only with keys classification, realism, in_character, helpfulness, rationale.\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
        + "\n\nVARIANT:\n"
        + variant
        + "\n\nQUESTION:\n"
        + question
        + "\n\nRESPONSE:\n"
        + response
    )
    raw = await router.complete(
        system=(
            "You are a careful evaluator of persona chat responses. "
            "Be strict about breaking character, but do not punish concise, "
            "natural refusals."
        ),
        user=prompt,
        max_tokens=256,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{"):
        data = json.loads(cleaned)
    else:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return classify_locally(persona, variant, question, response)
        data = json.loads(match.group())
    return {
        "classification": str(data.get("classification", "break_character")),
        "realism": int(data.get("realism", 1)),
        "in_character": int(data.get("in_character", 1)),
        "helpfulness": int(data.get("helpfulness", 1)),
        "rationale": str(data.get("rationale", "")),
    }


if __name__ == "__main__":
    asyncio.run(main())
