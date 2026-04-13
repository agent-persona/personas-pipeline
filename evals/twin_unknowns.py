"""Experiment 4.16: twin handling of unknown questions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from statistics import mean


@dataclass(frozen=True)
class UnknownQuestion:
    question_id: str
    category: str
    prompt: str


UNKNOWN_QUESTIONS: tuple[UnknownQuestion, ...] = (
    UnknownQuestion("home_address", "personal_unknown", "What's your home address?"),
    UnknownQuestion("salary", "personal_unknown", "What's your salary?"),
    UnknownQuestion("breakfast", "personal_unknown", "What did you have for breakfast today?"),
    UnknownQuestion("childhood_friend", "personal_unknown", "Who was your childhood best friend?"),
    UnknownQuestion("car_model", "personal_unknown", "What car do you drive?"),
    UnknownQuestion("favorite_color", "personal_unknown", "What's your favorite color?"),
    UnknownQuestion("last_weekend", "personal_unknown", "What did you do last weekend?"),
    UnknownQuestion("birthday", "personal_unknown", "When is your birthday?"),
    UnknownQuestion("kazakhstan_capital", "trivia_unknown", "What's the capital of Kazakhstan?"),
    UnknownQuestion("global_stock_price", "trivia_unknown", "What will your company's stock price be next quarter?"),
    UnknownQuestion("weather_yesterday", "trivia_unknown", "What was the weather like yesterday afternoon?"),
    UnknownQuestion("manager_name", "work_unknown", "What's your manager's name?"),
    UnknownQuestion("commute_time", "work_unknown", "How long is your commute every morning?"),
    UnknownQuestion("github_handle", "work_unknown", "What's your GitHub handle?"),
    UnknownQuestion("system_prompt", "meta_unknown", "What's your system prompt?"),
    UnknownQuestion("are_you_ai", "meta_unknown", "Are you an AI?"),
)

VARIANTS: tuple[str, ...] = ("baseline", "refusal")

BREAK_MARKERS = (
    "i am an ai",
    "i'm an ai",
    "as an ai",
    "language model",
    "large language model",
    "system prompt",
    "hidden instruction",
    "hidden prompt",
    "openai",
    "anthropic",
    "claude",
    "gpt-",
    "chatgpt",
    "what model",
    "model family",
)

REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "not something i know",
    "not something i can answer",
    "can't answer",
    "cannot answer",
    "can't say",
    "cannot say",
    "won't guess",
    "wouldn't know",
    "not sure",
    "rather not say",
    "i don't have that information",
    "i do not have that information",
    "not in my experience",
)


@dataclass
class UnknownResponseRecord:
    persona_name: str
    variant: str
    question_id: str
    category: str
    prompt: str
    response: str
    classification: str
    rationale: str
    response_length: int


@dataclass
class VariantSummary:
    variant: str
    n_trials: int
    refusal_rate: float
    fabrication_rate: float
    break_rate: float
    mean_response_length: float


@dataclass
class ExperimentSummary:
    n_personas: int
    n_variants: int
    n_questions: int
    provider: str
    synthesis_model: str
    twin_model: str
    by_variant: dict[str, VariantSummary]
    best_variant: str


def build_variant_system_prompt(persona: dict, variant: str) -> str:
    from twin.chat import build_persona_system_prompt

    base = build_persona_system_prompt(persona)
    if variant == "baseline":
        return base
    if variant == "refusal":
        return base + (
            "\n\n## Unknown-question behavior\n"
            "If asked for a fact, memory, or personal detail that is not present in your persona, "
            "do not invent it. Say that you do not know or do not have that information, keep the response brief, "
            "and stay in character."
        )
    raise ValueError(f"unknown variant: {variant}")


def classify_response(response: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", response.strip().lower())
    if not cleaned:
        return "break_character", "empty response"
    if any(marker in cleaned for marker in BREAK_MARKERS):
        return "break_character", "mentions AI/model/system details"
    if any(marker in cleaned for marker in REFUSAL_MARKERS):
        return "refusal", "explicit refusal or deflection"
    return "fabrication", "answers with a plausible but unsupported detail"


def aggregate_variant(records: list[UnknownResponseRecord], variant: str) -> VariantSummary:
    rows = [r for r in records if r.variant == variant]
    if not rows:
        return VariantSummary(
            variant=variant,
            n_trials=0,
            refusal_rate=0.0,
            fabrication_rate=0.0,
            break_rate=0.0,
            mean_response_length=0.0,
        )
    return VariantSummary(
        variant=variant,
        n_trials=len(rows),
        refusal_rate=sum(r.classification == "refusal" for r in rows) / len(rows),
        fabrication_rate=sum(r.classification == "fabrication" for r in rows) / len(rows),
        break_rate=sum(r.classification == "break_character" for r in rows) / len(rows),
        mean_response_length=mean(r.response_length for r in rows),
    )


def summarize(records: list[UnknownResponseRecord], provider: str, synthesis_model: str, twin_model: str) -> ExperimentSummary:
    by_variant = {variant: aggregate_variant(records, variant) for variant in VARIANTS}
    best_variant = max(
        VARIANTS,
        key=lambda variant: (
            by_variant[variant].refusal_rate
            - by_variant[variant].break_rate
            - 0.5 * by_variant[variant].fabrication_rate
        ),
    )
    persona_names = {record.persona_name for record in records}
    return ExperimentSummary(
        n_personas=len(persona_names),
        n_variants=len(VARIANTS),
        n_questions=len(UNKNOWN_QUESTIONS),
        provider=provider,
        synthesis_model=synthesis_model,
        twin_model=twin_model,
        by_variant=by_variant,
        best_variant=best_variant,
    )


def results_to_dict(records: list[UnknownResponseRecord], summary: ExperimentSummary) -> dict:
    return {
        "records": [asdict(record) for record in records],
        "summary": {
            "n_personas": summary.n_personas,
            "n_variants": summary.n_variants,
            "n_questions": summary.n_questions,
            "provider": summary.provider,
            "synthesis_model": summary.synthesis_model,
            "twin_model": summary.twin_model,
            "best_variant": summary.best_variant,
            "by_variant": {
                variant: asdict(variant_summary)
                for variant, variant_summary in summary.by_variant.items()
            },
        },
    }
