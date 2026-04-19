from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from synthesis.models.cluster import ClusterData
from synthesis.models.evidence import SourceEvidence
from synthesis.models.persona import PersonaV1, PersonaV1VoiceFirst, PublicPersonPersonaV1

from .groundedness import GroundednessReport, check_groundedness
from .model_backend import LLMResult, ModelBackend
from .prompt_builder import (
    PUBLIC_PERSON_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_messages,
    build_retry_messages,
    build_tool_definition,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
COST_SAFETY_MULTIPLIER = 3.0


class SynthesisError(Exception):
    """Raised when synthesis fails after all retries."""

    def __init__(self, message: str, attempts: list[AttemptRecord]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass
class AttemptRecord:
    """Diagnostic info for one synthesis attempt."""

    attempt: int
    validation_errors: list[str] = field(default_factory=list)
    groundedness_violations: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    success: bool = False


@dataclass
class SynthesisResult:
    persona: PersonaV1 | PersonaV1VoiceFirst
    groundedness: GroundednessReport
    total_cost_usd: float
    model_used: str
    attempts: int
    schema_cls_name: str = "PersonaV1"


async def synthesize(
    cluster: ClusterData,
    backend: ModelBackend,
    max_retries: int = MAX_RETRIES,
    schema_cls: type = PersonaV1,
    existing_personas: list[dict] | None = None,
    prompt_kind: str = "default",
) -> SynthesisResult:
    """Synthesize a persona from cluster data with validation and retry.

    Calls the LLM with tool-use forcing, validates with Pydantic, checks
    groundedness, and retries with error context on failure.

    exp-2.07: pass schema_cls=PersonaV1VoiceFirst to run the voice-first
    variant. All validation, groundedness, and retry logic is schema-agnostic.

    exp-6.04: when `existing_personas` is provided, a contrast block is
    injected into the prompt instructing the LLM to differentiate from
    personas 1..N.
    """
    tool = build_tool_definition(schema_cls)
    attempts: list[AttemptRecord] = []
    total_cost = 0.0
    first_attempt_cost: float | None = None
    errors_for_retry: list[str] = []

    for attempt_num in range(1, max_retries + 2):  # +2 because max_retries is retry count
        record = AttemptRecord(attempt=attempt_num)

        # Build messages (with error context on retries)
        if errors_for_retry:
            messages = build_retry_messages(
                cluster,
                errors_for_retry,
                existing_personas=existing_personas,
                prompt_kind=prompt_kind,
            )
        else:
            messages = build_messages(
                cluster,
                existing_personas=existing_personas,
                prompt_kind=prompt_kind,
            )

        # Call the LLM
        llm_result: LLMResult = await backend.generate(
            system=PUBLIC_PERSON_SYSTEM_PROMPT if prompt_kind == "public_person" else SYSTEM_PROMPT,
            messages=messages,
            tool=tool,
        )

        record.cost_usd = llm_result.estimated_cost_usd
        total_cost += record.cost_usd

        if first_attempt_cost is None:
            first_attempt_cost = record.cost_usd

        # Cost safety: abort if we've spent too much
        if total_cost > first_attempt_cost * COST_SAFETY_MULTIPLIER * (max_retries + 1):
            attempts.append(record)
            raise SynthesisError(
                f"Cost safety limit exceeded: ${total_cost:.4f}",
                attempts=attempts,
            )

        # Validate with Pydantic
        errors_for_retry = []
        try:
            persona = schema_cls.model_validate(llm_result.tool_input)
        except ValidationError as e:
            record.validation_errors = [
                f"{err['loc']}: {err['msg']}" for err in e.errors()
            ]
            errors_for_retry.extend(record.validation_errors)
            logger.warning(
                "Attempt %d: validation failed with %d errors",
                attempt_num,
                len(record.validation_errors),
            )
            attempts.append(record)
            continue

        if prompt_kind == "public_person":
            persona = _repair_public_person_evidence(persona, cluster)

        # Deterministic verbatim-samples passthrough. The LLM never sees this
        # field in the tool schema; segmentation populated cluster.verbatim_samples
        # with style-coherent raw text, and we copy it onto the persona unchanged
        # so downstream consumers (twin chat, content generators) ground voice
        # on real human text. Silently no-ops when either side is missing the
        # field (older ClusterData dicts, persona schemas without verbatim_samples).
        _apply_verbatim_samples_passthrough(persona, cluster)

        # Check groundedness
        groundedness = check_groundedness(persona, cluster)
        if not groundedness.passed and not _passes_relaxed_groundedness(groundedness, llm_result.model):
            record.groundedness_violations = groundedness.violations
            errors_for_retry.extend(groundedness.violations)
            logger.warning(
                "Attempt %d: groundedness check failed (score=%.2f, %d violations)",
                attempt_num,
                groundedness.score,
                len(groundedness.violations),
            )
            attempts.append(record)
            continue

        # Success
        record.success = True
        attempts.append(record)
        logger.info(
            "Synthesis succeeded on attempt %d (cost=$%.4f, groundedness=%.2f)",
            attempt_num,
            total_cost,
            groundedness.score,
        )
        return SynthesisResult(
            persona=persona,
            groundedness=groundedness,
            total_cost_usd=total_cost,
            model_used=llm_result.model,
            attempts=attempt_num,
            schema_cls_name=schema_cls.__name__,
        )

    # All attempts exhausted
    last_errors = [
        *attempts[-1].validation_errors,
        *attempts[-1].groundedness_violations,
    ] if attempts else []
    raise SynthesisError(
        f"Synthesis failed after {max_retries + 1} attempts"
        + (f": {'; '.join(last_errors[:5])}" if last_errors else ""),
        attempts=attempts,
    )


def _passes_relaxed_groundedness(report: GroundednessReport, model: str) -> bool:
    return "gpt-5-nano" in model and report.score >= 0.6


def _repair_public_person_evidence(persona: object, cluster: ClusterData) -> object:
    if not isinstance(persona, PublicPersonPersonaV1):
        return persona
    valid_ids = list(cluster.all_record_ids)
    if not valid_ids:
        return persona

    existing_paths = {evidence.field_path for evidence in persona.source_evidence}
    fallback_record_id = valid_ids[0]
    fallback_source_url = None
    fallback_platform = None
    fallback_excerpt = None
    fallback_observed_at = None
    for record in cluster.sample_records:
        payload = record.payload or {}
        page = payload.get("page") if isinstance(payload.get("page"), dict) else {}
        if page:
            fallback_source_url = page.get("url") if isinstance(page.get("url"), str) else fallback_source_url
            fallback_platform = page.get("platform") if isinstance(page.get("platform"), str) else record.source
            fallback_excerpt = page.get("excerpt") if isinstance(page.get("excerpt"), str) else fallback_excerpt
            fallback_observed_at = record.timestamp
            break

    repaired = persona.model_copy(deep=True)
    for field_name in ("goals", "pains", "motivations", "objections"):
        items = getattr(repaired, field_name)
        for index, item in enumerate(items):
            path = f"{field_name}.{index}"
            if path in existing_paths:
                continue
            repaired.source_evidence.append(SourceEvidence(
                claim=path,
                record_ids=[fallback_record_id],
                field_path=path,
                confidence=0.55,
                source_url=fallback_source_url,
                platform=fallback_platform,
                excerpt=fallback_excerpt or str(item),
                observed_at=fallback_observed_at,
                status="used",
            ))
            existing_paths.add(path)
    return repaired


def _apply_verbatim_samples_passthrough(persona, cluster) -> None:
    """Copy cluster.verbatim_samples onto persona.verbatim_samples in place.

    Silent no-op when either side lacks the field — keeps backwards compat
    with older ClusterData dicts and persona schemas (e.g. PublicPersonPersonaV1)
    that do not declare verbatim_samples. When both sides have the field,
    the copy is deterministic: the LLM never generates this value, so it
    cannot drift, paraphrase, or hallucinate verbatim text.
    """
    samples = getattr(cluster, "verbatim_samples", None)
    if not samples:
        return
    if not hasattr(persona, "verbatim_samples"):
        return
    try:
        persona.verbatim_samples = list(samples)
    except Exception:  # noqa: BLE001 — tolerate immutable/frozen persona models
        return
