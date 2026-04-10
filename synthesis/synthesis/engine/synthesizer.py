from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel
from pydantic import ValidationError

from synthesis.config import settings
from synthesis.models.cluster import ClusterData
from synthesis.models.cohort import CohortBuilder
from synthesis.models.persona import PersonaV1
from synthesis.models.persona_v2 import PersonaV2

from .digest import DigestError, digest_provider_output
from .groundedness import GroundednessReport, check_groundedness
from .model_backend import LLMResult, ModelBackend
from .prompt_builder import (
    RENDER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_cohort_context,
    build_messages,
    build_retry_messages,
    build_tool_definition,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
COST_SAFETY_MULTIPLIER = 3.0
DEBUG_ARTIFACT_DIR = Path("output") / "synthesis_failures"


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
    debug_artifact_path: str | None = None
    cost_usd: float = 0.0
    success: bool = False


@dataclass
class SynthesisResult:
    persona: BaseModel
    groundedness: GroundednessReport
    total_cost_usd: float
    model_used: str
    attempts: int


async def synthesize(
    cluster: ClusterData,
    backend: ModelBackend,
    max_retries: int = MAX_RETRIES,
) -> SynthesisResult:
    """Synthesize a persona from cluster data with validation and retry.

    Calls the LLM with tool-use forcing, validates with Pydantic, checks
    groundedness, and retries with error context on failure.
    """
    tool = build_tool_definition()
    attempts: list[AttemptRecord] = []
    total_cost = 0.0
    first_attempt_cost: float | None = None
    errors_for_retry: list[str] = []

    for attempt_num in range(1, max_retries + 2):  # +2 because max_retries is retry count
        record = AttemptRecord(attempt=attempt_num)

        # Build messages (with error context on retries)
        if errors_for_retry:
            messages = build_retry_messages(cluster, errors_for_retry)
        else:
            messages = build_messages(cluster)

        # Call the LLM
        llm_result: LLMResult = await backend.generate(
            system=SYSTEM_PROMPT,
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

        raw_output = llm_result.tool_input

        # Digest provider-native output into PersonaV1 shape
        errors_for_retry = []
        try:
            digested_output = digest_provider_output(
                provider=settings.model_provider,
                model=llm_result.model,
                raw_output=raw_output,
            )
            persona = PersonaV1.model_validate(digested_output)
        except DigestError as e:
            record.validation_errors = [str(e), *e.warnings]
            errors_for_retry.extend(record.validation_errors)
            record.debug_artifact_path = _write_debug_artifact(
                cluster=cluster,
                attempt=attempt_num,
                provider=settings.model_provider,
                model=llm_result.model,
                raw_output=raw_output,
                digested_output=None,
                failure_reason="digest",
                errors=record.validation_errors,
            )
            logger.warning(
                "Attempt %d: digest failed with %d errors",
                attempt_num,
                len(record.validation_errors),
            )
            attempts.append(record)
            continue
        except ValidationError as e:
            record.validation_errors = [
                f"{err['loc']}: {err['msg']}" for err in e.errors()
            ]
            errors_for_retry.extend(record.validation_errors)
            record.debug_artifact_path = _write_debug_artifact(
                cluster=cluster,
                attempt=attempt_num,
                provider=settings.model_provider,
                model=llm_result.model,
                raw_output=raw_output,
                digested_output=digested_output,
                failure_reason="validation",
                errors=record.validation_errors,
            )
            logger.warning(
                "Attempt %d: validation failed with %d errors",
                attempt_num,
                len(record.validation_errors),
            )
            attempts.append(record)
            continue

        # Check groundedness
        groundedness = check_groundedness(persona, cluster)
        if not groundedness.passed:
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
        )

    # All attempts exhausted
    raise SynthesisError(
        f"Synthesis failed after {max_retries + 1} attempts",
        attempts=attempts,
    )


async def synthesize_v2(
    cluster: ClusterData,
    backend: ModelBackend,
    *,
    birth_year: int,
    eval_year: int = 2026,
    max_retries: int = MAX_RETRIES,
) -> SynthesisResult:
    """Incremental V2 path: same retry loop, richer schema, cohort-injected prompt."""
    cohort = CohortBuilder().build(birth_year=birth_year, eval_year=eval_year)
    tool = build_tool_definition(PersonaV2)
    attempts: list[AttemptRecord] = []
    total_cost = 0.0
    first_attempt_cost: float | None = None
    errors_for_retry: list[str] = []
    cohort_context = build_cohort_context(cohort)

    for attempt_num in range(1, max_retries + 2):
        record = AttemptRecord(attempt=attempt_num)

        if errors_for_retry:
            messages = build_retry_messages(
                cluster,
                errors_for_retry,
                extra_sections=[cohort_context],
            )
        else:
            messages = build_messages(cluster, extra_sections=[cohort_context])

        llm_result: LLMResult = await backend.generate(
            system=RENDER_SYSTEM_PROMPT,
            messages=messages,
            tool=tool,
        )

        record.cost_usd = llm_result.estimated_cost_usd
        total_cost += record.cost_usd
        if first_attempt_cost is None:
            first_attempt_cost = record.cost_usd

        if total_cost > first_attempt_cost * COST_SAFETY_MULTIPLIER * (max_retries + 1):
            attempts.append(record)
            raise SynthesisError(
                f"Cost safety limit exceeded: ${total_cost:.4f}",
                attempts=attempts,
            )

        raw_output = llm_result.tool_input
        errors_for_retry = []

        try:
            persona = PersonaV2.model_validate(raw_output)
        except ValidationError as e:
            record.validation_errors = [
                f"{err['loc']}: {err['msg']}" for err in e.errors()
            ]
            errors_for_retry.extend(record.validation_errors)
            record.debug_artifact_path = _write_debug_artifact(
                cluster=cluster,
                attempt=attempt_num,
                provider=settings.model_provider,
                model=llm_result.model,
                raw_output=raw_output,
                digested_output=None,
                failure_reason="validation_v2",
                errors=record.validation_errors,
            )
            attempts.append(record)
            continue

        groundedness = check_groundedness(persona, cluster)
        if not groundedness.passed:
            record.groundedness_violations = groundedness.violations
            errors_for_retry.extend(groundedness.violations)
            attempts.append(record)
            continue

        record.success = True
        attempts.append(record)
        return SynthesisResult(
            persona=persona,
            groundedness=groundedness,
            total_cost_usd=total_cost,
            model_used=llm_result.model,
            attempts=attempt_num,
        )

    raise SynthesisError(
        f"Synthesis V2 failed after {max_retries + 1} attempts",
        attempts=attempts,
    )


async def synthesize_for_schema_version(
    cluster: ClusterData,
    backend: ModelBackend,
    *,
    schema_version: str | None = None,
    birth_year: int | None = None,
    eval_year: int | None = None,
    max_retries: int = MAX_RETRIES,
) -> SynthesisResult:
    resolved_schema_version = (schema_version or settings.persona_schema_version).strip().lower()
    if resolved_schema_version == "v1":
        return await synthesize(
            cluster,
            backend,
            max_retries=max_retries,
        )
    if resolved_schema_version == "v2":
        return await synthesize_v2(
            cluster,
            backend,
            birth_year=birth_year or settings.persona_birth_year,
            eval_year=eval_year or settings.persona_eval_year,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported schema version '{resolved_schema_version}'")


def _write_debug_artifact(
    *,
    cluster: ClusterData,
    attempt: int,
    provider: str,
    model: str,
    raw_output: dict,
    digested_output: dict | None,
    failure_reason: str,
    errors: list[str],
) -> str:
    DEBUG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = DEBUG_ARTIFACT_DIR / (
        f"{cluster.cluster_id}_attempt_{attempt:02d}.json"
    )
    artifact = {
        "cluster_id": cluster.cluster_id,
        "attempt": attempt,
        "provider": provider,
        "model": model,
        "failure_reason": failure_reason,
        "validation_errors": errors,
        "raw_output": raw_output,
        "digested_output": digested_output,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, default=str))
    return str(artifact_path)
