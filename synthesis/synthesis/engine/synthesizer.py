from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from .groundedness import GroundednessReport, check_groundedness
from .model_backend import LLMResult, ModelBackend
from .prompt_builder import (
    SYSTEM_PROMPT,
    build_messages,
    build_retry_messages,
    build_tool_definition,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
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
    persona: PersonaV1
    groundedness: GroundednessReport
    total_cost_usd: float
    model_used: str
    attempts: int
    degraded: bool = False


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

        # Validate with Pydantic
        errors_for_retry = []
        try:
            persona = PersonaV1.model_validate(llm_result.tool_input)
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

        # Check groundedness
        groundedness = check_groundedness(persona, cluster)
        if not groundedness.passed:
            record.groundedness_violations = groundedness.violations
            errors_for_retry.extend(groundedness.violations)
            logger.warning(
                "Attempt %d: groundedness check failed (score=%.2f, threshold=%.2f, %d violations)",
                attempt_num,
                groundedness.score,
                groundedness.threshold,
                len(groundedness.violations),
            )
            attempts.append(record)
            continue

        # Success (possibly degraded)
        record.success = True
        attempts.append(record)
        if groundedness.degraded:
            logger.info(
                "Synthesis succeeded DEGRADED on attempt %d "
                "(cost=$%.4f, groundedness=%.2f, threshold=%.2f)",
                attempt_num,
                total_cost,
                groundedness.score,
                groundedness.threshold,
            )
        else:
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
            degraded=groundedness.degraded,
        )

    # All attempts exhausted
    raise SynthesisError(
        f"Synthesis failed after {max_retries + 1} attempts",
        attempts=attempts,
    )
