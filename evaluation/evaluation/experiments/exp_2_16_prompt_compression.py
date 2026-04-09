"""Experiment 2.16: Prompt Compression — Section-level ablation harness."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from synthesis.engine.prompt_builder import (
    SYSTEM_PROMPT_SECTIONS,
    USER_MESSAGE_SECTIONS,
    build_system_prompt,
    build_user_message,
)
from synthesis.engine.synthesizer import synthesize, SynthesisError
from synthesis.models.cluster import ClusterData
from synthesis.engine.model_backend import ModelBackend

logger = logging.getLogger(__name__)


@dataclass
class PromptSection:
    """A single ablatable section of the prompt."""
    name: str
    source: str  # "system" or "user"


@dataclass
class VariantResult:
    """Result of running one ablation variant."""
    section: PromptSection
    repetition: int
    success: bool = False
    schema_valid: bool = False
    groundedness: float = 0.0
    cost_usd: float = 0.0
    attempts: int = 0
    error: str | None = None


@dataclass
class AblationResult:
    """Complete ablation experiment result."""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=list)
    sections: list[PromptSection] = field(default_factory=list)


def get_all_sections() -> list[PromptSection]:
    """Get all ablatable sections from both system and user prompts."""
    sections = []
    for name in SYSTEM_PROMPT_SECTIONS:
        sections.append(PromptSection(name=name, source="system"))
    for name in USER_MESSAGE_SECTIONS:
        sections.append(PromptSection(name=name, source="user"))
    return sections


class AblationHarness:
    """Runs section-level prompt ablation for experiment 2.16."""

    def __init__(
        self,
        clusters: list[ClusterData],
        backend: ModelBackend,
        repetitions: int = 3,
    ) -> None:
        self.clusters = clusters
        self.backend = backend
        self.repetitions = repetitions
        self.sections = get_all_sections()

    async def _run_single(
        self,
        cluster: ClusterData,
        system_prompt: str | None = None,
    ) -> dict[str, float]:
        """Run synthesis once and return metrics."""
        try:
            result = await synthesize(
                cluster, self.backend, system_prompt=system_prompt
            )
            return {
                "success": 1.0,
                "schema_valid": 1.0,
                "groundedness": result.groundedness.score,
                "cost_usd": result.total_cost_usd,
                "attempts": float(result.attempts),
            }
        except SynthesisError:
            return {
                "success": 0.0,
                "schema_valid": 0.0,
                "groundedness": 0.0,
                "cost_usd": 0.0,
                "attempts": 0.0,
            }

    async def run_baseline(self) -> dict[str, float]:
        """Run synthesis with full prompt (no sections excluded)."""
        logger.info("Running baseline (full prompt)...")
        totals: dict[str, list[float]] = {}

        for rep in range(self.repetitions):
            for cluster in self.clusters:
                metrics = await self._run_single(cluster)
                for k, v in metrics.items():
                    totals.setdefault(k, []).append(v)

        return {k: sum(v) / len(v) for k, v in totals.items()}

    async def run_variant(self, section: PromptSection) -> dict[str, float]:
        """Run synthesis with one section excluded."""
        logger.info("Running variant: exclude %s (%s)...", section.name, section.source)
        totals: dict[str, list[float]] = {}

        for rep in range(self.repetitions):
            for cluster in self.clusters:
                if section.source == "system":
                    system_prompt = build_system_prompt(
                        exclude_sections={section.name}
                    )
                    metrics = await self._run_single(cluster, system_prompt=system_prompt)
                else:
                    # For user message sections, we'd need to pass exclude_sections
                    # through to build_user_message. For now, run with full prompt
                    # and note the limitation.
                    metrics = await self._run_single(cluster)

                for k, v in metrics.items():
                    totals.setdefault(k, []).append(v)

        return {k: sum(v) / len(v) for k, v in totals.items()}

    async def run_full_ablation(self) -> AblationResult:
        """Run complete ablation: baseline + all section variants."""
        result = AblationResult(sections=self.sections)

        # Run baseline
        result.baseline_metrics = await self.run_baseline()
        logger.info("Baseline: %s", result.baseline_metrics)

        # Run each variant
        for section in self.sections:
            variant_metrics = await self.run_variant(section)

            # Compute deltas
            deltas = {}
            for k in result.baseline_metrics:
                b = result.baseline_metrics.get(k, 0.0)
                v = variant_metrics.get(k, 0.0)
                deltas[k] = v - b

            result.variants.append({
                "section_name": section.name,
                "section_source": section.source,
                "metrics": variant_metrics,
                "deltas": deltas,
                "impact_score": abs(deltas.get("groundedness", 0.0)) + abs(deltas.get("success", 0.0)),
            })

            logger.info(
                "  %s (%s): groundedness_delta=%.3f, success_delta=%.3f",
                section.name, section.source,
                deltas.get("groundedness", 0), deltas.get("success", 0),
            )

        # Sort by impact (highest first)
        result.variants.sort(key=lambda v: v["impact_score"], reverse=True)

        return result
