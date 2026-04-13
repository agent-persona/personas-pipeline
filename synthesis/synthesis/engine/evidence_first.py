"""Experiment 3.04: Evidence-first persona generation.

Two-pass approach that reverses the standard synthesis pipeline:

Pass 1 — **Evidence Selection**: The LLM examines all cluster data and
selects the 20-30 most representative records. For each, it extracts a
verbatim quote or behavioral signal and explains its relevance.

Pass 2 — **Conditioned Synthesis**: The LLM writes a full PersonaV1
conditioned ONLY on the pre-selected evidence package. It cannot see
the raw cluster data — only the curated evidence brief.

This forced grounding sequence prevents the model from generating
demographics or claims before establishing facts, and anchors the
persona's voice in authentic customer language.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from .groundedness import GroundednessReport, check_groundedness
from .model_backend import AnthropicBackend, LLMResult
from .prompt_builder import SYSTEM_PROMPT, build_tool_definition, build_user_message

logger = logging.getLogger(__name__)


# ── Pass 1: Evidence selection ───────────────────────────────────────

EVIDENCE_SELECTOR_PROMPT = """\
You are an evidence analyst preparing a grounding package for persona \
synthesis. Your job is to select the most representative records from \
the cluster data and extract verbatim quotes or behavioral signals.

Rules:
- Select 10-25 of the most informative records.
- For each record, extract a specific quote, behavior, or data point \
  that reveals something about this customer segment.
- Explain WHY each piece of evidence matters for persona synthesis.
- Prioritize diversity: cover goals, pains, demographics, technology \
  preferences, and communication style.
- Use the exact record IDs provided.

Your output will be the ONLY input the persona synthesizer sees — anything \
you omit will not appear in the persona. Be thorough.
"""

EVIDENCE_SELECTION_TOOL = {
    "name": "select_evidence",
    "description": (
        "Select representative records and extract evidence for persona synthesis. "
        "This curated evidence package will be the sole input to the synthesizer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "string",
                            "description": "The record ID from the cluster data",
                        },
                        "source": {
                            "type": "string",
                            "description": "Data source (ga4, hubspot, intercom, etc.)",
                        },
                        "extracted_signal": {
                            "type": "string",
                            "description": "Verbatim quote or specific behavioral signal",
                        },
                        "relevance": {
                            "type": "string",
                            "description": "Why this evidence matters for persona synthesis",
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "goal", "pain", "motivation", "objection",
                                "demographic", "firmographic", "behavior",
                                "communication", "vocabulary", "other",
                            ],
                            "description": "What aspect of the persona this evidence informs",
                        },
                    },
                    "required": ["record_id", "source", "extracted_signal", "relevance", "category"],
                },
                "minItems": 10,
                "maxItems": 25,
                "description": "Curated evidence records",
            },
            "cluster_theme": {
                "type": "string",
                "description": "1-2 sentence summary of the cluster's defining theme",
            },
        },
        "required": ["selected_records", "cluster_theme"],
    },
}


# ── Pass 2: Conditioned synthesis prompt ─────────────────────────────

CONDITIONED_SYSTEM_PROMPT = """\
You are a persona synthesis expert. Your job is to write a detailed persona \
based EXCLUSIVELY on the pre-selected evidence package below. You must NOT \
invent any claims, demographics, or quotes that are not directly supported \
by the evidence.

Quality criteria:
- **Grounded**: EVERY claim must come from the evidence package. If the \
  evidence doesn't mention something, leave it vague or generic — do NOT \
  hallucinate specifics.
- **Distinctive**: Use the verbatim quotes and vocabulary from the evidence \
  to create a vivid, specific persona voice.
- **Actionable**: Goals, pains, and objections must be traceable to specific \
  evidence entries.
- **Consistent**: All fields must describe the same coherent person.

Evidence rules:
- Each source_evidence entry must reference record IDs from the evidence \
  package (listed below).
- Confidence should be high (0.85-1.0) for claims with direct verbatim \
  support, and lower (0.5-0.7) for reasonable inferences.
- Every item in goals, pains, motivations, and objections MUST have a \
  corresponding source_evidence entry.

CRITICAL: Do NOT fabricate demographics (age, gender, location) unless the \
evidence package contains demographic signals. If no demographic data exists, \
use generic ranges like "unknown" or "not specified".
"""


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class SelectedRecord:
    record_id: str
    source: str
    extracted_signal: str
    relevance: str
    category: str


@dataclass
class EvidencePackage:
    selected_records: list[SelectedRecord]
    cluster_theme: str
    record_ids: list[str]
    selection_cost_usd: float = 0.0


@dataclass
class EvidenceFirstResult:
    """Result of a two-pass evidence-first synthesis."""
    persona: PersonaV1
    groundedness: GroundednessReport
    evidence_package: EvidencePackage
    total_cost_usd: float
    selection_cost_usd: float
    synthesis_cost_usd: float
    model_used: str
    synthesis_attempts: int


# ── Pass 1: Select evidence ─────────────────────────────────────────

async def select_evidence(
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
) -> EvidencePackage:
    """Pass 1: Select representative records and extract evidence."""
    user_msg = build_user_message(cluster)

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=EVIDENCE_SELECTOR_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        tools=[EVIDENCE_SELECTION_TOOL],
        tool_choice={"type": "tool", "name": "select_evidence"},
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = tool_block.input

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    elif "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    records = [
        SelectedRecord(
            record_id=r["record_id"],
            source=r["source"],
            extracted_signal=r["extracted_signal"],
            relevance=r["relevance"],
            category=r["category"],
        )
        for r in result.get("selected_records", [])
    ]

    return EvidencePackage(
        selected_records=records,
        cluster_theme=result.get("cluster_theme", ""),
        record_ids=[r.record_id for r in records],
        selection_cost_usd=cost,
    )


# ── Pass 2: Conditioned synthesis ────────────────────────────────────

def _build_evidence_brief(
    package: EvidencePackage,
    cluster: ClusterData,
) -> str:
    """Build the user message for conditioned synthesis from evidence only."""
    sections = []

    sections.append("## Cluster Theme")
    sections.append(package.cluster_theme)

    sections.append("\n## Tenant Context")
    sections.append(f"- Industry: {cluster.tenant.industry}")
    sections.append(f"- Product: {cluster.tenant.product_description}")

    sections.append(f"\n## Pre-Selected Evidence ({len(package.selected_records)} records)")
    sections.append("These are the ONLY data points you may use. Do NOT invent additional claims.\n")

    by_category: dict[str, list[SelectedRecord]] = {}
    for r in package.selected_records:
        by_category.setdefault(r.category, []).append(r)

    for cat, records in sorted(by_category.items()):
        sections.append(f"### {cat.title()} signals")
        for r in records:
            sections.append(f"- **{r.record_id}** ({r.source}): {r.extracted_signal}")
            sections.append(f"  _Relevance: {r.relevance}_")

    sections.append("\n## Available Record IDs")
    sections.append(
        "Use these IDs in source_evidence.record_ids: "
        + ", ".join(package.record_ids)
    )

    sections.append(
        "\nSynthesize a single persona from ONLY the evidence above. "
        "Use the create_persona tool to structure your output."
    )

    return "\n".join(sections)


async def synthesize_from_evidence(
    cluster: ClusterData,
    package: EvidencePackage,
    client: AsyncAnthropic,
    model: str,
    max_retries: int = 2,
) -> EvidenceFirstResult:
    """Pass 2: Synthesize a persona conditioned on pre-selected evidence."""
    tool = build_tool_definition()
    evidence_brief = _build_evidence_brief(package, cluster)

    total_cost = package.selection_cost_usd
    attempts = 0
    errors_for_retry: list[str] = []

    for attempt_num in range(1, max_retries + 2):
        attempts = attempt_num

        if errors_for_retry:
            user_content = (
                "## Previous Attempt Errors\n"
                + "".join(f"- {e}\n" for e in errors_for_retry)
                + "\nPlease fix and try again.\n\n"
                + evidence_brief
            )
        else:
            user_content = evidence_brief

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=CONDITIONED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "create_persona"},
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")

        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        if "opus" in model:
            cost = (in_tok * 15 + out_tok * 75) / 1_000_000
        elif "haiku" in model:
            cost = (in_tok * 1 + out_tok * 5) / 1_000_000
        else:
            cost = (in_tok * 3 + out_tok * 15) / 1_000_000
        total_cost += cost

        errors_for_retry = []
        try:
            persona = PersonaV1.model_validate(tool_block.input)
        except ValidationError as e:
            errors_for_retry = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            logger.warning("Evidence-first attempt %d: validation failed", attempt_num)
            continue

        groundedness = check_groundedness(persona, cluster)
        if not groundedness.passed:
            errors_for_retry = groundedness.violations
            logger.warning(
                "Evidence-first attempt %d: groundedness %.2f",
                attempt_num, groundedness.score,
            )
            continue

        return EvidenceFirstResult(
            persona=persona,
            groundedness=groundedness,
            evidence_package=package,
            total_cost_usd=total_cost,
            selection_cost_usd=package.selection_cost_usd,
            synthesis_cost_usd=total_cost - package.selection_cost_usd,
            model_used=model,
            synthesis_attempts=attempts,
        )

    raise RuntimeError(f"Evidence-first synthesis failed after {max_retries + 1} attempts")


# ── Full pipeline ────────────────────────────────────────────────────

async def evidence_first_synthesize(
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
    max_retries: int = 2,
) -> EvidenceFirstResult:
    """Full two-pass evidence-first synthesis."""
    package = await select_evidence(cluster, client, model)
    return await synthesize_from_evidence(cluster, package, client, model, max_retries)
