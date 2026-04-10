from __future__ import annotations

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# ── Layout A (baseline) ─────────────────────────────────────────────────────
# Full system prompt: preamble + quality criteria + evidence rules + example
# User message: cluster data only
# ── Layout B ────────────────────────────────────────────────────────────────
# Minimal system prompt: role declaration only
# User message: all instructions + evidence rules + cluster data
# ── Layout C ────────────────────────────────────────────────────────────────
# System prompt: preamble + quality criteria only (no evidence rules, no example)
# User message: cluster data + inline evidence reminder at end

_PREAMBLE = """\
You are a persona synthesis expert. Your job is to analyze behavioral data from a \
customer cluster and produce a single, richly detailed persona that a product marketer \
and a data scientist would both trust.\
"""

_QUALITY_CRITERIA = """\
Quality criteria:
- **Grounded**: Every claim must trace back to specific source records. Use the \
record IDs provided in the data to populate source_evidence entries.
- **Distinctive**: The persona should feel like a real individual, not a generic \
average. Use specific vocabulary, concrete quotes, and sharp motivations.
- **Actionable**: Goals, pains, and objections should be specific enough to inform \
product and marketing decisions.
- **Consistent**: Demographics, firmographics, vocabulary, and quotes should all \
describe the same coherent person.\
"""

_EVIDENCE_RULES = """\
Evidence rules:
- Each entry in source_evidence must reference at least one record_id from the \
provided sample records.
- The field_path must use dot notation pointing to the persona field the evidence \
supports (e.g. "goals.0", "pains.2", "motivations.1").
- Every item in goals, pains, motivations, and objections MUST have a corresponding \
source_evidence entry.
- Confidence should reflect how directly the data supports the claim (1.0 = verbatim \
from data, 0.5 = reasonable inference).\
"""

_EVIDENCE_EXAMPLE = """\
Example source_evidence entry:
{
  "claim": "Wants to reduce manual data entry by 50%",
  "record_ids": ["rec_0042", "rec_0087"],
  "field_path": "goals.0",
  "confidence": 0.85
}\
"""

# Layout A system prompt (preserved exactly for backward-compat)
SYSTEM_PROMPT = f"""\
{_PREAMBLE}

{_QUALITY_CRITERIA}

{_EVIDENCE_RULES}

{_EVIDENCE_EXAMPLE}
"""


def build_system_prompt(layout: str = "A") -> str:
    """Return the system prompt for the given layout.

    Layout A (default): full prompt — preamble + quality criteria + evidence rules + example.
    Layout B: role declaration only — all instructions moved to user message.
    Layout C: preamble + quality criteria only — no evidence rules or example.
    """
    if layout == "A":
        return SYSTEM_PROMPT
    if layout == "B":
        return "You are a persona synthesis expert."
    if layout == "C":
        return f"{_PREAMBLE}\n\n{_QUALITY_CRITERIA}\n"
    raise ValueError(f"Unknown layout: {layout!r}. Must be 'A', 'B', or 'C'.")


def build_tool_definition() -> dict:
    """Build the Claude tool definition from the PersonaV1 JSON schema."""
    return {
        "name": "create_persona",
        "description": (
            "Create a structured persona from the analyzed cluster data. "
            "All fields are required and must be grounded in the provided source records."
        ),
        "input_schema": PersonaV1.model_json_schema(),
    }


def _build_cluster_sections(cluster: ClusterData) -> list[str]:
    """Build the cluster data sections (shared across all layouts)."""
    sections: list[str] = []

    # Tenant context
    sections.append("## Tenant Context")
    sections.append(f"- Tenant ID: {cluster.tenant.tenant_id}")
    if cluster.tenant.industry:
        sections.append(f"- Industry: {cluster.tenant.industry}")
    if cluster.tenant.product_description:
        sections.append(f"- Product: {cluster.tenant.product_description}")
    if cluster.tenant.existing_persona_names:
        names = ", ".join(cluster.tenant.existing_persona_names)
        sections.append(
            f"- Existing personas (avoid overlap): {names}"
        )

    # Cluster summary
    sections.append("\n## Cluster Summary")
    sections.append(f"- Cluster ID: {cluster.cluster_id}")
    sections.append(f"- Cluster size: {cluster.summary.cluster_size} records")
    if cluster.summary.top_behaviors:
        sections.append(
            f"- Top behaviors: {', '.join(cluster.summary.top_behaviors)}"
        )
    if cluster.summary.top_pages:
        sections.append(f"- Top pages: {', '.join(cluster.summary.top_pages)}")
    if cluster.summary.conversion_rate is not None:
        sections.append(
            f"- Conversion rate: {cluster.summary.conversion_rate:.1%}"
        )
    if cluster.summary.avg_session_duration_seconds is not None:
        sections.append(
            f"- Avg session duration: {cluster.summary.avg_session_duration_seconds:.0f}s"
        )
    if cluster.summary.top_referrers:
        sections.append(
            f"- Top referrers: {', '.join(cluster.summary.top_referrers)}"
        )
    if cluster.summary.extra:
        for k, v in cluster.summary.extra.items():
            sections.append(f"- {k}: {v}")

    # Sample records
    sections.append("\n## Sample Records")
    for rec in cluster.sample_records:
        sections.append(
            f"- **{rec.record_id}** (source: {rec.source})"
        )
        if rec.timestamp:
            sections.append(f"  - timestamp: {rec.timestamp}")
        if rec.payload:
            for k, v in rec.payload.items():
                sections.append(f"  - {k}: {v}")

    # Enrichment
    if cluster.enrichment.firmographic or cluster.enrichment.intent_signals:
        sections.append("\n## Enrichment Data")
        if cluster.enrichment.firmographic:
            sections.append("### Firmographic")
            for k, v in cluster.enrichment.firmographic.items():
                sections.append(f"- {k}: {v}")
        if cluster.enrichment.intent_signals:
            sections.append("### Intent Signals")
            for signal in cluster.enrichment.intent_signals:
                sections.append(f"- {signal}")
        if cluster.enrichment.technographic:
            sections.append("### Technographic")
            for k, v in cluster.enrichment.technographic.items():
                sections.append(f"- {k}: {v}")

    # Available record IDs (for evidence citing)
    sections.append("\n## Available Record IDs")
    sections.append(
        "Use these IDs in source_evidence.record_ids: "
        + ", ".join(cluster.all_record_ids)
    )

    return sections


def build_user_message(cluster: ClusterData, layout: str = "A") -> str:
    """Build the user message for the given layout.

    Layout A (default): cluster data only (instructions are in system prompt).
    Layout B: quality criteria + evidence rules + evidence example + cluster data.
    Layout C: cluster data + brief inline evidence reminder at end.
    """
    sections: list[str] = []

    if layout == "B":
        # All instructions move here since system prompt has role only
        sections.append(_QUALITY_CRITERIA)
        sections.append("")
        sections.append(_EVIDENCE_RULES)
        sections.append("")
        sections.append(_EVIDENCE_EXAMPLE)
        sections.append("")

    sections.extend(_build_cluster_sections(cluster))

    if layout == "C":
        # Inline evidence reminder at end
        sections.append(
            "\n**Evidence reminder**: Every item in goals, pains, motivations, and "
            "objections must have a source_evidence entry referencing a valid record_id "
            "from the Available Record IDs list above."
        )

    sections.append(
        "\nSynthesize a single persona from this data. "
        "Use the create_persona tool to structure your output."
    )

    return "\n".join(sections)


def build_messages(cluster: ClusterData, layout: str = "A") -> list[dict]:
    """Build the full message list for the Anthropic API call."""
    return [
        {"role": "user", "content": build_user_message(cluster, layout=layout)},
    ]


def build_retry_messages(
    cluster: ClusterData,
    errors: list[str],
    layout: str = "A",
) -> list[dict]:
    """Build messages for a retry attempt, including previous errors."""
    user_msg = build_user_message(cluster, layout=layout)
    error_section = "\n## Previous Attempt Errors\n"
    error_section += "Your previous attempt had these issues. Please fix them:\n"
    for err in errors:
        error_section += f"- {err}\n"
    error_section += "\nPlease try again, addressing all errors above."

    return [
        {"role": "user", "content": error_section + "\n\n" + user_msg},
    ]
