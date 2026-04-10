from __future__ import annotations

import json as _json

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1


def _chars_to_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return max(1, len(text) // 4)

SYSTEM_PROMPT = """\
You are a persona synthesis expert. Your job is to analyze behavioral data from a \
customer cluster and produce a single, richly detailed persona that a product marketer \
and a data scientist would both trust.

Quality criteria:
- **Grounded**: Every claim must trace back to specific source records. Use the \
record IDs provided in the data to populate source_evidence entries.
- **Distinctive**: The persona should feel like a real individual, not a generic \
average. Use specific vocabulary, concrete quotes, and sharp motivations.
- **Actionable**: Goals, pains, and objections should be specific enough to inform \
product and marketing decisions.
- **Consistent**: Demographics, firmographics, vocabulary, and quotes should all \
describe the same coherent person.

Evidence rules:
- Each entry in source_evidence must reference at least one record_id from the \
provided sample records.
- The field_path must use dot notation pointing to the persona field the evidence \
supports (e.g. "goals.0", "pains.2", "motivations.1").
- Every item in goals, pains, motivations, and objections MUST have a corresponding \
source_evidence entry.
- Confidence should reflect how directly the data supports the claim (1.0 = verbatim \
from data, 0.5 = reasonable inference).

Example source_evidence entry:
{
  "claim": "Wants to reduce manual data entry by 50%",
  "record_ids": ["rec_0042", "rec_0087"],
  "field_path": "goals.0",
  "confidence": 0.85
}
"""


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


def build_user_message(cluster: ClusterData) -> str:
    """Build the user message containing all cluster data for the LLM."""
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

    sections.append(
        "\nSynthesize a single persona from this data. "
        "Use the create_persona tool to structure your output."
    )

    return "\n".join(sections)


def build_messages(cluster: ClusterData) -> list[dict]:
    """Build the full message list for the Anthropic API call."""
    return [
        {"role": "user", "content": build_user_message(cluster)},
    ]


def build_cached_messages(
    cluster: ClusterData,
    backend: object | None = None,
) -> tuple[list[dict], dict]:
    """Build messages with cache_control markers on static sections.

    Puts the static system prompt content (instructions + tool schema) in a
    content block with cache_control={"type":"ephemeral"} so Anthropic can
    cache the prefix across requests. The dynamic cluster data stays in a
    plain content block with no cache marker.

    Returns:
        (messages, token_counts) where token_counts has keys:
            static_tokens, dynamic_tokens, cacheable_fraction
    """
    # Static content: system instructions + tool schema (identical for every call)
    static_text = SYSTEM_PROMPT
    tool_def = build_tool_definition()
    static_text += "\n\nTool schema:\n" + _json.dumps(tool_def, indent=2)

    # Dynamic content: cluster-specific data (changes per call)
    dynamic_text = build_user_message(cluster)

    static_tokens = _chars_to_tokens(static_text)
    dynamic_tokens = _chars_to_tokens(dynamic_text)
    cacheable_fraction = static_tokens / (static_tokens + dynamic_tokens)

    messages = [
        {
            "role": "user",
            "content": [
                # Static cached block — first so it maximizes prefix length
                {
                    "type": "text",
                    "text": static_text,
                    "cache_control": {"type": "ephemeral"},
                },
                # Dynamic block — cluster-specific, no cache marker
                {
                    "type": "text",
                    "text": dynamic_text,
                },
            ],
        }
    ]

    token_counts = {
        "static_tokens": static_tokens,
        "dynamic_tokens": dynamic_tokens,
        "cacheable_fraction": round(cacheable_fraction, 4),
    }

    return messages, token_counts


def build_retry_messages(
    cluster: ClusterData,
    errors: list[str],
) -> list[dict]:
    """Build messages for a retry attempt, including previous errors."""
    user_msg = build_user_message(cluster)
    error_section = "\n## Previous Attempt Errors\n"
    error_section += "Your previous attempt had these issues. Please fix them:\n"
    for err in errors:
        error_section += f"- {err}\n"
    error_section += "\nPlease try again, addressing all errors above."

    return [
        {"role": "user", "content": error_section + "\n\n" + user_msg},
    ]
