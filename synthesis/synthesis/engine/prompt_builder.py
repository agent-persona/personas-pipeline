from __future__ import annotations

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# ── Experiment 1.17: per-field token budgets ──────────────────────────
# Base token budgets at multiplier=1.0 (the "50-token" tier).
# Multiply by budget_multiplier to sweep: 0.4x → 20 tok, 1x → 50 tok,
# 4x → 200 tok, None → unbounded (control).
FIELD_BASE_BUDGETS: dict[str, int] = {
    "summary": 50,
    "goals": 30,
    "pains": 30,
    "motivations": 30,
    "objections": 30,
    "channels": 20,
    "vocabulary": 20,
    "decision_triggers": 25,
    "sample_quotes": 40,
}

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


def _apply_length_budgets(
    schema: dict,
    budget_multiplier: float,
) -> dict:
    """Inject maxLength / item-level hints into the JSON schema descriptions."""
    schema = schema.copy()
    props = schema.get("properties", {})
    for field_name, base_tokens in FIELD_BASE_BUDGETS.items():
        if field_name not in props:
            continue
        limit = int(base_tokens * budget_multiplier)
        prop = props[field_name] = props[field_name].copy()
        hint = f" (TARGET: keep each entry under ~{limit} tokens)"
        if "items" in prop:  # list field
            items = prop["items"] = prop["items"].copy()
            items["description"] = items.get("description", "") + hint
        else:
            prop["description"] = prop.get("description", "") + hint
    schema["properties"] = props
    return schema


def build_tool_definition(budget_multiplier: float | None = None) -> dict:
    """Build the Claude tool definition from the PersonaV1 JSON schema.

    Args:
        budget_multiplier: If set, injects per-field token-budget hints into
            the schema descriptions.  Use 0.4 → ~20 tok, 1.0 → ~50 tok,
            4.0 → ~200 tok, None → unbounded (control/default).
    """
    schema = PersonaV1.model_json_schema()
    description = (
        "Create a structured persona from the analyzed cluster data. "
        "All fields are required and must be grounded in the provided source records."
    )
    if budget_multiplier is not None:
        schema = _apply_length_budgets(schema, budget_multiplier)
        description += (
            f" IMPORTANT: Respect the per-field token budgets indicated in "
            f"each field description (multiplier={budget_multiplier})."
        )
    return {
        "name": "create_persona",
        "description": description,
        "input_schema": schema,
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


def build_messages(
    cluster: ClusterData,
    budget_multiplier: float | None = None,
) -> list[dict]:
    """Build the full message list for the Anthropic API call."""
    user_msg = build_user_message(cluster)
    if budget_multiplier is not None:
        user_msg += (
            f"\n\nIMPORTANT: Keep each field value concise — target the token "
            f"budgets noted in the tool schema (multiplier={budget_multiplier})."
        )
    return [
        {"role": "user", "content": user_msg},
    ]


def build_retry_messages(
    cluster: ClusterData,
    errors: list[str],
    budget_multiplier: float | None = None,
) -> list[dict]:
    """Build messages for a retry attempt, including previous errors."""
    user_msg = build_user_message(cluster)
    error_section = "\n## Previous Attempt Errors\n"
    error_section += "Your previous attempt had these issues. Please fix them:\n"
    for err in errors:
        error_section += f"- {err}\n"
    error_section += "\nPlease try again, addressing all errors above."
    if budget_multiplier is not None:
        error_section += (
            f"\nRemember: respect the per-field token budgets "
            f"(multiplier={budget_multiplier})."
        )

    return [
        {"role": "user", "content": error_section + "\n\n" + user_msg},
    ]
