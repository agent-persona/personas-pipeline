from __future__ import annotations

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

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
- **Negative space**: `not_this` captures identity-level negatives — concrete \
things this persona would NOT do, say, or believe. These are NOT sales \
objections (those go in `objections`). Think "wouldn't be caught dead doing X" \
or "rolls their eyes at Y". Make them specific and rooted in the persona's \
values, role, or habits, not generic ("doesn't like spam").

Psychological extraction rules (communication_style, emotional_profile, \
moral_framework are REQUIRED):
- **communication_style** — Infer tone, formality, and vocabulary_level from \
verbatim support messages and the technical register of their language. \
preferred_channels must reflect where this cluster actually communicates.
- **emotional_profile** — baseline_mood is the dominant tone across their \
messages. stress_triggers are specific situations that produce complaint \
messages or long error-filled sessions. coping_mechanisms are observable \
behaviors they take when stressed.
- **moral_framework** — core_values are what they repeatedly advocate for in \
their own words (2-6). ethical_stance is your best-fit classification. \
moral_foundations weights the six MFT foundations (care, fairness, loyalty, \
authority, sanctity, liberty) in [0.0, 1.0] — only include foundations with \
clear evidence; omit rather than guess.
- Each of communication_style, emotional_profile, and moral_framework must \
have at least one source_evidence entry with a field_path rooted in that \
sub-object.

Evidence rules:
- Each entry in source_evidence must reference at least one record_id from the \
provided sample records.
- The field_path must use dot notation pointing to the persona field the evidence \
supports (e.g. "goals.0", "pains.2", "motivations.1").
- Every item in goals, pains, motivations, and objections MUST have a corresponding \
source_evidence entry.
- Do NOT create source_evidence entries for `not_this` — that field is \
identity-shaped and does not require citation.
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


def build_tool_definition(schema_cls: type = PersonaV1) -> dict:
    """Build the Claude tool definition from a persona schema class.

    exp-2.07: pass `PersonaV1VoiceFirst` to run the voice-first field-order
    variant. The schema class determines the field order Claude emits during
    tool-use structured output, since Pydantic v2 preserves class-level
    declaration order in model_json_schema().
    """
    return {
        "name": "create_persona",
        "description": (
            "Create a structured persona from the analyzed cluster data. "
            "All fields are required and must be grounded in the provided source records."
        ),
        "input_schema": schema_cls.model_json_schema(),
    }


def build_user_message(
    cluster: ClusterData,
    existing_personas: list[dict] | None = None,
) -> str:
    """Build the user message containing all cluster data for the LLM.

    When `existing_personas` is provided (list of persona dicts with at least
    name, summary, goals, pains, vocabulary), a contrast block is injected
    instructing the LLM to differentiate the new persona on specific axes.
    """
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

    # Contrast block: inject existing personas so the LLM differentiates
    if existing_personas:
        sections.append("\n## Existing Personas — You MUST Differ From These")
        sections.append(
            "The following personas have already been created for this tenant. "
            "Your new persona must be **clearly distinguishable** from each of them. "
            "Specifically:"
        )
        sections.append(
            "- **Goals**: Do not repeat their goals. Find the behavioral niche they don't cover."
        )
        sections.append(
            "- **Pains**: Identify frictions unique to this cluster's data, not already captured."
        )
        sections.append(
            "- **Vocabulary**: Use different terminology. If they say 'pipeline', you find another word."
        )
        sections.append(
            "- **Voice**: Sample quotes must sound like a different person — different cadence, "
            "different concerns, different attitude."
        )
        sections.append(
            "- **Still grounded**: Distinctiveness does NOT mean inventing claims. "
            "Every goal, pain, and motivation must still trace to record IDs from this cluster's data."
        )
        for i, p in enumerate(existing_personas):
            sections.append(f"\n### Existing Persona {i + 1}: {p.get('name', 'Unknown')}")
            if p.get("summary"):
                sections.append(f"- Summary: {p['summary']}")
            if p.get("goals"):
                sections.append(f"- Goals: {', '.join(p['goals'][:4])}")
            if p.get("pains"):
                sections.append(f"- Pains: {', '.join(p['pains'][:4])}")
            if p.get("vocabulary"):
                sections.append(f"- Vocabulary: {', '.join(p['vocabulary'][:8])}")

    sections.append(
        "\nSynthesize a single persona from this data. "
        "Use the create_persona tool to structure your output."
    )

    return "\n".join(sections)


def build_messages(
    cluster: ClusterData,
    existing_personas: list[dict] | None = None,
) -> list[dict]:
    """Build the full message list for the Anthropic API call."""
    return [
        {"role": "user", "content": build_user_message(cluster, existing_personas=existing_personas)},
    ]


def build_retry_messages(
    cluster: ClusterData,
    errors: list[str],
    existing_personas: list[dict] | None = None,
) -> list[dict]:
    """Build messages for a retry attempt, including previous errors."""
    user_msg = build_user_message(cluster, existing_personas=existing_personas)
    error_section = "\n## Previous Attempt Errors\n"
    error_section += "Your previous attempt had these issues. Please fix them:\n"
    for err in errors:
        error_section += f"- {err}\n"
    error_section += "\nPlease try again, addressing all errors above."

    return [
        {"role": "user", "content": error_section + "\n\n" + user_msg},
    ]
