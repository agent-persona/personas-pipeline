from __future__ import annotations

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1


def build_system_prompt(
    *,
    strip_vocabulary: bool = False,
    strip_quotes: bool = False,
) -> str:
    """Assemble the synthesis system prompt.

    When `strip_vocabulary` or `strip_quotes` is set, the corresponding
    lines in the "Distinctive" / "Consistent" quality criteria are rewritten
    so the model is not told to produce a field it will not be allowed to
    emit. Experiment 1.3 (vocabulary anchoring ablation) exercises this.
    """
    distinctive_tail_options = {
        (False, False): "Use specific vocabulary, concrete quotes, and sharp motivations.",
        (True, False): "Use concrete quotes and sharp motivations.",
        (False, True): "Use specific vocabulary and sharp motivations.",
        (True, True): "Use sharp motivations and concrete details.",
    }
    consistent_tail_options = {
        (False, False): "Demographics, firmographics, vocabulary, and quotes should all describe the same coherent person.",
        (True, False): "Demographics, firmographics, and quotes should all describe the same coherent person.",
        (False, True): "Demographics, firmographics, and vocabulary should all describe the same coherent person.",
        (True, True): "Demographics, firmographics, goals, and pains should all describe the same coherent person.",
    }
    key = (strip_vocabulary, strip_quotes)

    return (
        "You are a persona synthesis expert. Your job is to analyze behavioral data from a "
        "customer cluster and produce a single, richly detailed persona that a product marketer "
        "and a data scientist would both trust.\n"
        "\n"
        "Quality criteria:\n"
        "- **Grounded**: Every claim must trace back to specific source records. Use the "
        "record IDs provided in the data to populate source_evidence entries.\n"
        f"- **Distinctive**: The persona should feel like a real individual, not a generic "
        f"average. {distinctive_tail_options[key]}\n"
        "- **Actionable**: Goals, pains, and objections should be specific enough to inform "
        "product and marketing decisions.\n"
        f"- **Consistent**: {consistent_tail_options[key]}\n"
        "\n"
        "Evidence rules:\n"
        "- Each entry in source_evidence must reference at least one record_id from the "
        "provided sample records.\n"
        "- The field_path must use dot notation pointing to the persona field the evidence "
        'supports (e.g. "goals.0", "pains.2", "motivations.1").\n'
        "- Every item in goals, pains, motivations, and objections MUST have a corresponding "
        "source_evidence entry.\n"
        "- Confidence should reflect how directly the data supports the claim (1.0 = verbatim "
        "from data, 0.5 = reasonable inference).\n"
        "\n"
        "Example source_evidence entry:\n"
        "{\n"
        '  "claim": "Wants to reduce manual data entry by 50%",\n'
        '  "record_ids": ["rec_0042", "rec_0087"],\n'
        '  "field_path": "goals.0",\n'
        '  "confidence": 0.85\n'
        "}\n"
    )


# Backwards-compatible constant for any caller that still imports the name.
SYSTEM_PROMPT = build_system_prompt()


def build_tool_definition(
    *,
    strip_vocabulary: bool = False,
    strip_quotes: bool = False,
) -> dict:
    """Build the Claude tool definition from the PersonaV1 JSON schema.

    If `strip_vocabulary` or `strip_quotes` is True, that field is removed
    from the tool's input schema (both `properties` and `required`). This
    lets experiment 1.3 ablate the "voice anchor" fields at call time
    without touching `PersonaV1` itself.
    """
    schema = PersonaV1.model_json_schema()
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    if strip_vocabulary:
        properties.pop("vocabulary", None)
        if "vocabulary" in required:
            required.remove("vocabulary")
    if strip_quotes:
        properties.pop("sample_quotes", None)
        if "sample_quotes" in required:
            required.remove("sample_quotes")

    return {
        "name": "create_persona",
        "description": (
            "Create a structured persona from the analyzed cluster data. "
            "All fields are required and must be grounded in the provided source records."
        ),
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
    *,
    strip_vocabulary: bool = False,
    strip_quotes: bool = False,
) -> list[dict]:
    """Build the full message list for the Anthropic API call.

    The `strip_*` kwargs are accepted for symmetry with `build_tool_definition`
    and `build_system_prompt`; they don't affect the user message (cluster
    data is identical across conditions) but keeping the signature parallel
    makes the call sites in `synthesize()` uniform.
    """
    del strip_vocabulary, strip_quotes  # no effect on user message
    return [
        {"role": "user", "content": build_user_message(cluster)},
    ]


def build_retry_messages(
    cluster: ClusterData,
    errors: list[str],
    *,
    strip_vocabulary: bool = False,
    strip_quotes: bool = False,
) -> list[dict]:
    """Build messages for a retry attempt, including previous errors."""
    del strip_vocabulary, strip_quotes  # no effect on user message
    user_msg = build_user_message(cluster)
    error_section = "\n## Previous Attempt Errors\n"
    error_section += "Your previous attempt had these issues. Please fix them:\n"
    for err in errors:
        error_section += f"- {err}\n"
    error_section += "\nPlease try again, addressing all errors above."

    return [
        {"role": "user", "content": error_section + "\n\n" + user_msg},
    ]
