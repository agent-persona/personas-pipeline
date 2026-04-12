from __future__ import annotations

import json as _json
from pathlib import Path

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

_SYNTHETIC_PRIORS_DIR = Path(__file__).parent.parent / "fixtures" / "synthetic_priors"


def _load_synthetic_priors() -> list[dict]:
    """Load all synthetic prior persona files."""
    priors = []
    if _SYNTHETIC_PRIORS_DIR.exists():
        for f in sorted(_SYNTHETIC_PRIORS_DIR.glob("*.json")):
            with open(f) as fh:
                priors.append(_json.load(fh))
    return priors


def _render_prior(prior: dict) -> str:
    """Render one synthetic prior as compact prose."""
    parts = [
        f"**{prior['archetype']}** ({prior.get('industry', 'unknown')})",
        prior.get("summary", ""),
        f"Goals: {'; '.join(prior.get('goals', []))}",
        f"Pains: {'; '.join(prior.get('pains', []))}",
        f"Vocabulary: {', '.join(prior.get('vocabulary', []))}",
        f"Quotes: " + " | ".join(f'"{q}"' for q in prior.get("sample_quotes", [])),
    ]
    return "\n".join(parts)


def build_warmstart_system_prompt(few_shot_count: int = 0) -> str:
    """Return SYSTEM_PROMPT with synthetic priors prepended as contextual warmstart.

    The priors are from a DIFFERENT industry than the test tenant, providing
    structural examples of what good personas look like without leaking
    domain-specific content.
    """
    priors = _load_synthetic_priors()
    if not priors:
        return SYSTEM_PROMPT

    warmstart = "## Synthetic Context (reference personas from other industries)\n\n"
    warmstart += "The following are example personas from unrelated industries. They show the "
    warmstart += "level of specificity and structural completeness expected. **Do NOT copy their "
    warmstart += "content** — use them only as style/depth references.\n\n"
    warmstart += "\n\n---\n\n".join(_render_prior(p) for p in priors)
    warmstart += "\n\n---\n\n"

    return warmstart + SYSTEM_PROMPT


def build_tool_definition(schema_cls: type = PersonaV1) -> dict:
    """Build the Claude tool definition from a persona schema's JSON schema."""
    return {
        "name": "create_persona",
        "description": (
            "Create a structured persona from the analyzed cluster data. "
            "All fields are required and must be grounded in the provided source records."
        ),
        "input_schema": schema_cls.model_json_schema(),
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
