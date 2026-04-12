from __future__ import annotations

import json
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

_FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "golden_examples.json"
_GOLDEN_EXAMPLES_CACHE: list[PersonaV1] | None = None


def _load_golden_examples() -> list[PersonaV1]:
    """Load and validate the hand-curated exemplar personas. Cached after first call."""
    global _GOLDEN_EXAMPLES_CACHE
    if _GOLDEN_EXAMPLES_CACHE is None:
        with open(_FIXTURES_PATH) as f:
            raw = json.load(f)
        _GOLDEN_EXAMPLES_CACHE = [PersonaV1.model_validate(d) for d in raw]
    return _GOLDEN_EXAMPLES_CACHE


def _render_exemplar_prose(p: PersonaV1) -> str:
    """Render one PersonaV1 as compact field-by-field prose.

    Deliberately NOT JSON to reduce schema-echoing / field-value copying
    pressure. source_evidence is omitted — the model should never see the
    fake EXAMPLE_rec_* IDs; any appearance of those in synthesis output is
    unambiguous cloning.
    """
    parts = [
        f"**{p.name}**",
        f"Summary: {p.summary}",
        f"Demographics: {p.demographics.age_range}, {p.demographics.gender_distribution}; "
        f"locations: {', '.join(p.demographics.location_signals)}",
        f"Role: {', '.join(p.firmographics.role_titles)}"
        + (f" ({p.firmographics.industry})" if p.firmographics.industry else ""),
        f"Goals: {'; '.join(p.goals)}",
        f"Pains: {'; '.join(p.pains)}",
        f"Motivations: {'; '.join(p.motivations)}",
        f"Objections: {'; '.join(p.objections)}",
        f"Channels: {', '.join(p.channels)}",
        f"Vocabulary: {', '.join(p.vocabulary)}",
        "Sample quotes: " + " | ".join(f'"{q}"' for q in p.sample_quotes),
    ]
    return "\n".join(parts)


_FEW_SHOT_HEADER = """\

## Example personas (style reference)

The following are hand-curated example personas from different industries. \
They show the level of specificity, vocabulary distinctiveness, and field \
completeness expected in a good persona. **Do NOT copy their content.** These \
are from industries unrelated to the tenant below; your job is to produce a \
persona grounded in the tenant's actual records, at a comparable level of \
craft.

"""


def build_system_prompt(few_shot_count: int = 0) -> str:
    """Return the synthesis system prompt, optionally with N few-shot exemplars.

    At few_shot_count=0, returns SYSTEM_PROMPT unchanged (byte-equal). At
    few_shot_count>0, appends a `## Example personas` section with the first N
    exemplars from the golden_examples fixture rendered as prose.
    """
    if few_shot_count <= 0:
        return SYSTEM_PROMPT
    exemplars = _load_golden_examples()[:few_shot_count]
    rendered = "\n\n---\n\n".join(_render_exemplar_prose(p) for p in exemplars)
    return SYSTEM_PROMPT + _FEW_SHOT_HEADER + rendered + "\n"


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
