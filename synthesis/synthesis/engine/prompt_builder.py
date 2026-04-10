from __future__ import annotations

import json
from typing import Literal

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# ── Experiment 1.19: schema artifact format ───────────────────────────
# Three ways to render PersonaV1's structure in the system prompt.
# The actual tool schema (input_schema) stays the same across variants —
# only the textual description of the expected output changes.
SchemaFormat = Literal["pydantic", "jsonschema", "typescript"]


def _render_pydantic_schema() -> str:
    """Render PersonaV1 as Pydantic model repr (Python class definition)."""
    return '''\
class Demographics(BaseModel):
    age_range: str               # e.g. '25-34'
    gender_distribution: str     # e.g. 'predominantly female'
    location_signals: list[str]
    education_level: str | None = None
    income_bracket: str | None = None

class Firmographics(BaseModel):
    company_size: str | None = None   # e.g. 'SMB (10-50 employees)'
    industry: str | None = None
    role_titles: list[str] = []
    tech_stack_signals: list[str] = []

class JourneyStage(BaseModel):
    stage: str           # e.g. 'awareness', 'consideration', 'decision'
    mindset: str
    key_actions: list[str]
    content_preferences: list[str]

class PersonaV1(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str
    summary: str                 # 2-3 sentence overview
    demographics: Demographics
    firmographics: Firmographics
    goals: list[str]             # 2-8 items
    pains: list[str]             # 2-8 items
    motivations: list[str]       # 2-6 items
    objections: list[str]        # 1-6 items
    channels: list[str]          # 1-8 items
    vocabulary: list[str]        # 3-15 words/phrases
    decision_triggers: list[str] # 1-6 items
    sample_quotes: list[str]     # 2-5 items
    journey_stages: list[JourneyStage]  # 2-5 stages
    source_evidence: list[SourceEvidence]  # 3+ entries'''


def _render_jsonschema_schema() -> str:
    """Render PersonaV1 as JSON Schema (compact, readable)."""
    schema = PersonaV1.model_json_schema()
    return json.dumps(schema, indent=2)


def _render_typescript_schema() -> str:
    """Render PersonaV1 as TypeScript type definitions."""
    return '''\
interface Demographics {
  age_range: string;               // e.g. '25-34'
  gender_distribution: string;     // e.g. 'predominantly female'
  location_signals: string[];
  education_level?: string | null;
  income_bracket?: string | null;
}

interface Firmographics {
  company_size?: string | null;    // e.g. 'SMB (10-50 employees)'
  industry?: string | null;
  role_titles: string[];
  tech_stack_signals: string[];
}

interface JourneyStage {
  stage: string;           // e.g. 'awareness', 'consideration', 'decision'
  mindset: string;
  key_actions: string[];
  content_preferences: string[];
}

interface SourceEvidence {
  claim: string;
  record_ids: string[];    // min 1
  field_path: string;      // dot notation, e.g. 'goals.0'
  confidence: number;      // 0.0 - 1.0
}

interface PersonaV1 {
  schema_version: "1.0";
  name: string;
  summary: string;                   // 2-3 sentence overview
  demographics: Demographics;
  firmographics: Firmographics;
  goals: string[];                   // 2-8 items
  pains: string[];                   // 2-8 items
  motivations: string[];             // 2-6 items
  objections: string[];              // 1-6 items
  channels: string[];                // 1-8 items
  vocabulary: string[];              // 3-15 words/phrases
  decision_triggers: string[];       // 1-6 items
  sample_quotes: string[];           // 2-5 in their voice
  journey_stages: JourneyStage[];    // 2-5 stages
  source_evidence: SourceEvidence[]; // 3+ entries
}'''


SCHEMA_RENDERERS: dict[SchemaFormat, callable] = {
    "pydantic": _render_pydantic_schema,
    "jsonschema": _render_jsonschema_schema,
    "typescript": _render_typescript_schema,
}

SCHEMA_FORMAT_LABELS: dict[SchemaFormat, str] = {
    "pydantic": "Pydantic model (Python)",
    "jsonschema": "JSON Schema",
    "typescript": "TypeScript interface",
}


def build_system_prompt(schema_format: SchemaFormat | None = None) -> str:
    """Build the system prompt, optionally embedding a schema description.

    Args:
        schema_format: If set, appends the PersonaV1 schema in the given format
            to the system prompt. None = control (no schema in prompt).
    """
    prompt = SYSTEM_PROMPT
    if schema_format is not None:
        renderer = SCHEMA_RENDERERS[schema_format]
        label = SCHEMA_FORMAT_LABELS[schema_format]
        prompt += (
            f"\n\n## Expected output schema ({label})\n"
            f"```\n{renderer()}\n```\n"
            f"Your output MUST conform to this structure."
        )
    return prompt


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
