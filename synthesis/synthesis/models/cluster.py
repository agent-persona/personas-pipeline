from __future__ import annotations

from pydantic import BaseModel, Field


class SampleRecord(BaseModel):
    """A single behavioral record from the cluster, with its source ID."""

    record_id: str
    source: str = Field(description="e.g. 'ga4', 'hubspot', 'linkedin'")
    timestamp: str | None = None
    payload: dict = Field(
        default_factory=dict,
        description="Arbitrary source-specific data",
    )


class ClusterSummary(BaseModel):
    """Aggregate statistics about the cluster."""

    cluster_size: int = Field(ge=1)
    top_behaviors: list[str] = Field(default_factory=list)
    top_pages: list[str] = Field(default_factory=list)
    conversion_rate: float | None = None
    avg_session_duration_seconds: float | None = None
    top_referrers: list[str] = Field(default_factory=list)
    extra: dict = Field(
        default_factory=dict,
        description="Additional cluster-level stats",
    )


class EnrichmentPayload(BaseModel):
    """Third-party enrichment data (firmographic lookups, intent signals, etc.)."""

    firmographic: dict = Field(default_factory=dict)
    intent_signals: list[str] = Field(default_factory=list)
    technographic: dict = Field(default_factory=dict)
    extra: dict = Field(default_factory=dict)


class TenantContext(BaseModel):
    """Metadata about the tenant — used for persona distinctiveness and relevance."""

    tenant_id: str
    industry: str | None = None
    product_description: str | None = None
    existing_persona_names: list[str] = Field(
        default_factory=list,
        description="Names of personas already created for this tenant (avoid overlap)",
    )


class ClusterData(BaseModel):
    """Everything the synthesizer needs to produce a persona from one cluster."""

    cluster_id: str
    tenant: TenantContext
    summary: ClusterSummary
    sample_records: list[SampleRecord] = Field(min_length=1)
    enrichment: EnrichmentPayload = Field(default_factory=EnrichmentPayload)

    @property
    def all_record_ids(self) -> list[str]:
        return [r.record_id for r in self.sample_records]
