from __future__ import annotations

from pydantic import BaseModel, Field


class RawRecord(BaseModel):
    """A normalized record from the crawler stage.

    This is the input contract: crawler writes records in this shape to
    Iceberg tables. Segmentation reads them and groups them into clusters.

    Fields are intentionally close to PRD_CRAWLER's contract:
        {source, tenant, timestamp, payload, schema_version}

    plus the extracted behavioral signals that drive clustering.
    """

    record_id: str = Field(description="Globally unique record ID from the crawler")
    tenant_id: str
    source: str = Field(description="e.g. 'ga4', 'hubspot', 'intercom', 'linkedin'")
    timestamp: str | None = None
    user_id: str | None = Field(
        default=None,
        description="Identifier for the underlying person — multiple records may share one",
    )
    behaviors: list[str] = Field(
        default_factory=list,
        description="Normalized behavior tags extracted from the record (e.g. 'api_setup')",
    )
    pages: list[str] = Field(
        default_factory=list,
        description="Page paths visited in this record",
    )
    payload: dict = Field(
        default_factory=dict,
        description="Source-specific fields preserved verbatim",
    )
    schema_version: str = "1.0"
