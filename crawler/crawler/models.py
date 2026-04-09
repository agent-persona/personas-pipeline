from __future__ import annotations

from pydantic import BaseModel, Field


class Record(BaseModel):
    """A normalized record emitted by a crawler connector.

    This is the contract crawler honors and segmentation reads. It mirrors
    `segmentation.models.RawRecord` so the two can drop-in interop without
    a cross-package import.
    """

    record_id: str
    tenant_id: str
    source: str = Field(description="e.g. 'ga4', 'intercom', 'hubspot'")
    timestamp: str | None = None
    user_id: str | None = None
    behaviors: list[str] = Field(default_factory=list)
    pages: list[str] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)
    schema_version: str = "1.0"
