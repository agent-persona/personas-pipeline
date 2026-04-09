from __future__ import annotations

from pydantic import BaseModel, Field


class UserFeatures(BaseModel):
    """Aggregated feature set for a single user across their records."""

    user_id: str
    tenant_id: str
    behaviors: set[str] = Field(default_factory=set)
    pages: set[str] = Field(default_factory=set)
    sources: set[str] = Field(default_factory=set)
    record_ids: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
