from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    SET = "set"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class UserFeatures(BaseModel):
    """Aggregated feature set for a single user across their records."""

    user_id: str
    tenant_id: str
    behaviors: set[str] = Field(default_factory=set)
    pages: set[str] = Field(default_factory=set)
    sources: set[str] = Field(default_factory=set)
    record_ids: list[str] = Field(default_factory=list)
    numeric_features: dict[str, float] = Field(default_factory=dict)
    categorical_features: dict[str, str] = Field(default_factory=dict)
    set_features: dict[str, set[str]] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
