from typing import Literal

from pydantic import BaseModel, Field


class SourceEvidence(BaseModel):
    """Links a persona claim back to source records for groundedness tracking."""

    claim: str = Field(description="The specific persona claim this evidence supports")
    record_ids: list[str] = Field(
        min_length=1,
        description="IDs of source records that justify this claim",
    )
    field_path: str = Field(
        description="Dot-notation path to the persona field, e.g. 'goals.0'",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the data supports this claim",
    )
    source_url: str | None = None
    platform: str | None = None
    excerpt: str | None = None
    observed_at: str | None = None
    match_reason: str | None = None
    match_signals: list[str] | None = None
    status: Literal["used", "candidate", "blocked"] | None = None
