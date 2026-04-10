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
    date_range: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2} to \d{4}-\d{2}$",
        description=(
            "Earliest to latest month of the cited source records, "
            "formatted as 'YYYY-MM to YYYY-MM' (e.g. '2026-03 to 2026-04')"
        ),
    )
