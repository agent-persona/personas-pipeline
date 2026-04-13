from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TemporalSlice:
    """A single point-in-time snapshot of a persona."""

    year: int
    label: Literal["past", "present", "future"]
    summary: str
    sample_quotes: list[str]
    role_title: str
    key_traits: list[str]


@dataclass
class PersonaTemporal:
    """Wraps three TemporalSlice instances for a persona: past, present, future.

    Each slice captures the same underlying person at a different career stage,
    enabling temporal depth testing in the twin runtime.
    """

    base_name: str
    past: TemporalSlice
    present: TemporalSlice
    future: TemporalSlice

    def get_slice(self, label: Literal["past", "present", "future"]) -> TemporalSlice:
        return getattr(self, label)

    def slices(self) -> list[TemporalSlice]:
        return [self.past, self.present, self.future]
