"""Load test fixtures into RawRecord or dict form."""

from __future__ import annotations

import json
from pathlib import Path

from segmentation.models.record import RawRecord

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_records(fixture_name: str) -> list[RawRecord]:
    """Load a JSON fixture as a list of RawRecord."""
    with open(FIXTURES_DIR / fixture_name) as f:
        data = json.load(f)
    return [RawRecord.model_validate(d) for d in data]


def load_snapshot(fixture_name: str) -> dict:
    """Load a JSON snapshot as a dict."""
    with open(FIXTURES_DIR / fixture_name) as f:
        return json.load(f)


def normalize_cluster_order(clusters: list[dict]) -> list[dict]:
    """Sort clusters by sorted member record_ids for stable comparison."""
    def sort_key(c):
        rids = sorted(sr["record_id"] for sr in c["sample_records"])
        return rids
    return sorted(clusters, key=sort_key)
