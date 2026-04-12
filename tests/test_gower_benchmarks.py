"""Task 8: Benchmark tests — runtime verification."""

from __future__ import annotations

import time
import random

from segmentation.models.record import RawRecord
from segmentation.pipeline import segment


def _synthetic_records(n_users: int, tenant: str = "t_bench") -> list[RawRecord]:
    """Generate n_users synthetic records with mixed features."""
    rng = random.Random(42)
    behaviors_pool = [f"behavior_{i}" for i in range(20)]
    pages_pool = [f"/page/{i}" for i in range(15)]
    industries = ["fintech", "saas", "enterprise", "design", "health"]
    roles = ["engineer", "designer", "manager", "analyst", "researcher"]
    sources = ["ga4", "hubspot", "intercom"]

    records = []
    for i in range(n_users):
        source = rng.choice(sources)
        n_behaviors = rng.randint(1, 5)
        user_behaviors = rng.sample(behaviors_pool, min(n_behaviors, len(behaviors_pool)))
        n_pages = rng.randint(0, 3)
        user_pages = rng.sample(pages_pool, min(n_pages, len(pages_pool)))

        payload = {}
        if source == "ga4":
            payload = {"session_duration": rng.randint(100, 5000)}
        elif source == "hubspot":
            payload = {
                "industry": rng.choice(industries),
                "company_size": rng.choice(["1-10", "10-50", "50-200", "200-500"]),
                "contact_title": rng.choice(roles),
            }
        elif source == "intercom":
            payload = {"topic": rng.choice(["api", "billing", "feature", "bug", "onboarding"])}

        records.append(RawRecord(
            record_id=f"bench_{i}",
            tenant_id=tenant,
            source=source,
            timestamp=None,
            user_id=f"user_{i}",
            behaviors=user_behaviors,
            pages=user_pages,
            payload=payload,
        ))
    return records


def test_benchmark_small_under_1s():
    """~40 records: Gower segment completes in <1 second."""
    records = _synthetic_records(40)
    start = time.monotonic()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Took {elapsed:.2f}s"
    assert len(clusters) >= 1


def test_benchmark_400_under_5s():
    """400 records: Gower segment completes in <5 seconds."""
    records = _synthetic_records(400)
    start = time.monotonic()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"Took {elapsed:.2f}s"
    assert len(clusters) >= 1


def test_benchmark_1000_under_30s():
    """1000 records: Gower segment completes in <30 seconds."""
    records = _synthetic_records(1000)
    start = time.monotonic()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    elapsed = time.monotonic() - start
    assert elapsed < 30.0, f"Took {elapsed:.2f}s"
    assert len(clusters) >= 1


def test_benchmark_10000_recorded():
    """10000 records: record timing (not a hard gate, but <300s expected)."""
    records = _synthetic_records(10000)
    start = time.monotonic()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    elapsed = time.monotonic() - start
    print(f"\n  BENCHMARK: 10000 users → {elapsed:.1f}s, {len(clusters)} clusters")
    assert elapsed < 300.0, f"Took {elapsed:.1f}s — exceeds 5-minute budget"
    assert len(clusters) >= 1
