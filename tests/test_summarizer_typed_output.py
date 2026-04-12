"""Summarizer typed output: direct tests of build_cluster_data with typed features."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "../synthesis")

from segmentation.models.features import UserFeatures
from segmentation.models.record import RawRecord


def _make_users_and_records():
    """Create a cluster of 3 users with typed features and matching records."""
    users = [
        UserFeatures(
            user_id="u1", tenant_id="t1",
            behaviors={"api_setup", "webhook_config"},
            pages={"/api/docs"},
            sources={"ga4", "hubspot"},
            record_ids=["r1", "r2"],
            numeric_features={"session_duration": 2000.0},
            categorical_features={"industry": "fintech", "role": "Engineer"},
        ),
        UserFeatures(
            user_id="u2", tenant_id="t1",
            behaviors={"api_setup", "terraform_setup"},
            pages={"/api/docs", "/terraform"},
            sources={"ga4", "hubspot"},
            record_ids=["r3", "r4"],
            numeric_features={"session_duration": 1500.0},
            categorical_features={"industry": "saas", "role": "DevOps"},
        ),
        UserFeatures(
            user_id="u3", tenant_id="t1",
            behaviors={"api_setup"},
            pages={"/api/docs"},
            sources={"ga4"},
            record_ids=["r5"],
            numeric_features={"session_duration": 1800.0},
            categorical_features={"industry": "fintech"},
        ),
    ]
    records = [
        RawRecord(record_id="r1", tenant_id="t1", source="ga4", user_id="u1",
                  behaviors=["api_setup"], pages=["/api/docs"], payload={"session_duration": 2000}),
        RawRecord(record_id="r2", tenant_id="t1", source="hubspot", user_id="u1",
                  behaviors=["technical_role"], pages=[], payload={"industry": "fintech"}),
        RawRecord(record_id="r3", tenant_id="t1", source="ga4", user_id="u2",
                  behaviors=["api_setup"], pages=["/api/docs"], payload={"session_duration": 1500}),
        RawRecord(record_id="r4", tenant_id="t1", source="hubspot", user_id="u2",
                  behaviors=["technical_role"], pages=[], payload={"industry": "saas"}),
        RawRecord(record_id="r5", tenant_id="t1", source="ga4", user_id="u3",
                  behaviors=["api_setup"], pages=["/api/docs"], payload={"session_duration": 1800}),
    ]
    return users, records


def test_typed_features_in_extra():
    """Summary includes typed_features in extra when users have typed features."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    assert "typed_features" in result["summary"]["extra"]


def test_numeric_averages_in_typed():
    """typed_features contains numeric_averages with session_duration."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    avgs = result["summary"]["extra"]["typed_features"]["numeric_averages"]
    assert "session_duration" in avgs
    expected = (2000.0 + 1500.0 + 1800.0) / 3
    assert abs(avgs["session_duration"] - expected) < 0.01


def test_categorical_modes_in_typed():
    """typed_features contains categorical_modes with correct mode."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    modes = result["summary"]["extra"]["typed_features"]["categorical_modes"]
    assert "industry" in modes
    assert modes["industry"] == "fintech"  # 2 fintech vs 1 saas


def test_set_unions_in_typed():
    """typed_features contains set_unions (empty if no set_features on users)."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    unions = result["summary"]["extra"]["typed_features"]["set_unions"]
    assert isinstance(unions, dict)


def test_avg_session_duration_populated():
    """avg_session_duration_seconds is computed from numeric features."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    duration = result["summary"]["avg_session_duration_seconds"]
    assert duration is not None
    expected = (2000.0 + 1500.0 + 1800.0) / 3
    assert abs(duration - expected) < 0.01


def test_typed_features_json_serializable():
    """typed_features dict is fully JSON-serializable."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    tf = result["summary"]["extra"]["typed_features"]
    serialized = json.dumps(tf)
    assert isinstance(serialized, str)
    assert "session_duration" in serialized


def test_sample_records_still_representative():
    """Sample records still follow the representative-sample selection rules."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    samples = result["sample_records"]
    assert len(samples) >= 1
    sample_ids = {sr["record_id"] for sr in samples}
    all_ids = {r.record_id for r in records}
    assert sample_ids.issubset(all_ids)


def test_legacy_fields_present():
    """Legacy summary fields (top_behaviors, top_pages, source_breakdown) remain."""
    from segmentation.engine.summarizer import build_cluster_data

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1")
    s = result["summary"]
    assert "top_behaviors" in s
    assert "top_pages" in s
    assert "source_breakdown" in s["extra"]
    assert "cluster_size" in s


def test_validates_against_clusterdata():
    """Output validates against the real ClusterData Pydantic model."""
    from segmentation.engine.summarizer import build_cluster_data
    from synthesis.models.cluster import ClusterData

    users, records = _make_users_and_records()
    result = build_cluster_data(users, records, "t1", tenant_industry="B2B SaaS")
    validated = ClusterData.model_validate(result)
    assert validated.summary.cluster_size == 3
