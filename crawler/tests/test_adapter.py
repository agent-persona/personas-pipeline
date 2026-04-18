from __future__ import annotations

from pathlib import Path

from crawler import fetch_from_run
from crawler.adapter import bronze_to_flat, load_run_jsonl
from crawler.feature_crawler.__main__ import _demo_discord
from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.connectors import ApprovedWebConnector
from crawler.feature_crawler.models import (
    AccountRecord,
    CrawlTarget,
    EvidencePointer,
    InteractionRecord,
    MessageRecord,
    ProfileSnapshotRecord,
    ThreadRecord,
)
from crawler.feature_crawler.pipeline import BronzeWriter, CrawlerRunner
from crawler.feature_crawler.policy import CollectionBasis, PolicyRegistry
from crawler.feature_crawler.sink import JsonlSink
from segmentation.models.record import RawRecord
from segmentation.pipeline import segment


FIXTURE_DIR = Path(__file__).resolve().parent / "feature_crawler" / "fixtures"
APPROVED_WEB_FIXTURE = FIXTURE_DIR / "approved_web_sample.html"


def _pointer(source_url: str) -> EvidencePointer:
    return EvidencePointer(
        source_url=source_url,
        fetched_at="2026-04-11T14:00:00Z",
    )


def test_bronze_to_flat_maps_behavioral_records() -> None:
    bronze_records = [
        ThreadRecord(
            record_type="thread",
            platform="linkedin",
            thread_id="thread_1",
            community_id="jane-builder",
            title="How should founders use AI for recruiting?",
            author_platform_user_id="user_founder",
            created_at="2026-04-11T14:00:00Z",
            observed_at="2026-04-11T14:00:00Z",
            crawl_run_id="run_test_li_001",
            metadata={"source_mode": "public-html"},
            evidence_pointer=_pointer("https://www.linkedin.com/in/jane-builder/"),
        ),
        MessageRecord(
            record_type="message",
            platform="linkedin",
            message_id="msg_1",
            thread_id="thread_1",
            community_id="jane-builder",
            author_platform_user_id="user_founder",
            body="Hiring engineers? We use AI workflows and share examples at https://example.com",
            created_at="2026-04-11T14:01:00Z",
            observed_at="2026-04-11T14:01:00Z",
            crawl_run_id="run_test_li_001",
            reply_to_message_id=None,
            reply_to_user_id=None,
            metadata={"source_mode": "public-html"},
            evidence_pointer=_pointer("https://www.linkedin.com/feed/update/urn:li:activity:1/"),
        ),
        InteractionRecord(
            record_type="interaction",
            platform="linkedin",
            interaction_type="reply",
            source_user_id="user_founder",
            target_user_id="user_engineer",
            message_id="msg_1",
            thread_id="thread_1",
            community_id="jane-builder",
            created_at="2026-04-11T14:02:00Z",
            crawl_run_id="run_test_li_001",
            evidence_pointer=_pointer("https://www.linkedin.com/feed/update/urn:li:activity:1/"),
        ),
        ProfileSnapshotRecord(
            record_type="profile_snapshot",
            platform="linkedin",
            platform_user_id="user_founder",
            snapshot_at="2026-04-11T14:03:00Z",
            crawl_run_id="run_test_li_001",
            fields={
                "headline": "Founder building AI recruiting tools",
                "experience": [{"title": "Founder"}],
                "activity": [{"title": "Hiring tips"}],
            },
            evidence_pointer=_pointer("https://www.linkedin.com/in/jane-builder/"),
        ),
        AccountRecord(
            record_type="account",
            platform="linkedin",
            platform_user_id="user_founder",
            username="Jane Builder",
            account_created_at=None,
            first_observed_at="2026-04-11T14:04:00Z",
            crawl_run_id="run_test_li_001",
            evidence_pointer=_pointer("https://www.linkedin.com/in/jane-builder/"),
        ),
    ]

    records = bronze_to_flat(bronze_records, tenant_id="tenant_demo")

    assert len(records) == 5
    by_id = {record.record_id: record for record in records}
    message = next(record for record in by_id.values() if "posted_message" in record.behaviors)
    assert message.user_id == "user_founder"
    assert "shared_link" in message.behaviors
    assert "topic_ai" in message.behaviors
    assert message.pages

    thread = next(record for record in by_id.values() if "started_thread" in record.behaviors)
    assert "asked_question" in thread.behaviors

    profile = next(record for record in by_id.values() if "profile_snapshot" in record.behaviors)
    assert "role_founder" in profile.behaviors
    assert "active_poster" in profile.behaviors

    interaction = next(record for record in by_id.values() if "interaction_reply" in record.behaviors)
    assert "replied_to_user" in interaction.behaviors


def test_load_run_jsonl_reads_single_jsonl_output(tmp_path: Path) -> None:
    target = CrawlTarget(
        platform="web",
        target_id="sample-source",
        url=APPROVED_WEB_FIXTURE.as_uri(),
        community_name="sample-source",
        collection_basis="owned",
        allow_persona_inference=True,
    )
    connector = ApprovedWebConnector()
    context = CrawlContext(
        crawl_run_id="run_test_approved_web",
        observed_at="2026-04-11T14:10:00Z",
    )

    bronze_records = list(connector.fetch(target=target, context=context))
    output_path = JsonlSink(tmp_path).write(bronze_records)

    records = load_run_jsonl(output_path, tenant_id="tenant_web")

    assert records
    assert all(RawRecord.model_validate(record.model_dump()) for record in records)
    assert any("posted_message" in record.behaviors for record in records)


def test_load_run_jsonl_reads_bronze_writer_directory(tmp_path: Path) -> None:
    connector, target = _demo_discord()
    run = CrawlerRunner(PolicyRegistry(), BronzeWriter(tmp_path)).run(connector, target)

    records = fetch_from_run(run.root, tenant_id="tenant_discord")

    assert records
    assert any(record.source == "discord" for record in records)
    assert any("started_thread" in record.behaviors for record in records)
    assert any("posted_message" in record.behaviors for record in records)


def test_adapted_records_work_with_segmentation(tmp_path: Path) -> None:
    connector, target = _demo_discord()
    run = CrawlerRunner(PolicyRegistry(), BronzeWriter(tmp_path)).run(connector, target)

    flat_records = fetch_from_run(run.root, tenant_id="tenant_segment")
    raw_records = [RawRecord.model_validate(record.model_dump()) for record in flat_records]

    clusters = segment(
        raw_records,
        similarity_threshold=0.0,
        min_cluster_size=2,
    )

    assert clusters
    assert any(summary["summary"]["cluster_size"] >= 2 for summary in clusters)
    assert any(
        "posted_message" in summary["summary"]["top_behaviors"]
        or "started_thread" in summary["summary"]["top_behaviors"]
        for summary in clusters
    )
