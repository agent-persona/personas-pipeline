from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.models import AccountRecord, CrawlTarget, EvidencePointer, ProfileSnapshotRecord, record_to_dict
from crawler.feature_crawler.platforms.linkedin.connector_browser import (
    BrowserConnectionsPayload,
    LinkedInBrowserConnector,
)
from crawler.feature_crawler.platforms.linkedin.connector_official import (
    LinkedInOfficialConnector,
    LinkedInOidcClient,
)
from crawler.feature_crawler.platforms.linkedin.connector_profile import LinkedInProfileConnector
from crawler.feature_crawler.sink import JsonlSink


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "linkedin_profile_sample.html"
)


class LinkedInProfileConnectorTest(unittest.TestCase):
    def test_public_html_connector_emits_canonical_profile_records(self) -> None:
        connector = LinkedInProfileConnector(mode="public-html")
        target = CrawlTarget(
            platform="linkedin",
            target_id="jane-builder",
            url=FIXTURE_PATH.as_uri(),
            community_name="Jane Builder",
            collection_basis="consented",
            allow_persona_inference=True,
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_001",
            observed_at="2026-04-11T14:00:00Z",
        )

        records = list(connector.fetch(target=target, context=context))
        record_types = [record.record_type for record in records]
        self.assertEqual(record_types.count("community"), 1)
        self.assertEqual(record_types.count("account"), 1)
        self.assertEqual(record_types.count("profile_snapshot"), 1)
        self.assertEqual(record_types.count("thread"), 1)
        self.assertGreaterEqual(record_types.count("message"), 4)

        snapshot = next(
            record_to_dict(record) for record in records if record.record_type == "profile_snapshot"
        )
        self.assertEqual(snapshot["fields"]["public_identifier"], "jane-builder")
        self.assertEqual(snapshot["fields"]["location"], "New York, NY")
        self.assertEqual(len(snapshot["fields"]["experience"]), 2)
        self.assertEqual(len(snapshot["fields"]["activity"]), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = JsonlSink(Path(tmpdir)).write(records)
            self.assertTrue(output_path.exists())
            self.assertIn("/linkedin/jane-builder/", output_path.as_posix())

    def test_session_mode_requires_cookie(self) -> None:
        with self.assertRaises(ValueError):
            LinkedInProfileConnector(mode="session-html")

    def test_official_oidc_connector_emits_profile_records(self) -> None:
        connector = LinkedInOfficialConnector(
            LinkedInOidcClient(
                access_token="token",
                fetch_json=lambda _url: {
                    "sub": "abcd1234",
                    "name": "Jane Builder",
                    "given_name": "Jane",
                    "family_name": "Builder",
                    "picture": "https://media.licdn.com/example.jpg",
                    "email": "jane@example.com",
                    "locale": "en-US",
                },
            )
        )
        target = CrawlTarget(
            platform="linkedin",
            target_id="jane-builder-official",
            url="https://api.linkedin.com/v2/userinfo",
            community_name="Jane Builder",
            collection_basis="consented",
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_002",
            observed_at="2026-04-11T15:00:00Z",
        )

        records = list(connector.fetch(target=target, context=context))
        record_types = [record.record_type for record in records]
        self.assertEqual(record_types.count("community"), 1)
        self.assertEqual(record_types.count("account"), 1)
        self.assertEqual(record_types.count("profile_snapshot"), 1)
        self.assertEqual(record_types.count("thread"), 1)
        self.assertGreaterEqual(record_types.count("message"), 2)

        snapshot = next(
            record_to_dict(record) for record in records if record.record_type == "profile_snapshot"
        )
        self.assertEqual(snapshot["fields"]["source_mode"], "official-oidc")
        self.assertEqual(snapshot["fields"]["email"], "jane@example.com")

    def test_session_browser_connector_emits_network_records(self) -> None:
        target = CrawlTarget(
            platform="linkedin",
            target_id="jane-builder",
            url=FIXTURE_PATH.as_uri(),
            community_name="Jane Builder",
            collection_basis="consented",
            metadata={"include_network": True},
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_005",
            observed_at="2026-04-11T17:00:00Z",
        )
        profile_connector = type(
            "ProfileConnectorStub",
            (),
            {
                "fetch": lambda _self, target, context, since=None: [
                    AccountRecord(
                        record_type="account",
                        platform="linkedin",
                        platform_user_id="linkedin-user-jane-builder",
                        username="Jane Builder",
                        account_created_at=None,
                        first_observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(source_url=target.url, fetched_at=context.observed_at),
                    ),
                    ProfileSnapshotRecord(
                        record_type="profile_snapshot",
                        platform="linkedin",
                        platform_user_id="linkedin-user-jane-builder",
                        snapshot_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        fields={"public_identifier": "jane-builder", "display_name": "Jane Builder"},
                        evidence_pointer=EvidencePointer(source_url=target.url, fetched_at=context.observed_at),
                    ),
                ]
            },
        )()
        browser_client = type(
            "BrowserClientStub",
            (),
            {
                "fetch_connections": lambda _self: BrowserConnectionsPayload(
                    source_url="https://www.linkedin.com/mynetwork/invite-connect/connections/",
                    connection_count=2,
                    connections=[
                        {
                            "name": "Pat Example",
                            "headline": "Founder at Example Co",
                            "profile_url": "https://www.linkedin.com/in/pat-example/",
                        },
                        {
                            "name": "Chris Builder",
                            "headline": "Design Engineer",
                            "profile_url": "https://www.linkedin.com/in/chris-builder/",
                        },
                    ],
                )
            },
        )()
        connector = LinkedInBrowserConnector(
            profile_connector=profile_connector,
            browser_client=browser_client,
        )

        records = list(connector.fetch(target=target, context=context))
        record_types = [record.record_type for record in records]
        self.assertEqual(record_types.count("thread"), 1)
        self.assertEqual(record_types.count("interaction"), 2)
        self.assertEqual(record_types.count("message"), 2)

        network_thread = next(
            record_to_dict(record)
            for record in records
            if record.record_type == "thread"
        )
        self.assertEqual(network_thread["metadata"]["source_mode"], "session-browser")
        self.assertEqual(network_thread["metadata"]["connection_count"], 2)
