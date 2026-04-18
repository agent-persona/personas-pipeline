from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.models import AccountRecord, CrawlTarget, EvidencePointer, ProfileSnapshotRecord, record_to_dict
from crawler.feature_crawler.platforms.linkedin.connector_browser import (
    BrowserConnectionsPayload,
    BrowserProfilePayload,
    LinkedInBrowserConnector,
    _parse_browser_profile,
    _split_section_blob,
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

    def test_session_browser_connector_falls_back_when_profile_prefetch_fails(self) -> None:
        target = CrawlTarget(
            platform="linkedin",
            target_id="jane-builder",
            url="https://www.linkedin.com/in/jane-builder/",
            community_name="Jane Builder",
            collection_basis="consented",
            metadata={"include_network": True},
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_006",
            observed_at="2026-04-11T18:00:00Z",
        )
        profile_connector = type(
            "ProfileConnectorFailStub",
            (),
            {
                "fetch": lambda _self, target, context, since=None: (_ for _ in ()).throw(RuntimeError("302 loop")),
            },
        )()
        browser_client = type(
            "BrowserClientStub",
            (),
            {
                "fetch_connections": lambda _self: BrowserConnectionsPayload(
                    source_url="https://www.linkedin.com/mynetwork/invite-connect/connections/",
                    connection_count=1,
                    connections=[
                        {
                            "name": "Pat Example",
                            "headline": "Founder at Example Co",
                            "profile_url": "https://www.linkedin.com/in/pat-example/",
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
        self.assertEqual(record_types.count("account"), 1)
        self.assertEqual(record_types.count("profile_snapshot"), 1)
        self.assertEqual(record_types.count("message"), 1)
        self.assertEqual(record_types.count("interaction"), 1)

    def test_session_browser_connector_uses_browser_for_profile(self) -> None:
        target = CrawlTarget(
            platform="linkedin",
            target_id="shruti-jn",
            url="https://www.linkedin.com/in/shruti-jn/",
            community_name="Shruti JN",
            collection_basis="consented",
            allow_persona_inference=True,
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_007",
            observed_at="2026-04-12T10:00:00Z",
        )
        browser_client = type(
            "BrowserClientProfileStub",
            (),
            {
                "fetch_profile": lambda _self, profile_url: BrowserProfilePayload(
                    profile_url="https://www.linkedin.com/in/shruti-jn/",
                    name="Shruti JN",
                    headline="Intentional Technologist",
                    about="Building thoughtful tech...",
                    location="San Francisco, CA",
                    experience=[
                        "Software Engineer at Google 2020-2023",
                        "Staff Engineer at Meta 2023-present",
                    ],
                    activity=[
                        "Great thread on AI safety!",
                        "Excited to share my latest project",
                    ],
                    education=["BS Computer Science, Stanford"],
                    skills=["Python", "Machine Learning"],
                    section_map={},
                ),
            },
        )()
        profile_connector = type(
            "ProfileConnectorMustNotCall",
            (),
            {
                "fetch": lambda _self, target, context, since=None: (_ for _ in ()).throw(
                    AssertionError("profile_connector.fetch() should not be called when browser succeeds")
                ),
            },
        )()
        connector = LinkedInBrowserConnector(
            profile_connector=profile_connector,
            browser_client=browser_client,
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
        self.assertIsInstance(snapshot["fields"]["experience"], list)
        self.assertEqual(len(snapshot["fields"]["experience"]), 2)
        self.assertIsInstance(snapshot["fields"]["activity"], list)
        self.assertEqual(len(snapshot["fields"]["activity"]), 2)
        self.assertEqual(snapshot["fields"]["source_mode"], "session-browser")

        messages = [
            record_to_dict(record) for record in records if record.record_type == "message"
        ]
        message_bodies = [msg["body"] for msg in messages]
        self.assertIn("Intentional Technologist", message_bodies)
        self.assertIn("Building thoughtful tech...", message_bodies)
        self.assertIn("Software Engineer at Google 2020-2023", message_bodies)
        self.assertIn("Staff Engineer at Meta 2023-present", message_bodies)
        self.assertIn("Great thread on AI safety!", message_bodies)
        self.assertIn("Excited to share my latest project", message_bodies)

    def test_session_browser_connector_falls_back_to_http_on_browser_failure(self) -> None:
        target = CrawlTarget(
            platform="linkedin",
            target_id="jane-builder",
            url="https://www.linkedin.com/in/jane-builder/",
            community_name="Jane Builder",
            collection_basis="consented",
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_008",
            observed_at="2026-04-12T11:00:00Z",
        )
        browser_client = type(
            "BrowserClientFailStub",
            (),
            {
                "fetch_profile": lambda _self, profile_url: (_ for _ in ()).throw(
                    RuntimeError("browser extraction timed out")
                ),
            },
        )()
        profile_connector = type(
            "ProfileConnectorFallbackStub",
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
        connector = LinkedInBrowserConnector(
            profile_connector=profile_connector,
            browser_client=browser_client,
        )

        records = list(connector.fetch(target=target, context=context))
        record_types = [record.record_type for record in records]
        self.assertEqual(record_types.count("account"), 1)
        self.assertEqual(record_types.count("profile_snapshot"), 1)

        snapshot = next(
            record_to_dict(record) for record in records if record.record_type == "profile_snapshot"
        )
        self.assertEqual(snapshot["fields"]["public_identifier"], "jane-builder")

    def test_browser_profile_with_network(self) -> None:
        target = CrawlTarget(
            platform="linkedin",
            target_id="shruti-jn",
            url="https://www.linkedin.com/in/shruti-jn/",
            community_name="Shruti JN",
            collection_basis="consented",
            metadata={"include_network": True},
        )
        context = CrawlContext(
            crawl_run_id="run_test_li_009",
            observed_at="2026-04-12T12:00:00Z",
        )
        browser_client = type(
            "BrowserClientFullStub",
            (),
            {
                "fetch_profile": lambda _self, profile_url: BrowserProfilePayload(
                    profile_url="https://www.linkedin.com/in/shruti-jn/",
                    name="Shruti JN",
                    headline="Intentional Technologist",
                    about="Building thoughtful tech...",
                    location="San Francisco, CA",
                    experience=["Software Engineer at Google 2020-2023"],
                    activity=["Great thread on AI safety!"],
                    education=["BS Computer Science, Stanford"],
                    skills=["Python"],
                    section_map={},
                ),
                "fetch_connections": lambda _self: BrowserConnectionsPayload(
                    source_url="https://www.linkedin.com/mynetwork/invite-connect/connections/",
                    connection_count=1,
                    connections=[
                        {
                            "name": "Pat Example",
                            "headline": "Founder at Example Co",
                            "profile_url": "https://www.linkedin.com/in/pat-example/",
                        },
                    ],
                ),
            },
        )()
        profile_connector = type(
            "ProfileConnectorUnused",
            (),
            {
                "fetch": lambda _self, target, context, since=None: (_ for _ in ()).throw(
                    AssertionError("should not be called")
                ),
            },
        )()
        connector = LinkedInBrowserConnector(
            profile_connector=profile_connector,
            browser_client=browser_client,
        )

        records = list(connector.fetch(target=target, context=context))
        record_types = [record.record_type for record in records]

        # Profile records
        self.assertEqual(record_types.count("community"), 1)
        self.assertGreaterEqual(record_types.count("account"), 1)
        self.assertGreaterEqual(record_types.count("profile_snapshot"), 1)
        self.assertGreaterEqual(record_types.count("thread"), 1)
        self.assertGreaterEqual(record_types.count("message"), 1)

        # Network records
        self.assertGreaterEqual(record_types.count("interaction"), 1)
        # Should have profile thread + network thread
        self.assertEqual(record_types.count("thread"), 2)

    def test_split_section_blob(self) -> None:
        multi_line_blob = (
            "Software Engineer at Google\n"
            "2020-2023 Mountain View, CA\n"
            "Staff Engineer at Meta\n"
            "2023-present Menlo Park, CA"
        )
        result = _split_section_blob(multi_line_blob)
        self.assertGreaterEqual(len(result), 2)

        single_line_blob = "Software Engineer at Google 2020-2023"
        result_single = _split_section_blob(single_line_blob)
        self.assertEqual(len(result_single), 1)
        self.assertEqual(result_single[0], single_line_blob)

        empty_blob = ""
        result_empty = _split_section_blob(empty_blob)
        self.assertEqual(result_empty, [])

    def test_parse_browser_profile_from_raw(self) -> None:
        raw = {
            "fetched_at": "2026-04-12T10:00:00Z",
            "profile_url": "https://www.linkedin.com/in/shruti-jn/",
            "title": "Shruti JN | LinkedIn",
            "name": "  Shruti JN  ",
            "headline": "Intentional Technologist",
            "location": "San Francisco, CA",
            "section_map": {
                "About": "Building thoughtful tech for the world.",
                "Experience": "Software Engineer at Google\n2020-2023\nStaff Engineer at Meta\n2023-present",
                "Education": "BS Computer Science, Stanford University",
                "Skills": "Python Machine Learning System Design",
            },
            "section_items": {
                "Experience": [
                    "Software Engineer at Google 2020-2023",
                    "Staff Engineer at Meta 2023-present",
                ],
                "Skills": ["Python", "Machine Learning", "System Design"],
            },
            "activity_items": [
                "Great thread on AI safety! Everyone should read this.",
                "Excited to share my latest project on responsible AI tooling.",
            ],
        }

        payload = _parse_browser_profile(raw)

        self.assertIsInstance(payload, BrowserProfilePayload)
        self.assertEqual(payload.name, "Shruti JN")
        self.assertEqual(payload.headline, "Intentional Technologist")
        self.assertEqual(payload.about, "Building thoughtful tech for the world.")
        self.assertEqual(payload.location, "San Francisco, CA")
        self.assertEqual(payload.profile_url, "https://www.linkedin.com/in/shruti-jn/")
        self.assertEqual(len(payload.experience), 2)
        self.assertIn("Software Engineer at Google 2020-2023", payload.experience)
        self.assertEqual(len(payload.activity), 2)
        self.assertIn("Great thread on AI safety! Everyone should read this.", payload.activity)
        self.assertEqual(len(payload.skills), 3)
        self.assertIn("Python", payload.skills)
