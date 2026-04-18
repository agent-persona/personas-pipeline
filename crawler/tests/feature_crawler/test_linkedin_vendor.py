from __future__ import annotations

import os
import unittest

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.models import CrawlTarget
from crawler.feature_crawler.platforms.linkedin.connector_vendor import (
    ApifyVendorClient,
    LinkedInVendorConnector,
    LinkdApiVendorClient,
)


class LinkedInVendorConnectorTest(unittest.TestCase):
    def test_linkdapi_vendor_emits_posts_comments_and_network(self) -> None:
        def fake_fetch(method: str, url: str, headers: dict[str, str], _json_body):
            if "/profile/full?" in url:
                return {
                    "fullName": "Jane Builder",
                    "publicIdentifier": "jane-builder",
                    "headline": "Founder at Persona Works",
                    "summary": "Builds evidence-backed persona systems.",
                    "location": "New York, NY",
                    "experience": [{"title": "Founder", "companyName": "Persona Works"}],
                }
            if "/profile/posts?" in url:
                return {
                    "posts": [
                        {
                            "id": "post-1",
                            "text": "Structured evidence beats vibe-based personas.",
                            "createdAt": "2026-04-11T12:00:00Z",
                            "url": "https://www.linkedin.com/feed/update/post-1/",
                            "comments": [
                                {
                                    "id": "comment-1",
                                    "text": "Agree. Replayability matters.",
                                    "authorName": "Alex",
                                    "authorPublicIdentifier": "alex",
                                    "createdAt": "2026-04-11T12:05:00Z",
                                }
                            ],
                        }
                    ]
                }
            if "/profile/connections?" in url:
                return {
                    "connections": [
                        {
                            "publicIdentifier": "casey",
                            "name": "Casey",
                            "headline": "Product lead",
                        }
                    ]
                }
            raise AssertionError(f"unexpected request: {method} {url}")

        connector = LinkedInVendorConnector(
            "linkdapi",
            LinkdApiVendorClient("token-1", fetch_json=fake_fetch),
        )
        target = CrawlTarget(
            platform="linkedin",
            target_id="jane-builder",
            url="https://www.linkedin.com/in/jane-builder/",
            community_name="Jane Builder",
            collection_basis="consented",
            allow_persona_inference=True,
            metadata={
                "include_posts": True,
                "include_network": True,
                "post_limit": 10,
                "comment_limit": 10,
                "network_limit": 10,
                "page_limit": 2,
            },
        )
        context = CrawlContext(crawl_run_id="run_test_vendor_001", observed_at="2026-04-11T18:00:00Z")
        records = list(connector.fetch(target=target, context=context))
        types = [record.record_type for record in records]
        self.assertIn("community", types)
        self.assertIn("profile_snapshot", types)
        self.assertGreaterEqual(types.count("thread"), 3)
        self.assertGreaterEqual(types.count("message"), 4)
        self.assertGreaterEqual(types.count("interaction"), 2)

    def test_apify_client_runs_actor_and_reads_dataset(self) -> None:
        seen: list[tuple[str, str]] = []

        def fake_fetch(method: str, url: str, _headers: dict[str, str], json_body):
            seen.append((method, url))
            if "/acts/" in url:
                self.assertEqual(method, "POST")
                self.assertEqual(json_body["maxProfiles"], 1)
                return {"data": {"defaultDatasetId": "dataset-1"}}
            if "/datasets/dataset-1/items" in url:
                self.assertEqual(method, "GET")
                return [{"fullName": "Jane Builder"}]
            raise AssertionError(f"unexpected request: {method} {url}")

        client = ApifyVendorClient(
            "token-1",
            profile_actor="actor-1",
            fetch_json=fake_fetch,
        )
        profile = client.fetch_profile("https://www.linkedin.com/in/jane-builder/", "jane-builder")
        self.assertEqual(profile["fullName"], "Jane Builder")
        self.assertEqual(len(seen), 2)

    def test_from_env_requires_vendor_secrets(self) -> None:
        original = dict(os.environ)
        try:
            os.environ.pop("LINKDAPI_API_KEY", None)
            with self.assertRaises(ValueError):
                LinkedInVendorConnector.from_env(vendor="linkdapi")
        finally:
            os.environ.clear()
            os.environ.update(original)
