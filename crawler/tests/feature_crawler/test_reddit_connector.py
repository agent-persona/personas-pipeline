from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.platforms.reddit.connector_api import (
    RedditApiClient,
    RedditApiConnector,
    RedditPublicJsonClient,
)
from crawler.feature_crawler.models import CrawlTarget
from crawler.feature_crawler.sink import JsonlSink


class RedditApiConnectorTest(unittest.TestCase):
    def test_connector_emits_threads_comments_and_interactions(self) -> None:
        def fake_fetch(path: str):
            if path.startswith("/r/python/about"):
                return {
                    "data": {
                        "display_name_prefixed": "r/python",
                        "public_description": "Python discussions",
                        "subscribers": 123456,
                    }
                }
            if path.startswith("/r/python/new?"):
                return {
                    "data": {
                        "children": [
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "abc123",
                                    "permalink": "/r/python/comments/abc123/example_post/",
                                    "title": "Example post",
                                    "author": "op_user",
                                    "created_utc": 1775736000,
                                    "selftext": "Official API path looks cleaner than scraping.",
                                    "score": 42,
                                    "num_comments": 2,
                                },
                            }
                        ]
                    }
                }
            if path.startswith("/r/python/comments/abc123?"):
                return [
                    {},
                    {
                        "data": {
                            "children": [
                                {
                                    "kind": "t1",
                                    "data": {
                                        "id": "c1",
                                        "name": "t1_c1",
                                        "author": "reply_user",
                                        "body": "Agreed, use OAuth and keep it civil.",
                                        "created_utc": 1775736300,
                                        "parent_id": "t3_abc123",
                                        "depth": 0,
                                        "score": 7,
                                        "replies": {
                                            "data": {
                                                "children": [
                                                    {
                                                        "kind": "t1",
                                                        "data": {
                                                            "id": "c2",
                                                            "name": "t1_c2",
                                                            "author": "nested_user",
                                                            "body": "PRAW is nice, but raw HTTP works here too.",
                                                            "created_utc": 1775736600,
                                                            "parent_id": "t1_c1",
                                                            "depth": 1,
                                                            "score": 3,
                                                            "replies": "",
                                                        },
                                                    }
                                                ]
                                            }
                                        },
                                    },
                                }
                            ]
                        }
                    },
                ]
            raise AssertionError(f"unexpected path: {path}")

        connector = RedditApiConnector(
            RedditApiClient(
                client_id="id",
                client_secret="secret",
                user_agent="agent-personas-crawler/0.1",
                fetch_json=fake_fetch,
            )
        )
        target = CrawlTarget(
            platform="reddit",
            target_id="python",
            url="https://www.reddit.com/r/python/",
            community_name="python",
            collection_basis="public-permitted",
            metadata={"subreddit": "python", "sort": "new", "limit": 10, "comment_limit": 32},
        )
        context = CrawlContext(crawl_run_id="run_test_004", observed_at="2026-04-09T12:00:00Z")

        records = list(connector.fetch(target=target, context=context, since="2026-04-08T00:00:00Z"))
        record_types = [record.record_type for record in records]
        self.assertEqual(record_types.count("thread"), 1)
        self.assertEqual(record_types.count("message"), 3)
        self.assertEqual(record_types.count("interaction"), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = JsonlSink(Path(tmpdir)).write(records)
            self.assertTrue(output_path.exists())
            self.assertIn("/reddit/python/", output_path.as_posix())

    def test_public_json_client_adds_json_suffixes(self) -> None:
        seen_paths: list[str] = []

        def fake_fetch(path: str):
            seen_paths.append(path)
            if path.startswith("/r/python/about.json?"):
                return {"data": {"display_name_prefixed": "r/python"}}
            if path.startswith("/r/python/new.json?"):
                return {"data": {"children": []}}
            return []

        connector = RedditApiConnector(
            RedditPublicJsonClient(
                user_agent="agent-personas-crawler/0.1",
                fetch_json=fake_fetch,
            )
        )
        target = CrawlTarget(
            platform="reddit",
            target_id="python",
            url="https://www.reddit.com/r/python/",
            community_name="python",
            collection_basis="public-permitted",
            metadata={"subreddit": "python", "sort": "new", "limit": 5, "comment_limit": 8},
        )

        records = list(
            connector.fetch(
                target=target,
                context=CrawlContext(crawl_run_id="run_test_005", observed_at="2026-04-09T12:00:00Z"),
            )
        )

        self.assertEqual(len(records), 1)
        self.assertTrue(any(path.startswith("/r/python/about.json?") for path in seen_paths))
        self.assertTrue(any(path.startswith("/r/python/new.json?") for path in seen_paths))
