from __future__ import annotations

import unittest
from pathlib import Path

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.connectors.threaded_web import ThreadAwareWebConnector
from crawler.feature_crawler.models import CrawlTarget


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "threaded_discussion_sample.html"
)


class ThreadAwareWebConnectorTest(unittest.TestCase):
    def test_thread_aware_parser_emits_reply_interactions(self) -> None:
        connector = ThreadAwareWebConnector()
        target = CrawlTarget(
            platform="web",
            target_id="fortnite-thread",
            url=FIXTURE_PATH.as_uri(),
            community_name="fortnite-thread",
            collection_basis="owned",
            allow_persona_inference=True,
            metadata={"render_mode": "http", "until": "2026-04-09T23:59:59Z"},
        )
        context = CrawlContext(crawl_run_id="run_test_002", observed_at="2026-04-09T12:00:00Z")

        records = list(connector.fetch(target=target, context=context, since="2026-04-01T00:00:00Z"))
        record_types = [record.record_type for record in records]
        self.assertEqual(record_types.count("message"), 2)
        self.assertEqual(record_types.count("interaction"), 1)
        thread_record = next(record for record in records if record.record_type == "thread")
        self.assertEqual(thread_record.metadata["thread_aware"], True)


if __name__ == "__main__":
    unittest.main()
