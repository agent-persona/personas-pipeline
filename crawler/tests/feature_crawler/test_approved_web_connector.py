from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.connectors import ApprovedWebConnector
from crawler.feature_crawler.models import CrawlTarget, record_to_dict
from crawler.feature_crawler.sink import JsonlSink


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "approved_web_sample.html"
)


class ApprovedWebConnectorTest(unittest.TestCase):
    def test_connector_emits_records_from_file_url(self) -> None:
        target = CrawlTarget(
            platform="web",
            target_id="sample-source",
            url=FIXTURE_PATH.as_uri(),
            community_name="sample-source",
            collection_basis="owned",
            allow_persona_inference=True,
        )

        connector = ApprovedWebConnector()
        context = CrawlContext(
            crawl_run_id="run_test_001",
            observed_at="2026-04-09T12:00:00Z",
        )

        records = list(connector.fetch(target=target, context=context))
        record_types = [record.record_type for record in records]

        self.assertIn("community", record_types)
        self.assertIn("thread", record_types)
        self.assertIn("account", record_types)
        self.assertGreaterEqual(record_types.count("message"), 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = JsonlSink(Path(temp_dir)).write(records)
            payloads = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(payloads), len(records))

        thread_record = next(
            record_to_dict(record) for record in records if record.record_type == "thread"
        )
        self.assertEqual(thread_record["metadata"]["collection_basis"], "owned")


if __name__ == "__main__":
    unittest.main()
