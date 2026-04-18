from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.__main__ import _demo_discord, _demo_web
from crawler.feature_crawler.pipeline import BronzeWriter, CrawlerRunner
from crawler.feature_crawler.policy import PolicyRegistry


class PipelineTest(unittest.TestCase):
    def test_discord_demo_writes_expected_files(self) -> None:
        connector, target = _demo_discord()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = CrawlerRunner(PolicyRegistry(), BronzeWriter(Path(tmpdir))).run(connector, target)
            self.assertEqual(result.record_counts["community"], 2)
            self.assertEqual(result.record_counts["message"], 2)
            self.assertEqual(result.record_counts["interaction"], 1)
            self.assertTrue(result.files_written["manifest"].exists())

    def test_web_demo_writes_accounts_and_threads(self) -> None:
        connector, target = _demo_web()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = CrawlerRunner(PolicyRegistry(), BronzeWriter(Path(tmpdir))).run(connector, target)
            self.assertEqual(result.record_counts["account"], 2)
            self.assertEqual(result.record_counts["thread"], 1)


if __name__ == "__main__":
    unittest.main()
