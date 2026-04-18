from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "feature_crawler" / "generate_mock_conversations.py"


class MockConversationGeneratorTest(unittest.TestCase):
    def test_generator_emits_topic_page_with_multiple_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT_PATH),
                    "--output-dir",
                    tmpdir,
                    "--platform",
                    "discord",
                    "--pages-per-channel",
                    "1",
                    "--channels-per-platform",
                    "1",
                ],
                cwd=PACKAGE_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            files = list(Path(tmpdir).rglob("*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertIn("/discord/mock/", files[0].as_posix())
            expected_date_fragment = datetime.now().strftime("/%-m-%-d-%Y/")
            self.assertIn(expected_date_fragment, files[0].as_posix())

            lines = files[0].read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 100)
            self.assertLessEqual(len(lines), 500)
            payloads = [json.loads(line) for line in lines]

            record_types = {payload["record_type"] for payload in payloads}
            self.assertIn("community", record_types)
            self.assertIn("thread", record_types)
            self.assertIn("message", record_types)
            self.assertIn("interaction", record_types)

            authors = {
                payload["author_platform_user_id"]
                for payload in payloads
                if payload["record_type"] == "message"
            }
            self.assertGreater(len(authors), 1)
            self.assertTrue(
                any(
                    payload["record_type"] == "message"
                    and payload["metadata"].get("mock_data") is True
                    for payload in payloads
                )
            )


if __name__ == "__main__":
    unittest.main()
