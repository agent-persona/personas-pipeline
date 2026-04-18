from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.models import (
    CommunityRecord,
    EvidencePointer,
    MessageRecord,
    ThreadRecord,
)
from crawler.feature_crawler.sink import JsonlSink


def _pointer(source_url: str) -> EvidencePointer:
    return EvidencePointer(
        source_url=source_url,
        fetched_at="2026-04-09T12:00:00Z",
    )


class JsonlSinkTest(unittest.TestCase):
    def test_web_sink_keeps_platform_community_date_layout(self) -> None:
        records = [
            CommunityRecord(
                record_type="community",
                platform="web",
                community_id="roblox-devforum",
                community_name="Roblox DevForum",
                community_type="forum",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_001",
                evidence_pointer=_pointer("https://devforum.roblox.com"),
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = JsonlSink(Path(tmpdir)).write(records)
            expected = (
                Path(tmpdir)
                / "web"
                / "roblox-devforum"
                / "2026-04-09"
                / "run_test_001.jsonl"
            )
            self.assertEqual(output_path, expected)

    def test_discord_sink_uses_server_channel_and_nested_date(self) -> None:
        records = [
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id="662267976984297473",
                community_name="Midjourney",
                community_type="server",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_002",
                evidence_pointer=_pointer("https://discord.com/channels/662267976984297473"),
            ),
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id="938713143759216720",
                community_name="discussion",
                community_type="channel",
                parent_community_id="662267976984297473",
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_002",
                evidence_pointer=_pointer(
                    "https://discord.com/channels/662267976984297473/938713143759216720"
                ),
            ),
            ThreadRecord(
                record_type="thread",
                platform="discord",
                thread_id="discord-thread-938713143759216720-run_test_002",
                community_id="938713143759216720",
                title="discussion timeline",
                author_platform_user_id=None,
                created_at="2026-04-09T12:00:00Z",
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_002",
                metadata={"channel_id": "938713143759216720"},
                evidence_pointer=_pointer(
                    "https://discord.com/channels/662267976984297473/938713143759216720"
                ),
            ),
            MessageRecord(
                record_type="message",
                platform="discord",
                message_id="message_1",
                thread_id="discord-thread-938713143759216720-run_test_002",
                community_id="938713143759216720",
                author_platform_user_id="user_1",
                body="Need crawl partitions by channel.",
                created_at="2026-04-09T12:00:00Z",
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_002",
                reply_to_message_id=None,
                reply_to_user_id=None,
                metadata={},
                evidence_pointer=_pointer(
                    "https://discord.com/channels/662267976984297473/938713143759216720/message_1"
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = JsonlSink(Path(tmpdir)).write(records)
            expected = (
                Path(tmpdir)
                / "discord"
                / "midjourney"
                / "discussion_938713143759216720"
                / "4"
                / "9"
                / "2026"
                / "run_test_002.jsonl"
            )
            self.assertEqual(output_path, expected)

    def test_discord_sink_falls_back_to_multi_channel_folder(self) -> None:
        records = [
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id="guild_1",
                community_name="Midjourney",
                community_type="server",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_003",
                evidence_pointer=_pointer("https://discord.com/channels/guild_1"),
            ),
            MessageRecord(
                record_type="message",
                platform="discord",
                message_id="message_1",
                thread_id="thread_1",
                community_id="channel_1",
                author_platform_user_id="user_1",
                body="first",
                created_at="2026-04-09T12:00:00Z",
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_003",
                reply_to_message_id=None,
                reply_to_user_id=None,
                metadata={},
                evidence_pointer=_pointer("https://discord.com/channels/guild_1/channel_1/message_1"),
            ),
            MessageRecord(
                record_type="message",
                platform="discord",
                message_id="message_2",
                thread_id="thread_2",
                community_id="channel_2",
                author_platform_user_id="user_2",
                body="second",
                created_at="2026-04-09T12:00:00Z",
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_003",
                reply_to_message_id=None,
                reply_to_user_id=None,
                metadata={},
                evidence_pointer=_pointer("https://discord.com/channels/guild_1/channel_2/message_2"),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = JsonlSink(Path(tmpdir)).write(records)
            expected = (
                Path(tmpdir)
                / "discord"
                / "midjourney"
                / "multi-channel"
                / "4"
                / "9"
                / "2026"
                / "run_test_003.jsonl"
            )
            self.assertEqual(output_path, expected)


if __name__ == "__main__":
    unittest.main()
