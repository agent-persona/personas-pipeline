from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.connectors.discord_user_api import DiscordUserApiConnector
from crawler.feature_crawler.models import CrawlTarget


class DiscordUserApiConnectorTest(unittest.TestCase):
    def test_fetch_respects_message_limit_metadata(self) -> None:
        client = MagicMock()
        client.get.side_effect = [
            {"id": "guild_1", "name": "Midjourney"},
            {"id": "channel_1", "name": "discussion", "type": 0},
            {"threads": []},
        ]
        connector = DiscordUserApiConnector(client=client)
        connector._iter_channel_messages = MagicMock(return_value=[])  # type: ignore[method-assign]

        target = CrawlTarget(
            platform="discord",
            target_id="guild_1",
            url="https://discord.com/channels/guild_1",
            community_name="Midjourney",
            collection_basis="public-permitted",
            allow_persona_inference=True,
            metadata={
                "guild_id": "guild_1",
                "channel_ids": ["channel_1"],
                "thread_ids": [],
                "message_limit": 17,
            },
        )
        context = CrawlContext(
            crawl_run_id="run_test_001",
            observed_at="2026-04-09T12:00:00Z",
        )

        list(connector.fetch(target=target, context=context))

        connector._iter_channel_messages.assert_called_once_with(  # type: ignore[attr-defined]
            "channel_1",
            "",
            "",
            limit=17,
        )


if __name__ == "__main__":
    unittest.main()
