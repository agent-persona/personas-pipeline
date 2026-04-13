from __future__ import annotations

import unittest

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.connectors.live_discord import DiscordApiClient, DiscordApiConnector
from crawler.feature_crawler.models import CrawlTarget


class LiveDiscordConnectorTest(unittest.TestCase):
    def test_connector_emits_records_from_api_payloads(self) -> None:
        payloads = {
            "/guilds/guild_1": {"id": "guild_1", "name": "Fortnite Lab"},
            "/channels/channel_1": {
                "id": "channel_1",
                "guild_id": "guild_1",
                "name": "build-meta",
                "type": 0,
            },
        }

        def fake_fetch(path: str):
            if path == "/guilds/guild_1/channels":
                return [payloads["/channels/channel_1"]]
            if path.startswith("/channels/channel_1/messages?"):
                if "after=1446100000000000000" in path:
                    return []
                return [
                    {
                        "id": "1446000000000000000",
                        "timestamp": "2026-04-03T12:00:00+00:00",
                        "content": "Turbo build timing feels different after the patch.",
                        "attachments": [],
                        "author": {"id": "user_1", "username": "alpha"},
                    },
                    {
                        "id": "1446100000000000000",
                        "timestamp": "2026-04-03T12:10:00+00:00",
                        "content": "Agreed, endgame edits are less consistent.",
                        "attachments": [],
                        "author": {"id": "user_2", "username": "beta"},
                        "message_reference": {"message_id": "1446000000000000000"},
                        "referenced_message": {
                            "author": {"id": "user_1", "username": "alpha"},
                        },
                    },
                ]
            return payloads[path]

        connector = DiscordApiConnector(DiscordApiClient(token="test", fetch_json=fake_fetch))
        target = CrawlTarget(
            platform="discord",
            target_id="guild_1",
            url="https://discord.com/channels/guild_1",
            community_name="Fortnite Lab",
            collection_basis="consented",
            allow_persona_inference=True,
            metadata={
                "guild_id": "guild_1",
                "channel_ids": ["channel_1"],
                "thread_ids": [],
                "until": "2026-04-09T23:59:59Z",
                "message_limit": 100,
            },
        )
        context = CrawlContext(crawl_run_id="run_test_001", observed_at="2026-04-09T12:00:00Z")

        records = list(connector.fetch(target=target, context=context, since="2026-04-01T00:00:00Z"))
        record_types = [record.record_type for record in records]
        self.assertIn("thread", record_types)
        self.assertEqual(record_types.count("message"), 2)
        self.assertEqual(record_types.count("interaction"), 1)


if __name__ == "__main__":
    unittest.main()
