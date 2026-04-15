from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from crawler.feature_crawler.base import CrawlContext
from crawler.feature_crawler.cli import main
from crawler.feature_crawler.models import CommunityRecord, EvidencePointer


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "approved_web_sample.html"
)


class CliTest(unittest.TestCase):
    def test_crawl_web_command_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main(
                [
                    "crawl-web",
                    "--url",
                    FIXTURE_PATH.as_uri(),
                    "--output-dir",
                    tmpdir,
                    "--collection-basis",
                    "owned",
                    "--allow-persona-inference",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(list(Path(tmpdir).rglob("*.jsonl")))

    @patch(
        "crawler.feature_crawler.platforms.linkedin.runner.JsonlSink.write",
        return_value=Path("/tmp/linkedin-official-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.linkedin.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_li_003",
            observed_at="2026-04-11T16:00:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.linkedin.runner.LinkedInOfficialConnector.from_env")
    def test_crawl_linkedin_official_mode_uses_access_token_env(
        self,
        from_env_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_env_mock.return_value.fetch.return_value = [
            CommunityRecord(
                record_type="community",
                platform="linkedin",
                community_id="jane-builder",
                community_name="Jane Builder",
                community_type="account",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-11T16:00:00Z",
                crawl_run_id="run_test_li_003",
                evidence_pointer=EvidencePointer(
                    source_url="https://api.linkedin.com/v2/userinfo",
                    fetched_at="2026-04-11T16:00:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-linkedin",
                "--url",
                "https://api.linkedin.com/v2/userinfo",
                "--output-dir",
                "/tmp/linkedin-official-out",
                "--mode",
                "official-oidc",
                "--access-token-env",
                "LINKEDIN_ACCESS_TOKEN_CUSTOM",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_env_mock.assert_called_once_with(access_token_env="LINKEDIN_ACCESS_TOKEN_CUSTOM")

    def test_linkedin_auth_url_command_prints_oidc_authorize_url(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "linkedin-auth-url",
                    "--client-id",
                    "client-123",
                    "--redirect-uri",
                    "http://127.0.0.1:8080/callback",
                    "--state",
                    "state-xyz",
                ]
            )
        self.assertEqual(exit_code, 0)
        output = buffer.getvalue().strip()
        self.assertIn("https://www.linkedin.com/oauth/v2/authorization?", output)
        self.assertIn("client_id=client-123", output)
        self.assertIn("state=state-xyz", output)

    @patch("crawler.feature_crawler.platforms.linkedin.runner.LinkedInAuthClient.from_env")
    def test_linkedin_exchange_code_uses_env_config(self, from_env_mock: MagicMock) -> None:
        from_env_mock.return_value.exchange_code.return_value = {"access_token": "token-abc"}
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "linkedin-exchange-code",
                    "--code",
                    "auth-code-1",
                    "--redirect-uri",
                    "http://127.0.0.1:8080/callback",
                    "--client-id-env",
                    "LINKEDIN_CLIENT_ID_CUSTOM",
                    "--client-secret-env",
                    "LINKEDIN_CLIENT_SECRET_CUSTOM",
                ]
            )
        self.assertEqual(exit_code, 0)
        from_env_mock.assert_called_once_with(
            client_id_env="LINKEDIN_CLIENT_ID_CUSTOM",
            client_secret_env="LINKEDIN_CLIENT_SECRET_CUSTOM",
            redirect_uri_env="LINKEDIN_REDIRECT_URI",
            redirect_uri="http://127.0.0.1:8080/callback",
        )
        self.assertIn("token-abc", buffer.getvalue())

    @patch("crawler.feature_crawler.platforms.linkedin.runner.wait_for_callback_once")
    def test_linkedin_wait_for_callback_prints_payload(self, wait_mock: MagicMock) -> None:
        wait_mock.return_value = type(
            "Payload",
            (),
            {
                "code": "abc123",
                "state": "state-xyz",
                "error": None,
                "raw_params": {"code": ["abc123"], "state": ["state-xyz"]},
            },
        )()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "linkedin-wait-for-callback",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8765",
                    "--timeout-seconds",
                    "1",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn('"code": "abc123"', buffer.getvalue())

    @patch(
        "crawler.feature_crawler.platforms.linkedin.runner.JsonlSink.write",
        return_value=Path("/tmp/linkedin-vendor-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.linkedin.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_li_004",
            observed_at="2026-04-11T16:30:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.linkedin.runner.LinkedInVendorConnector.from_env")
    def test_crawl_linkedin_vendor_mode_uses_vendor_connector(
        self,
        from_env_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_env_mock.return_value.fetch.return_value = [
            CommunityRecord(
                record_type="community",
                platform="linkedin",
                community_id="jane-builder",
                community_name="Jane Builder",
                community_type="account",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-11T16:30:00Z",
                crawl_run_id="run_test_li_004",
                evidence_pointer=EvidencePointer(
                    source_url="https://www.linkedin.com/in/jane-builder/",
                    fetched_at="2026-04-11T16:30:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-linkedin",
                "--url",
                "https://www.linkedin.com/in/jane-builder/",
                "--output-dir",
                "/tmp/linkedin-vendor-out",
                "--mode",
                "linkdapi",
                "--scope",
                "profile,activity,network",
                "--activity-limit",
                "20",
                "--network-limit",
                "10",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_env_mock.assert_called_once_with(vendor="linkdapi")

    @patch(
        "crawler.feature_crawler.platforms.linkedin.runner.JsonlSink.write",
        return_value=Path("/tmp/linkedin-browser-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.linkedin.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_li_005",
            observed_at="2026-04-11T17:00:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.linkedin.runner.LinkedInBrowserConnector.from_env")
    def test_crawl_linkedin_session_browser_mode_uses_browser_connector(
        self,
        from_env_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_env_mock.return_value.fetch.return_value = [
            CommunityRecord(
                record_type="community",
                platform="linkedin",
                community_id="jane-builder",
                community_name="Jane Builder",
                community_type="account",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-11T17:00:00Z",
                crawl_run_id="run_test_li_005",
                evidence_pointer=EvidencePointer(
                    source_url="https://www.linkedin.com/in/jane-builder/",
                    fetched_at="2026-04-11T17:00:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-linkedin",
                "--url",
                "https://www.linkedin.com/in/jane-builder/",
                "--output-dir",
                "/tmp/linkedin-browser-out",
                "--mode",
                "session-browser",
                "--scope",
                "profile,network",
                "--cookie-env",
                "LINKEDIN_COOKIE_CUSTOM",
                "--li-at-env",
                "LINKEDIN_LI_AT_CUSTOM",
                "--jsessionid-env",
                "LINKEDIN_JSESSIONID_CUSTOM",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_env_mock.assert_called_once_with(
            cookie_env="LINKEDIN_COOKIE_CUSTOM",
            li_at_env="LINKEDIN_LI_AT_CUSTOM",
            jsessionid_env="LINKEDIN_JSESSIONID_CUSTOM",
        )

    def test_crawl_linkedin_command_writes_output(self) -> None:
        fixture = (
            Path(__file__).resolve().parent / "fixtures" / "linkedin_profile_sample.html"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main(
                [
                    "crawl-linkedin",
                    "--url",
                    fixture.as_uri(),
                    "--output-dir",
                    tmpdir,
                    "--collection-basis",
                    "consented",
                    "--allow-persona-inference",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(list(Path(tmpdir).rglob("*.jsonl")))

    @patch(
        "crawler.feature_crawler.platforms.discord.runner.JsonlSink.write",
        return_value=Path("/tmp/discord-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.discord.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_001",
            observed_at="2026-04-09T12:00:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.discord.runner.DiscordUserApiConnector.from_env")
    def test_crawl_discord_user_passes_delay_flags(
        self,
        from_env_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_env_mock.return_value.fetch.return_value = [
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
                crawl_run_id="run_test_001",
                evidence_pointer=EvidencePointer(
                    source_url="https://discord.com/channels/guild_1",
                    fetched_at="2026-04-09T12:00:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-discord-user",
                "--guild-id",
                "guild_1",
                "--channel-id",
                "channel_1",
                "--output-dir",
                "/tmp/discord-out",
                "--min-delay",
                "2.5",
                "--max-delay",
                "4.0",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_env_mock.assert_called_once_with(
            token_env="DISCORD_USER_TOKEN",
            min_delay=2.5,
            max_delay=4.0,
        )

    @patch(
        "crawler.feature_crawler.platforms.reddit.runner.JsonlSink.write",
        return_value=Path("/tmp/reddit-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.reddit.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_003",
            observed_at="2026-04-09T12:00:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.reddit.runner.RedditApiConnector.from_env")
    def test_crawl_reddit_command_runs_official_connector(
        self,
        from_env_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_env_mock.return_value.fetch.return_value = [
            CommunityRecord(
                record_type="community",
                platform="reddit",
                community_id="python",
                community_name="r/python",
                community_type="subreddit",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_003",
                evidence_pointer=EvidencePointer(
                    source_url="https://www.reddit.com/r/python/",
                    fetched_at="2026-04-09T12:00:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-reddit",
                "--subreddit",
                "python",
                "--output-dir",
                "/tmp/reddit-out",
                "--auth-mode",
                "oauth",
                "--sort",
                "new",
                "--limit",
                "10",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_env_mock.assert_called_once_with(
            client_id_env="REDDIT_CLIENT_ID",
            client_secret_env="REDDIT_CLIENT_SECRET",
            user_agent_env="REDDIT_USER_AGENT",
        )

    @patch(
        "crawler.feature_crawler.platforms.reddit.runner.JsonlSink.write",
        return_value=Path("/tmp/reddit-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.reddit.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_006",
            observed_at="2026-04-09T12:00:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.reddit.runner.RedditApiConnector.from_public_json")
    def test_crawl_reddit_public_json_mode_uses_no_auth_connector(
        self,
        from_public_json_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_public_json_mock.return_value.fetch.return_value = [
            CommunityRecord(
                record_type="community",
                platform="reddit",
                community_id="webscraping",
                community_name="r/webscraping",
                community_type="subreddit",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_006",
                evidence_pointer=EvidencePointer(
                    source_url="https://www.reddit.com/r/webscraping/",
                    fetched_at="2026-04-09T12:00:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-reddit",
                "--subreddit",
                "webscraping",
                "--output-dir",
                "/tmp/reddit-out",
                "--auth-mode",
                "public-json",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_public_json_mock.assert_called_once_with(user_agent_env="REDDIT_USER_AGENT")

    @patch(
        "crawler.feature_crawler.platforms.reddit.runner.JsonlSink.write",
        return_value=Path("/tmp/reddit-out.jsonl"),
    )
    @patch(
        "crawler.feature_crawler.platforms.reddit.runner.CrawlContext.create",
        return_value=CrawlContext(
            crawl_run_id="run_test_007",
            observed_at="2026-04-09T12:00:00Z",
        ),
    )
    @patch("crawler.feature_crawler.platforms.reddit.runner.RedditApiConnector.from_public_json")
    @patch.dict(os.environ, {}, clear=True)
    def test_crawl_reddit_auto_falls_back_to_public_json_without_creds(
        self,
        from_public_json_mock: MagicMock,
        _context_mock: MagicMock,
        _sink_write_mock: MagicMock,
    ) -> None:
        from_public_json_mock.return_value.fetch.return_value = [
            CommunityRecord(
                record_type="community",
                platform="reddit",
                community_id="python",
                community_name="r/python",
                community_type="subreddit",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at="2026-04-09T12:00:00Z",
                crawl_run_id="run_test_007",
                evidence_pointer=EvidencePointer(
                    source_url="https://www.reddit.com/r/python/",
                    fetched_at="2026-04-09T12:00:00Z",
                ),
            )
        ]

        exit_code = main(
            [
                "crawl-reddit",
                "--subreddit",
                "python",
                "--output-dir",
                "/tmp/reddit-out",
            ]
        )

        self.assertEqual(exit_code, 0)
        from_public_json_mock.assert_called_once_with(user_agent_env="REDDIT_USER_AGENT")


if __name__ == "__main__":
    unittest.main()
