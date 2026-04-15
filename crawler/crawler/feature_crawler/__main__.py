from __future__ import annotations

import argparse
from pathlib import Path

from .connectors import DiscordSeed, DiscordSeedConnector, WebSeed, WebSeedConnector
from .connectors.discord import DiscordSeedMessage
from .connectors.web import WebSeedComment
from .pipeline import BronzeWriter, CrawlTarget, CrawlerRunner
from .policy import CollectionBasis, PolicyRegistry


def _demo_discord() -> tuple[DiscordSeedConnector, CrawlTarget]:
    seed = DiscordSeed(
        run_id="run_20260409_demo_discord",
        observed_at="2026-04-09T12:00:00Z",
        server_id="guild_1",
        server_name="agent-persona-lab",
        channel_id="channel_1",
        channel_name="persona-builders",
        thread_id="thread_1",
        thread_title="How should we score persona drift?",
        thread_author_id="user_a",
        thread_created_at="2026-04-08T18:00:00Z",
        invite_url="https://discord.gg/agentpersona",
        messages=[
            DiscordSeedMessage(
                message_id="msg_1",
                author_id="user_a",
                username="max",
                body="Need drift precision, not just stability.",
                created_at="2026-04-08T18:01:00Z",
                observed_at="2026-04-09T12:00:00Z",
            ),
            DiscordSeedMessage(
                message_id="msg_2",
                author_id="user_b",
                username="shruti",
                body="Agree. noise should not mutate the persona.",
                created_at="2026-04-08T18:02:00Z",
                observed_at="2026-04-09T12:00:00Z",
                reply_to_message_id="msg_1",
                reply_to_user_id="user_a",
            ),
        ],
    )
    target = CrawlTarget(
        platform="discord",
        community_id=seed.server_id,
        community_name=seed.server_name,
        collection_basis=CollectionBasis.CONSENTED,
        source_url=seed.invite_url,
    )
    return DiscordSeedConnector(seed), target


def _demo_web() -> tuple[WebSeedConnector, CrawlTarget]:
    seed = WebSeed(
        run_id="run_20260409_demo_web",
        observed_at="2026-04-09T12:05:00Z",
        site_id="forum_1",
        site_name="approved-forum",
        source_url="https://example.com/forum/persona-drift",
        thread_id="page_1",
        thread_title="Persona drift notes",
        thread_author_id="author_1",
        thread_created_at="2026-04-07T09:00:00Z",
        comments=[
            WebSeedComment(
                message_id="comment_1",
                author_id="author_1",
                author_name="alex",
                body="Stable traits should move slowly.",
                created_at="2026-04-07T09:00:00Z",
                observed_at="2026-04-09T12:05:00Z",
            ),
            WebSeedComment(
                message_id="comment_2",
                author_id="author_2",
                author_name="casey",
                body="Ephemeral state can change by session.",
                created_at="2026-04-07T09:05:00Z",
                observed_at="2026-04-09T12:05:00Z",
                reply_to_message_id="comment_1",
                reply_to_user_id="author_1",
            ),
        ],
    )
    target = CrawlTarget(
        platform="web",
        community_id=seed.site_id,
        community_name=seed.site_name,
        collection_basis=CollectionBasis.PUBLIC_PERMITTED,
        source_url=seed.source_url,
    )
    return WebSeedConnector(seed), target


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the feature_crawler bronze demo.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for bronze output")
    parser.add_argument(
        "--connector",
        choices=("discord", "web", "all"),
        default="all",
        help="Demo connector to run",
    )
    args = parser.parse_args()

    runner = CrawlerRunner(PolicyRegistry(), BronzeWriter(args.output))
    demos = []
    if args.connector in {"discord", "all"}:
        demos.append(_demo_discord())
    if args.connector in {"web", "all"}:
        demos.append(_demo_web())

    for connector, target in demos:
        result = runner.run(connector, target)
        print(f"{target.platform}: {result.root}")
        for record_type, count in sorted(result.record_counts.items()):
            print(f"  {record_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
