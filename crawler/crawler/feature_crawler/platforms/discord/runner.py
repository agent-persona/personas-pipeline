from __future__ import annotations

import argparse
from pathlib import Path

from ...core.base import CrawlContext
from ...core.models import CrawlTarget
from ...core.sink import JsonlSink
from .connector_api import DiscordApiConnector
from .connector_archive import DiscordArchiveConnector
from .connector_browser import DiscordBrowserConnector
from .connector_user_api import DiscordUserApiConnector


def register_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> dict[str, callable]:
    crawl_discord = subparsers.add_parser(
        "crawl-discord",
        help="Crawl Discord via the real bot API and write records to local JSONL.",
    )
    crawl_discord.add_argument("--guild-id", required=True)
    crawl_discord.add_argument("--output-dir", required=True)
    crawl_discord.add_argument("--community-name", default=None)
    crawl_discord.add_argument("--channel-id", action="append", default=[])
    crawl_discord.add_argument("--thread-id", action="append", default=[])
    crawl_discord.add_argument("--token-env", default="DISCORD_BOT_TOKEN")
    crawl_discord.add_argument("--since", default=None)
    crawl_discord.add_argument("--until", default=None)
    crawl_discord.add_argument("--message-limit", type=int, default=500)
    crawl_discord.add_argument(
        "--collection-basis",
        default="consented",
        choices=["owned", "consented", "public-permitted", "blocked"],
    )
    crawl_discord.add_argument("--allow-persona-inference", action="store_true")

    crawl_user = subparsers.add_parser(
        "crawl-discord-user",
        help="Crawl Discord via user-token API (public servers you've joined).",
    )
    crawl_user.add_argument("--guild-id", required=True)
    crawl_user.add_argument("--output-dir", required=True)
    crawl_user.add_argument("--community-name", default=None)
    crawl_user.add_argument("--channel-id", action="append", default=[])
    crawl_user.add_argument("--thread-id", action="append", default=[])
    crawl_user.add_argument("--token-env", default="DISCORD_USER_TOKEN")
    crawl_user.add_argument("--since", default=None)
    crawl_user.add_argument("--until", default=None)
    crawl_user.add_argument("--message-limit", type=int, default=200)
    crawl_user.add_argument("--min-delay", type=float, default=1.5)
    crawl_user.add_argument("--max-delay", type=float, default=3.0)
    crawl_user.add_argument(
        "--collection-basis",
        default="public-permitted",
        choices=["owned", "consented", "public-permitted"],
    )
    crawl_user.add_argument("--allow-persona-inference", action="store_true")
    crawl_user.add_argument("--join-invite", default=None)

    crawl_browser = subparsers.add_parser(
        "crawl-discord-browser",
        help="Crawl Discord via browser automation (Playwright).",
    )
    crawl_browser.add_argument("--url", required=True)
    crawl_browser.add_argument("--output-dir", required=True)
    crawl_browser.add_argument("--community-name", default=None)
    crawl_browser.add_argument("--storage-state", default=None)
    crawl_browser.add_argument("--headless", action="store_true", default=False)
    crawl_browser.add_argument("--max-scrolls", type=int, default=50)
    crawl_browser.add_argument("--scroll-pause", type=float, default=2.0)
    crawl_browser.add_argument("--since", default=None)
    crawl_browser.add_argument("--until", default=None)
    crawl_browser.add_argument(
        "--collection-basis",
        default="public-permitted",
        choices=["owned", "consented", "public-permitted"],
    )
    crawl_browser.add_argument("--allow-persona-inference", action="store_true")

    crawl_archive = subparsers.add_parser(
        "crawl-discord-archive",
        help="Crawl Discord via third-party archive services.",
    )
    crawl_archive.add_argument("--guild-id", required=True)
    crawl_archive.add_argument("--channel-id", required=True)
    crawl_archive.add_argument("--output-dir", required=True)
    crawl_archive.add_argument("--community-name", default=None)
    crawl_archive.add_argument("--since", default=None)
    crawl_archive.add_argument("--request-delay", type=float, default=2.0)
    crawl_archive.add_argument(
        "--collection-basis",
        default="public-permitted",
        choices=["owned", "consented", "public-permitted"],
    )
    crawl_archive.add_argument("--allow-persona-inference", action="store_true")

    save_login = subparsers.add_parser(
        "save-discord-login",
        help="Launch browser to log in to Discord and save session state.",
    )
    save_login.add_argument("--storage-state", required=True)

    return {
        "crawl-discord": run_discord_cli,
        "crawl-discord-user": run_discord_user_cli,
        "crawl-discord-browser": run_discord_browser_cli,
        "crawl-discord-archive": run_discord_archive_cli,
        "save-discord-login": run_save_login_cli,
    }


def _build_target(
    *,
    guild_id: str,
    community_name: str | None,
    collection_basis: str,
    allow_persona_inference: bool,
    url: str,
    metadata: dict[str, object],
) -> CrawlTarget:
    return CrawlTarget(
        platform="discord",
        target_id=guild_id,
        url=url,
        community_name=community_name or guild_id,
        collection_basis=collection_basis,
        allow_persona_inference=allow_persona_inference,
        metadata=metadata,
    )


def run_discord_cli(args: argparse.Namespace) -> int:
    if not args.channel_id and not args.thread_id:
        raise SystemExit("crawl-discord requires at least one --channel-id or --thread-id")
    target = _build_target(
        guild_id=args.guild_id,
        community_name=args.community_name,
        collection_basis=args.collection_basis,
        allow_persona_inference=args.allow_persona_inference,
        url=f"https://discord.com/channels/{args.guild_id}",
        metadata={
            "guild_id": args.guild_id,
            "channel_ids": args.channel_id,
            "thread_ids": args.thread_id,
            "until": args.until,
            "message_limit": args.message_limit,
        },
    )
    connector = DiscordApiConnector.from_env(token_env=args.token_env)
    context = CrawlContext.create()
    records = list(connector.fetch(target=target, context=context, since=args.since))
    sink = JsonlSink(Path(args.output_dir))
    output_path = sink.write(records)
    print(f"wrote {len(records)} records to {output_path}")
    return 0


def run_discord_user_cli(args: argparse.Namespace) -> int:
    if not args.channel_id and not args.thread_id:
        raise SystemExit("crawl-discord-user requires at least one --channel-id or --thread-id")
    target = _build_target(
        guild_id=args.guild_id,
        community_name=args.community_name,
        collection_basis=args.collection_basis,
        allow_persona_inference=args.allow_persona_inference,
        url=f"https://discord.com/channels/{args.guild_id}",
        metadata={
            "guild_id": args.guild_id,
            "channel_ids": args.channel_id,
            "thread_ids": args.thread_id,
            "until": args.until,
            "message_limit": args.message_limit,
        },
    )
    connector = DiscordUserApiConnector.from_env(
        token_env=args.token_env,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )
    if args.join_invite:
        result = connector.join_server(args.join_invite)
        print(f"joined server via invite: {result.get('guild', {}).get('name', 'unknown')}")
    context = CrawlContext.create()
    records = list(connector.fetch(target=target, context=context, since=args.since))
    sink = JsonlSink(Path(args.output_dir))
    output_path = sink.write(records)
    print(f"wrote {len(records)} records to {output_path}")
    return 0


def run_discord_browser_cli(args: argparse.Namespace) -> int:
    url_parts = args.url.rstrip("/").split("/")
    guild_id = url_parts[-2] if len(url_parts) >= 2 else "unknown"
    channel_id = url_parts[-1] if len(url_parts) >= 1 else "unknown"
    target = _build_target(
        guild_id=guild_id,
        community_name=args.community_name,
        collection_basis=args.collection_basis,
        allow_persona_inference=args.allow_persona_inference,
        url=args.url,
        metadata={
            "guild_id": guild_id,
            "channel_ids": [channel_id],
            "since": args.since,
            "until": args.until,
        },
    )
    connector = DiscordBrowserConnector(
        headless=args.headless,
        scroll_pause=args.scroll_pause,
        max_scrolls=args.max_scrolls,
        storage_state_path=args.storage_state,
    )
    context = CrawlContext.create()
    records = list(connector.fetch(target=target, context=context, since=args.since))
    if records:
        sink = JsonlSink(Path(args.output_dir))
        output_path = sink.write(records)
        print(f"wrote {len(records)} records to {output_path}")
    else:
        print("no records extracted")
    return 0


def run_discord_archive_cli(args: argparse.Namespace) -> int:
    target = _build_target(
        guild_id=args.guild_id,
        community_name=args.community_name,
        collection_basis=args.collection_basis,
        allow_persona_inference=args.allow_persona_inference,
        url=f"https://discord.com/channels/{args.guild_id}/{args.channel_id}",
        metadata={
            "guild_id": args.guild_id,
            "channel_id": args.channel_id,
            "channel_ids": [args.channel_id],
            "since": args.since,
        },
    )
    connector = DiscordArchiveConnector(request_delay=args.request_delay)
    context = CrawlContext.create()
    records = list(connector.fetch(target=target, context=context, since=args.since))
    if records:
        sink = JsonlSink(Path(args.output_dir))
        output_path = sink.write(records)
        print(f"wrote {len(records)} records to {output_path}")
    else:
        print("no archived records found; try crawl-discord-user or crawl-discord-browser instead")
    return 0


def run_save_login_cli(args: argparse.Namespace) -> int:
    DiscordBrowserConnector.save_login_state(args.storage_state)
    print(f"login state saved to {args.storage_state}")
    return 0
