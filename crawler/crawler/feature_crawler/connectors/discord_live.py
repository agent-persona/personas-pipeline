"""Live Discord connector — connects to Discord API via bot token and crawls
guild → channels → threads → messages, producing typed BronzeRecords.

Usage:
    python -m feature_crawler.crawler.connectors.discord_live \
        --token BOT_TOKEN \
        --guild 123456789 \
        --output ./crawl_output \
        --limit 500

Or via Doppler:
    doppler run --project api_keys --config dev -- \
        python -m feature_crawler.crawler.connectors.discord_live \
            --guild 123456789 --output ./crawl_output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

# Avoid import collision with sibling discord.py (the seed connector).
# Temporarily remove this directory from sys.path, import the real discord
# package, then restore.
_this_dir = str(Path(__file__).resolve().parent)
_parent_dir = str(Path(__file__).resolve().parents[1])
_orig_path = sys.path[:]
sys.path = [p for p in sys.path if p not in (_this_dir, _parent_dir, "")]
# Also remove any cached local "discord" module
sys.modules.pop("discord", None)
import discord  # noqa: E402 — now resolves to the real discord.py package
sys.path = _orig_path

# Resolve imports — try relative first, fall back to sys.path manipulation.
try:
    from ..records import (
        AccountRecord,
        BronzeRecord,
        CommunityRecord,
        EvidencePointer,
        InteractionRecord,
        MessageRecord,
        ProfileSnapshotRecord,
        ThreadRecord,
    )
except (ImportError, SystemError):
    # Standalone mode: add parent dirs to path, import records directly
    _crawler_dir = str(Path(__file__).resolve().parent.parent)
    if _crawler_dir not in sys.path:
        sys.path.insert(0, _crawler_dir)
    from records import (  # type: ignore[no-redef]
        AccountRecord,
        BronzeRecord,
        CommunityRecord,
        EvidencePointer,
        InteractionRecord,
        MessageRecord,
        ProfileSnapshotRecord,
        ThreadRecord,
    )

log = logging.getLogger("discord_live")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snowflake_time(snowflake_id: int) -> datetime:
    """Extract creation time from a Discord Snowflake ID."""
    return datetime.fromtimestamp(((snowflake_id >> 22) + 1420070400000) / 1000, tz=UTC)


def _source_url(guild_id: int, channel_id: int | None = None, message_id: int | None = None) -> str:
    base = f"https://discord.com/channels/{guild_id}"
    if channel_id:
        base += f"/{channel_id}"
    if message_id:
        base += f"/{message_id}"
    return base


# ---------------------------------------------------------------------------
# Crawl metrics (in-memory for run summary)
# ---------------------------------------------------------------------------

class CrawlMetrics:
    def __init__(self) -> None:
        self.records_by_type: Counter[str] = Counter()
        self.channels_crawled = 0
        self.threads_crawled = 0
        self.messages_fetched = 0
        self.bot_messages_filtered = 0
        self.rate_limit_hits = 0
        self.null_body_messages = 0
        self.unique_users: set[str] = set()
        self.started_at = datetime.now(UTC)
        self.errors: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": _iso(self.started_at),
            "completed_at": _iso(datetime.now(UTC)),
            "duration_seconds": (datetime.now(UTC) - self.started_at).total_seconds(),
            "channels_crawled": self.channels_crawled,
            "threads_crawled": self.threads_crawled,
            "messages_fetched": self.messages_fetched,
            "bot_messages_filtered": self.bot_messages_filtered,
            "rate_limit_hits": self.rate_limit_hits,
            "null_body_messages": self.null_body_messages,
            "unique_users": len(self.unique_users),
            "records_by_type": dict(self.records_by_type),
            "errors": self.errors[:50],
        }


# ---------------------------------------------------------------------------
# Live connector
# ---------------------------------------------------------------------------

class DiscordLiveConnector:
    """Connects to Discord via bot token and crawls a guild into BronzeRecords."""

    platform = "discord"

    def __init__(
        self,
        token: str,
        *,
        message_limit: int | None = None,
        crawl_threads: bool = True,
        filter_bots: bool = True,
    ) -> None:
        self.token = token
        self.message_limit = message_limit
        self.crawl_threads = crawl_threads
        self.filter_bots = filter_bots

    async def crawl_guild(self, guild_id: int) -> tuple[list[BronzeRecord], CrawlMetrics]:
        """Crawl a single guild. Returns (records, metrics)."""

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        client = discord.Client(intents=intents)
        records: list[BronzeRecord] = []
        metrics = CrawlMetrics()
        seen_users: set[str] = set()

        now = datetime.now(UTC)
        run_id = now.strftime("run_%Y%m%d_%H%M%S")
        observed_at = _iso(now)

        async def _process_guild():
            await client.wait_until_ready()

            guild = client.get_guild(guild_id)
            if guild is None:
                try:
                    guild = await client.fetch_guild(guild_id)
                except discord.errors.Forbidden:
                    metrics.errors.append(f"Cannot access guild {guild_id} — bot not invited or missing permissions")
                    await client.close()
                    return
                except discord.errors.NotFound:
                    metrics.errors.append(f"Guild {guild_id} not found")
                    await client.close()
                    return

            log.info("Connected to guild: %s (%d)", guild.name, guild.id)

            # ── Community record: server ──
            records.append(CommunityRecord(
                record_type="community",
                platform=self.platform,
                crawl_run_id=run_id,
                community_id=str(guild.id),
                community_name=guild.name,
                community_type="server",
                parent_community_id=None,
                description=guild.description,
                member_count=guild.member_count,
                rules_summary=None,
                observed_at=observed_at,
                evidence_pointer=EvidencePointer(
                    source_url=_source_url(guild.id),
                    fetched_at=observed_at,
                ),
            ))
            metrics.records_by_type["community"] += 1

            # ── Crawl each text channel ──
            for channel in guild.text_channels:
                try:
                    await self._crawl_channel(
                        channel, guild, run_id, observed_at,
                        records, metrics, seen_users,
                    )
                except discord.errors.Forbidden:
                    log.warning("No access to #%s — skipping", channel.name)
                    metrics.errors.append(f"Forbidden: #{channel.name} ({channel.id})")
                except Exception as exc:
                    log.error("Error crawling #%s: %s", channel.name, exc)
                    metrics.errors.append(f"Error in #{channel.name}: {exc}")

            # ── Crawl forum channels ──
            for channel in guild.forums:
                try:
                    await self._crawl_forum(
                        channel, guild, run_id, observed_at,
                        records, metrics, seen_users,
                    )
                except discord.errors.Forbidden:
                    log.warning("No access to forum #%s — skipping", channel.name)
                except Exception as exc:
                    log.error("Error crawling forum #%s: %s", channel.name, exc)
                    metrics.errors.append(f"Error in forum #{channel.name}: {exc}")

            await client.close()

        @client.event
        async def on_ready():
            asyncio.get_event_loop().create_task(_process_guild())

        await client.start(self.token)
        return records, metrics

    async def _crawl_channel(
        self,
        channel: discord.TextChannel,
        guild: discord.Guild,
        run_id: str,
        observed_at: str,
        records: list[BronzeRecord],
        metrics: CrawlMetrics,
        seen_users: set[str],
    ) -> None:
        """Crawl a single text channel + its threads."""
        log.info("  Crawling #%s (%d)", channel.name, channel.id)

        # Community record: channel
        records.append(CommunityRecord(
            record_type="community",
            platform=self.platform,
            crawl_run_id=run_id,
            community_id=str(channel.id),
            community_name=channel.name,
            community_type="channel",
            parent_community_id=str(guild.id),
            description=channel.topic,
            member_count=None,
            rules_summary=None,
            observed_at=observed_at,
            evidence_pointer=EvidencePointer(
                source_url=_source_url(guild.id, channel.id),
                fetched_at=observed_at,
            ),
        ))
        metrics.records_by_type["community"] += 1

        # Treat channel as a thread (thread_id = channel_id for non-threaded messages)
        thread_id = str(channel.id)

        # Thread record for the channel itself
        records.append(ThreadRecord(
            record_type="thread",
            platform=self.platform,
            crawl_run_id=run_id,
            thread_id=thread_id,
            community_id=str(channel.id),
            title=f"#{channel.name}",
            author_platform_user_id=str(guild.owner_id) if guild.owner_id else "unknown",
            created_at=_iso(_snowflake_time(channel.id)),
            observed_at=observed_at,
            metadata={"channel_type": "text", "server_id": str(guild.id)},
            evidence_pointer=EvidencePointer(
                source_url=_source_url(guild.id, channel.id),
                fetched_at=observed_at,
            ),
        ))
        metrics.records_by_type["thread"] += 1

        # Fetch messages
        msg_count = 0
        async for message in channel.history(limit=self.message_limit, oldest_first=True):
            self._process_message(
                message, guild, channel.id, thread_id,
                run_id, observed_at, records, metrics, seen_users,
            )
            msg_count += 1

        metrics.channels_crawled += 1
        log.info("    %d messages in #%s", msg_count, channel.name)

        # ── Threads within this channel ──
        if self.crawl_threads:
            await self._crawl_channel_threads(
                channel, guild, run_id, observed_at,
                records, metrics, seen_users,
            )

    async def _crawl_channel_threads(
        self,
        channel: discord.TextChannel,
        guild: discord.Guild,
        run_id: str,
        observed_at: str,
        records: list[BronzeRecord],
        metrics: CrawlMetrics,
        seen_users: set[str],
    ) -> None:
        """Crawl active + archived threads in a channel."""
        threads: list[discord.Thread] = []

        # Active threads
        for thread in channel.threads:
            threads.append(thread)

        # Archived threads (paginated)
        try:
            async for thread in channel.archived_threads(limit=100):
                threads.append(thread)
        except discord.errors.Forbidden:
            log.warning("    Cannot access archived threads in #%s", channel.name)

        for thread in threads:
            log.info("    Thread: %s (%d)", thread.name, thread.id)

            records.append(ThreadRecord(
                record_type="thread",
                platform=self.platform,
                crawl_run_id=run_id,
                thread_id=str(thread.id),
                community_id=str(channel.id),
                title=thread.name,
                author_platform_user_id=str(thread.owner_id) if thread.owner_id else "unknown",
                created_at=_iso(_snowflake_time(thread.id)),
                observed_at=observed_at,
                metadata={
                    "channel_type": "thread",
                    "server_id": str(guild.id),
                    "parent_channel_id": str(channel.id),
                    "archived": thread.archived,
                    "message_count": thread.message_count,
                },
                evidence_pointer=EvidencePointer(
                    source_url=_source_url(guild.id, thread.id),
                    fetched_at=observed_at,
                ),
            ))
            metrics.records_by_type["thread"] += 1

            msg_count = 0
            try:
                async for message in thread.history(limit=self.message_limit, oldest_first=True):
                    self._process_message(
                        message, guild, channel.id, str(thread.id),
                        run_id, observed_at, records, metrics, seen_users,
                    )
                    msg_count += 1
            except discord.errors.Forbidden:
                log.warning("      Cannot read thread %s", thread.name)

            metrics.threads_crawled += 1
            log.info("      %d messages in thread", msg_count)

    async def _crawl_forum(
        self,
        channel: discord.ForumChannel,
        guild: discord.Guild,
        run_id: str,
        observed_at: str,
        records: list[BronzeRecord],
        metrics: CrawlMetrics,
        seen_users: set[str],
    ) -> None:
        """Crawl a forum channel — every post is a thread."""
        log.info("  Crawling forum #%s (%d)", channel.name, channel.id)

        records.append(CommunityRecord(
            record_type="community",
            platform=self.platform,
            crawl_run_id=run_id,
            community_id=str(channel.id),
            community_name=channel.name,
            community_type="forum",
            parent_community_id=str(guild.id),
            description=None,
            member_count=None,
            rules_summary=None,
            observed_at=observed_at,
            evidence_pointer=EvidencePointer(
                source_url=_source_url(guild.id, channel.id),
                fetched_at=observed_at,
            ),
        ))
        metrics.records_by_type["community"] += 1

        threads: list[discord.Thread] = list(channel.threads)
        try:
            async for thread in channel.archived_threads(limit=100):
                threads.append(thread)
        except discord.errors.Forbidden:
            log.warning("    Cannot access archived threads in forum #%s", channel.name)

        for thread in threads:
            log.info("    Forum post: %s (%d)", thread.name, thread.id)

            records.append(ThreadRecord(
                record_type="thread",
                platform=self.platform,
                crawl_run_id=run_id,
                thread_id=str(thread.id),
                community_id=str(channel.id),
                title=thread.name,
                author_platform_user_id=str(thread.owner_id) if thread.owner_id else "unknown",
                created_at=_iso(_snowflake_time(thread.id)),
                observed_at=observed_at,
                metadata={
                    "channel_type": "forum_post",
                    "server_id": str(guild.id),
                    "parent_channel_id": str(channel.id),
                    "archived": thread.archived,
                    "message_count": thread.message_count,
                },
                evidence_pointer=EvidencePointer(
                    source_url=_source_url(guild.id, thread.id),
                    fetched_at=observed_at,
                ),
            ))
            metrics.records_by_type["thread"] += 1

            msg_count = 0
            try:
                async for message in thread.history(limit=self.message_limit, oldest_first=True):
                    self._process_message(
                        message, guild, channel.id, str(thread.id),
                        run_id, observed_at, records, metrics, seen_users,
                    )
                    msg_count += 1
            except discord.errors.Forbidden:
                log.warning("      Cannot read forum thread %s", thread.name)

            metrics.threads_crawled += 1

    def _process_message(
        self,
        message: discord.Message,
        guild: discord.Guild,
        channel_id: int,
        thread_id: str,
        run_id: str,
        observed_at: str,
        records: list[BronzeRecord],
        metrics: CrawlMetrics,
        seen_users: set[str],
    ) -> None:
        """Extract records from a single Discord message."""
        metrics.messages_fetched += 1

        # Filter bots and system messages
        if self.filter_bots and (message.author.bot or message.author.system):
            metrics.bot_messages_filtered += 1
            return

        if message.webhook_id is not None:
            metrics.bot_messages_filtered += 1
            return

        author_id = str(message.author.id)
        msg_url = _source_url(guild.id, message.channel.id, message.id)

        # ── Account + profile snapshot (first encounter per user) ──
        if author_id not in seen_users:
            user_url = _source_url(guild.id) + f"#user-{author_id}"

            # Extract account creation time from Snowflake
            account_created = _iso(_snowflake_time(message.author.id))

            records.append(AccountRecord(
                record_type="account",
                platform=self.platform,
                crawl_run_id=run_id,
                platform_user_id=author_id,
                username=str(message.author),
                account_created_at=account_created,
                first_observed_at=observed_at,
                evidence_pointer=EvidencePointer(source_url=user_url, fetched_at=observed_at),
            ))
            metrics.records_by_type["account"] += 1

            # Profile snapshot with available fields
            snapshot_fields: dict[str, Any] = {
                "display_name": message.author.display_name,
                "username": message.author.name,
                "discriminator": message.author.discriminator,
            }
            if message.author.avatar:
                snapshot_fields["avatar_url"] = str(message.author.avatar.url)
            if hasattr(message.author, "roles") and isinstance(message.author, discord.Member):
                snapshot_fields["roles"] = [r.name for r in message.author.roles if r.name != "@everyone"]
                snapshot_fields["joined_at"] = _iso(message.author.joined_at) if message.author.joined_at else None
                snapshot_fields["nick"] = message.author.nick

            records.append(ProfileSnapshotRecord(
                record_type="profile_snapshot",
                platform=self.platform,
                crawl_run_id=run_id,
                platform_user_id=author_id,
                snapshot_at=observed_at,
                fields=snapshot_fields,
                evidence_pointer=EvidencePointer(source_url=user_url, fetched_at=observed_at),
            ))
            metrics.records_by_type["profile_snapshot"] += 1

            seen_users.add(author_id)
            metrics.unique_users.add(author_id)

        # ── Message body ──
        body = message.content or ""
        if not body.strip():
            # Check for embeds/attachments as fallback content
            if message.embeds:
                body = " ".join(
                    e.description or e.title or "" for e in message.embeds
                ).strip()
            if message.attachments and not body:
                body = " ".join(a.filename for a in message.attachments)
            if not body:
                metrics.null_body_messages += 1
                return  # Skip empty messages — no persona evidence

        # ── Reply reference ──
        reply_to_msg_id: str | None = None
        reply_to_user_id: str | None = None
        if message.reference and message.reference.message_id:
            reply_to_msg_id = str(message.reference.message_id)
            # Try to resolve the referenced message's author
            if message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
                reply_to_user_id = str(message.reference.resolved.author.id)

        created_at = _iso(message.created_at)

        records.append(MessageRecord(
            record_type="message",
            platform=self.platform,
            crawl_run_id=run_id,
            message_id=str(message.id),
            thread_id=thread_id,
            community_id=str(channel_id),
            author_platform_user_id=author_id,
            body=body,
            created_at=created_at,
            observed_at=observed_at,
            reply_to_message_id=reply_to_msg_id,
            reply_to_user_id=reply_to_user_id,
            metadata={
                "has_attachments": bool(message.attachments),
                "has_embeds": bool(message.embeds),
                "mention_count": len(message.mentions),
                "is_pinned": message.pinned,
            },
            evidence_pointer=EvidencePointer(source_url=msg_url, fetched_at=observed_at),
        ))
        metrics.records_by_type["message"] += 1

        # ── Interaction records ──
        # Reply interaction
        if reply_to_user_id and reply_to_user_id != author_id:
            records.append(InteractionRecord(
                record_type="interaction",
                platform=self.platform,
                crawl_run_id=run_id,
                interaction_type="reply",
                source_user_id=author_id,
                target_user_id=reply_to_user_id,
                message_id=str(message.id),
                thread_id=thread_id,
                community_id=str(channel_id),
                created_at=created_at,
                evidence_pointer=EvidencePointer(
                    source_url=msg_url,
                    fetched_at=observed_at,
                    derived_from_message_id=str(message.id),
                ),
            ))
            metrics.records_by_type["interaction"] += 1

        # Mention interactions
        for mentioned in message.mentions:
            if mentioned.id != message.author.id and not mentioned.bot:
                records.append(InteractionRecord(
                    record_type="interaction",
                    platform=self.platform,
                    crawl_run_id=run_id,
                    interaction_type="mention",
                    source_user_id=author_id,
                    target_user_id=str(mentioned.id),
                    message_id=str(message.id),
                    thread_id=thread_id,
                    community_id=str(channel_id),
                    created_at=created_at,
                    evidence_pointer=EvidencePointer(
                        source_url=msg_url,
                        fetched_at=observed_at,
                        derived_from_message_id=str(message.id),
                    ),
                ))
                metrics.records_by_type["interaction"] += 1


# ---------------------------------------------------------------------------
# Output writer (local JSONL, matches existing BronzeWriter format)
# ---------------------------------------------------------------------------

def write_records(
    records: list[BronzeRecord],
    metrics: CrawlMetrics,
    output_dir: Path,
    guild_name: str,
    guild_id: int,
) -> Path:
    """Write records to JSONL files, partitioned by record type."""
    if not records:
        log.warning("No records to write")
        return output_dir

    run_id = records[0].crawl_run_id
    safe_name = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in guild_name)
    base = output_dir / "discord" / f"{guild_id}--{safe_name}" / run_id
    base.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        record.validate()
        d = record.to_dict()
        grouped.setdefault(record.record_type, []).append(d)

    for record_type, payloads in grouped.items():
        path = base / f"{record_type}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload, sort_keys=True, default=str))
                f.write("\n")
        log.info("Wrote %d %s records → %s", len(payloads), record_type, path)

    # Write manifest
    manifest = {
        "platform": "discord",
        "guild_id": str(guild_id),
        "guild_name": guild_name,
        "crawl_run_id": run_id,
        "record_counts": dict(metrics.records_by_type),
        "metrics": metrics.to_dict(),
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("Manifest → %s", manifest_path)

    return base


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl a Discord guild into BronzeRecords (JSONL)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("DISCORD_BOT_TOKEN"),
        help="Bot token (or set DISCORD_BOT_TOKEN env var / use Doppler)",
    )
    parser.add_argument(
        "--guild", type=int, required=True,
        help="Guild (server) ID to crawl",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./crawl_output"),
        help="Output directory (default: ./crawl_output)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max messages per channel/thread (None = all)",
    )
    parser.add_argument(
        "--no-threads", action="store_true",
        help="Skip thread crawling",
    )
    parser.add_argument(
        "--include-bots", action="store_true",
        help="Include bot/webhook messages",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    if not args.token:
        log.error(
            "No bot token provided. Use --token or set DISCORD_BOT_TOKEN.\n"
            "  Doppler: doppler run --project api_keys --config dev -- python -m ...\n"
            "  Direct:  python -m ... --token YOUR_TOKEN"
        )
        sys.exit(1)

    connector = DiscordLiveConnector(
        token=args.token,
        message_limit=args.limit,
        crawl_threads=not args.no_threads,
        filter_bots=not args.include_bots,
    )

    log.info("Starting crawl of guild %d ...", args.guild)
    records, metrics = await connector.crawl_guild(args.guild)

    if not records:
        log.error("No records produced. Check errors: %s", metrics.errors)
        sys.exit(1)

    # Find guild name from community records
    guild_name = "unknown"
    for r in records:
        if isinstance(r, CommunityRecord) and r.community_type == "server":
            guild_name = r.community_name
            break

    output_path = write_records(records, metrics, args.output, guild_name, args.guild)

    # Print summary
    summary = metrics.to_dict()
    print("\n" + "=" * 60)
    print(f"CRAWL COMPLETE: {guild_name}")
    print("=" * 60)
    print(f"  Duration:     {summary['duration_seconds']:.1f}s")
    print(f"  Channels:     {summary['channels_crawled']}")
    print(f"  Threads:      {summary['threads_crawled']}")
    print(f"  Messages:     {summary['messages_fetched']}")
    print(f"  Bots filtered:{summary['bot_messages_filtered']}")
    print(f"  Unique users: {summary['unique_users']}")
    print(f"  Records:")
    for rtype, count in sorted(summary["records_by_type"].items()):
        print(f"    {rtype}: {count}")
    if summary["errors"]:
        print(f"  Errors ({len(summary['errors'])}):")
        for err in summary["errors"][:5]:
            print(f"    - {err}")
    print(f"\n  Output: {output_path}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
