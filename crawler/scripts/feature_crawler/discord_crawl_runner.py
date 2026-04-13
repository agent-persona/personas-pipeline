#!/usr/bin/env python3
"""Automated Discord crawl runner.

Reads DISCORD_USER_TOKEN from environment or .env file,
crawls specified servers/channels, saves to feature_crawler/data.

Usage:
    # Single channel
    python3 scripts/discord_crawl_runner.py \
        --guild 662267976984297473 \
        --channel 999550150705954856

    # Multiple channels
    python3 scripts/discord_crawl_runner.py \
        --guild 662267976984297473 \
        --channel 999550150705954856 \
        --channel 952771221915840552

    # With token directly
    DISCORD_USER_TOKEN=xxx python3 scripts/discord_crawl_runner.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
log = logging.getLogger("discord_crawl_runner")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_token() -> str:
    """Load Discord user token from env or .env file."""
    token = os.environ.get("DISCORD_USER_TOKEN")
    if token:
        return token.strip()

    # Try .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DISCORD_USER_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError(
        "No DISCORD_USER_TOKEN found.\n"
        "Set it via:\n"
        "  export DISCORD_USER_TOKEN='your_token_here'\n"
        "  or add DISCORD_USER_TOKEN=xxx to .env"
    )


def crawl_channel(
    token: str,
    guild_id: str,
    channel_id: str,
    output_dir: Path,
    community_name: str,
    message_limit: int = 200,
    min_delay: float = 2.0,
    max_delay: float = 4.0,
    since: str | None = None,
) -> dict:
    """Crawl a single channel and write results. Returns summary dict."""
    from crawler.feature_crawler.connectors.discord_user_api import (
        DiscordUserApiClient,
        DiscordUserApiConnector,
    )
    from crawler.feature_crawler.base import CrawlContext
    from crawler.feature_crawler.models import CrawlTarget
    from crawler.feature_crawler.sink import JsonlSink

    client = DiscordUserApiClient(
        token=token,
        min_delay=min_delay,
        max_delay=max_delay,
    )
    connector = DiscordUserApiConnector(client=client)

    target = CrawlTarget(
        platform="discord",
        target_id=guild_id,
        url=f"https://discord.com/channels/{guild_id}/{channel_id}",
        community_name=community_name,
        collection_basis="public-permitted",
        allow_persona_inference=True,
        metadata={
            "guild_id": guild_id,
            "channel_ids": [channel_id],
            "thread_ids": [],
            "until": None,
            "message_limit": message_limit,
        },
    )

    context = CrawlContext.create()
    log.info("Crawling guild=%s channel=%s (limit=%d)", guild_id, channel_id, message_limit)

    records = list(connector.fetch(target=target, context=context, since=since))

    if not records:
        log.warning("No records from channel %s", channel_id)
        return {"channel_id": channel_id, "records": 0, "path": None}

    sink = JsonlSink(output_dir)
    output_path = sink.write(records)

    # Count by type
    type_counts = {}
    for r in records:
        rt = getattr(r, "record_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    summary = {
        "channel_id": channel_id,
        "records": len(records),
        "path": str(output_path),
        "types": type_counts,
        "crawl_run_id": context.crawl_run_id,
    }
    log.info("  → %d records written to %s", len(records), output_path)
    for rt, count in sorted(type_counts.items()):
        log.info("    %s: %d", rt, count)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Automated Discord crawl runner")
    parser.add_argument("--guild", required=True, help="Guild (server) ID")
    parser.add_argument("--channel", action="append", required=True, help="Channel ID(s) to crawl")
    parser.add_argument("--community-name", default=None, help="Community name for output")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: feature_crawler/data)")
    parser.add_argument("--message-limit", type=int, default=200)
    parser.add_argument("--min-delay", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=4.0)
    parser.add_argument("--since", default=None, help="ISO date to crawl from")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        token = load_token()
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("Token loaded (length=%d)", len(token))

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "feature_crawler" / "data"
    community_name = args.community_name or f"discord-{args.guild}"

    results = []
    for channel_id in args.channel:
        try:
            summary = crawl_channel(
                token=token,
                guild_id=args.guild,
                channel_id=channel_id,
                output_dir=output_dir,
                community_name=community_name,
                message_limit=args.message_limit,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                since=args.since,
            )
            results.append(summary)
        except Exception as e:
            log.error("Failed to crawl channel %s: %s", channel_id, e, exc_info=True)
            results.append({"channel_id": channel_id, "records": 0, "error": str(e)})

        # Serial: wait between channels to avoid sink collision
        if channel_id != args.channel[-1]:
            time.sleep(2)

    # Write run manifest
    manifest = {
        "guild_id": args.guild,
        "community_name": community_name,
        "ran_at": datetime.now(UTC).isoformat(),
        "channels": results,
        "total_records": sum(r.get("records", 0) for r in results),
    }
    manifest_path = output_dir / f"run_manifest_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"CRAWL COMPLETE: {community_name}")
    print(f"{'='*60}")
    print(f"  Channels: {len(results)}")
    print(f"  Total records: {manifest['total_records']}")
    for r in results:
        status = f"{r['records']} records" if r.get('records') else f"FAILED: {r.get('error', 'unknown')}"
        print(f"  #{r['channel_id']}: {status}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
