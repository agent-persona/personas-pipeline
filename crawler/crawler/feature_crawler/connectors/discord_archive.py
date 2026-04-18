"""
Discord Archive Connector

This connector fetches Discord messages from third-party archive/index services that
publicly index Discord server content. It is a best-effort fallback mechanism that
attempts to retrieve publicly archived Discord data from known archive services.

The connector does not interact with Discord's official API directly, but rather
accesses third-party archives that have independently indexed public Discord content.
This approach respects Discord's Terms of Service by accessing publicly available
archival data.

Archive services attempted:
- discordhistory.org: HTML archive of Discord messages
- top.gg: Server listing with cached content

If no archive data is found, the connector returns an empty list with a warning log.
"""

from __future__ import annotations

import html
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from ..base import CommunityConnector, CrawlContext
from ..models import (
    AccountRecord,
    CommunityRecord,
    CrawlTarget,
    EvidencePointer,
    MessageRecord,
    ProfileSnapshotRecord,
    Record,
    ThreadRecord,
)

logger = logging.getLogger("discord_archive")

DEFAULT_SOURCES = [
    {
        "name": "discordhistory",
        "base_url": "https://discordhistory.org",
        "type": "html_archive",
    },
    {
        "name": "top_gg",
        "base_url": "https://top.gg",
        "type": "server_listing",
    },
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class DiscordArchiveConnector(CommunityConnector):
    """
    Connector for fetching Discord messages from third-party archive services.

    This connector attempts to retrieve publicly archived Discord content from
    known third-party archive services. It is designed as a best-effort fallback
    when direct Discord API access is unavailable.
    """

    platform = "discord"

    def __init__(
        self,
        sources: list[dict[str, str]] | None = None,
        request_delay: float = 2.0,
    ) -> None:
        """
        Initialize the Discord archive connector.

        Args:
            sources: List of archive source configurations. Defaults to known public sources.
                    Each source should have: name, base_url, type
            request_delay: Seconds to wait between HTTP requests to respect rate limits
        """
        self.sources = sources or DEFAULT_SOURCES
        self.request_delay = request_delay
        self._last_request_time = 0.0

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> list[Record]:
        """
        Fetch archived Discord messages for a target server/channel.

        Args:
            target: The crawl target containing platform, target_id, and metadata
            context: The crawl context with run_id and observed_at timestamp
            since: Optional ISO timestamp to filter messages

        Returns:
            List of Record objects (community, threads, messages, and accounts)
        """
        guild_id = target.metadata.get("guild_id")
        channel_id = target.metadata.get("channel_id")

        if not guild_id or not channel_id:
            logger.warning(
                "Missing guild_id or channel_id in target metadata. "
                f"Target: {target.target_id}"
            )
            return []

        logger.info(
            f"Attempting to fetch archived Discord messages for guild={guild_id}, "
            f"channel={channel_id}"
        )

        all_records: list[Record] = []

        # Try each archive source
        for source in self.sources:
            try:
                records = self._fetch_from_source(
                    source, guild_id, channel_id, since, target, context
                )
                if records:
                    logger.info(
                        f"Successfully fetched {len(records)} records from "
                        f"{source['name']} for guild={guild_id}, channel={channel_id}"
                    )
                    all_records.extend(records)
                    break  # Use first successful source
            except Exception as e:
                logger.warning(
                    f"Failed to fetch from {source['name']}: {e}. Trying next source."
                )
                continue

        # If no results from archives, attempt web search fallback
        if not all_records:
            try:
                records = self._search_web_for_discord_content(
                    guild_id, channel_id, target, context
                )
                if records:
                    logger.info(
                        f"Found {len(records)} records via web search fallback"
                    )
                    all_records.extend(records)
            except Exception as e:
                logger.warning(f"Web search fallback failed: {e}")

        if not all_records:
            logger.warning(
                f"No archived content found for guild={guild_id}, channel={channel_id}"
            )

        return all_records

    def _fetch_from_source(
        self,
        source: dict[str, str],
        guild_id: str,
        channel_id: str,
        since: str | None,
        target: CrawlTarget,
        context: CrawlContext,
    ) -> list[Record]:
        """
        Fetch from a specific archive source based on its type.

        Args:
            source: Archive source configuration
            guild_id: Discord guild/server ID
            channel_id: Discord channel ID
            since: Optional timestamp filter
            target: The crawl target
            context: The crawl context

        Returns:
            List of records fetched from this source
        """
        source_type = source.get("type", "unknown")

        if source_type == "html_archive":
            return self._fetch_from_html_archive(
                source, guild_id, channel_id, since, target, context
            )
        elif source_type == "server_listing":
            return self._fetch_from_api_archive(
                source, guild_id, channel_id, since, target, context
            )
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return []

    def _fetch_from_html_archive(
        self,
        source: dict[str, str],
        guild_id: str,
        channel_id: str,
        since: str | None,
        target: CrawlTarget,
        context: CrawlContext,
    ) -> list[Record]:
        """
        Fetch from an HTML-based archive service.

        Args:
            source: Archive source configuration
            guild_id: Discord guild/server ID
            channel_id: Discord channel ID
            since: Optional timestamp filter
            target: The crawl target
            context: The crawl context

        Returns:
            List of records
        """
        base_url = source.get("base_url", "")

        # Construct URL for the channel archive
        archive_url = f"{base_url}/archive/{guild_id}/{channel_id}"

        html_content = self._http_get(archive_url)
        if not html_content:
            logger.debug(f"No HTML content retrieved from {archive_url}")
            return []

        # Parse messages from HTML
        messages = self._parse_archive_page(html_content, guild_id, channel_id)
        logger.debug(f"Parsed {len(messages)} messages from {source['name']}")

        # Convert to records
        return self._messages_to_records(
            messages, target, context, archive_url, guild_id, channel_id, since
        )

    def _fetch_from_api_archive(
        self,
        source: dict[str, str],
        guild_id: str,
        channel_id: str,
        since: str | None,
        target: CrawlTarget,
        context: CrawlContext,
    ) -> list[Record]:
        """
        Fetch from an API-based archive service.

        Args:
            source: Archive source configuration
            guild_id: Discord guild/server ID
            channel_id: Discord channel ID
            since: Optional timestamp filter
            target: The crawl target
            context: The crawl context

        Returns:
            List of records
        """
        base_url = source.get("base_url", "")

        # For top.gg and similar, try to access server info endpoint
        # This is speculative as actual API varies
        api_url = f"{base_url}/api/guild/{guild_id}"

        json_content = self._http_get(api_url)
        if not json_content:
            logger.debug(f"No API content retrieved from {api_url}")
            return []

        # Attempt to parse JSON (very basic - actual implementation depends on API)
        try:
            import json

            data = json.loads(json_content)

            # Extract messages if available (structure varies by service)
            messages = []
            if isinstance(data, dict):
                # Try common message field names
                for field in ["messages", "posts", "content", "history"]:
                    if field in data and isinstance(data[field], list):
                        messages = data[field]
                        break

            logger.debug(f"Parsed {len(messages)} messages from {source['name']}")

            return self._messages_to_records(
                messages, target, context, api_url, guild_id, channel_id, since
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse JSON from {source['name']}: {e}")
            return []

    def _search_web_for_discord_content(
        self,
        guild_id: str,
        channel_id: str,
        target: CrawlTarget,
        context: CrawlContext,
    ) -> list[Record]:
        """
        Search for cached/indexed Discord content via web fallback.

        This is a last-resort method that attempts to find Discord content
        through general web search and caching services.

        Args:
            guild_id: Discord guild/server ID
            channel_id: Discord channel ID
            target: The crawl target
            context: The crawl context

        Returns:
            List of records found via web search
        """
        logger.debug(
            f"Attempting web search fallback for guild={guild_id}, "
            f"channel={channel_id}"
        )

        # Search for cached content mentioning this guild and channel
        search_query = f"discord.com/channels/{guild_id}/{channel_id} cached"

        # This is a placeholder - actual web search would use external service
        # For now, we log and return empty to avoid making live requests
        logger.debug(f"Web search would use query: {search_query}")

        return []

    def _parse_archive_page(
        self,
        html: str,
        guild_id: str,
        channel_id: str,
    ) -> list[dict[str, Any]]:
        """
        Parse messages from archive HTML page.

        Attempts to extract message data from common archive HTML patterns.

        Args:
            html: HTML content from archive
            guild_id: Guild ID for context
            channel_id: Channel ID for context

        Returns:
            List of message dictionaries with keys:
            - message_id
            - author_id
            - username
            - content
            - created_at
        """
        messages = []

        # Common patterns in Discord archive HTML
        # Pattern 1: Message containers with data attributes
        message_pattern = re.compile(
            r'<div[^>]*class="[^"]*message[^"]*"[^>]*data-message-id="([^"]*)"[^>]*>.*?'
            r'<span[^>]*class="[^"]*author[^"]*">([^<]*)</span>.*?'
            r'<span[^>]*class="[^"]*timestamp[^"]*">([^<]*)</span>.*?'
            r'<div[^>]*class="[^"]*content[^"]*">([^<]*)</div>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in message_pattern.finditer(html):
            try:
                message_id = match.group(1).strip()
                author = match.group(2).strip()
                timestamp = match.group(3).strip()
                content = html.unescape(match.group(4).strip())

                if not message_id or not author or not content:
                    continue

                # Generate author ID from username (since archives don't have real IDs)
                author_id = f"archived_{hash(author) % (10**8)}"

                messages.append(
                    {
                        "message_id": message_id,
                        "author_id": author_id,
                        "username": author,
                        "content": content,
                        "created_at": timestamp,
                    }
                )
            except (IndexError, AttributeError) as e:
                logger.debug(f"Failed to parse message from regex: {e}")
                continue

        # Pattern 2: Alternative JSON-LD markup
        json_ld_pattern = re.compile(
            r'<script type="application/ld\+json">(.*?)</script>',
            re.DOTALL,
        )

        for match in json_ld_pattern.finditer(html):
            try:
                import json

                data = json.loads(match.group(1))
                if isinstance(data, dict) and data.get("@type") == "Message":
                    messages.append(
                        {
                            "message_id": data.get("identifier"),
                            "author_id": data.get("author", {}).get("identifier"),
                            "username": data.get("author", {}).get("name"),
                            "content": data.get("text"),
                            "created_at": data.get("datePublished"),
                        }
                    )
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.debug(f"Failed to parse JSON-LD: {e}")
                continue

        logger.debug(f"Parsed {len(messages)} messages from HTML")
        return messages

    def _messages_to_records(
        self,
        messages: list[dict[str, Any]],
        target: CrawlTarget,
        context: CrawlContext,
        source_url: str,
        guild_id: str,
        channel_id: str,
        since: str | None = None,
    ) -> list[Record]:
        """
        Convert parsed messages to standardized Record objects.

        Args:
            messages: List of message dictionaries
            target: The crawl target
            context: The crawl context
            source_url: URL where messages were retrieved from
            guild_id: Guild/server ID
            channel_id: Channel ID
            since: Optional timestamp filter

        Returns:
            List of Record objects ready for sink
        """
        records: list[Record] = []
        seen_users: set[str] = set()

        # Add community records for server and channel
        records.append(
            CommunityRecord(
                record_type="community",
                platform=self.platform,
                community_id=guild_id,
                community_name=target.community_name,
                community_type="server",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=EvidencePointer(
                    source_url=source_url,
                    fetched_at=context.observed_at,
                ),
            )
        )

        records.append(
            CommunityRecord(
                record_type="community",
                platform=self.platform,
                community_id=channel_id,
                community_name=f"#{target.community_name}",
                community_type="channel",
                parent_community_id=guild_id,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=EvidencePointer(
                    source_url=source_url,
                    fetched_at=context.observed_at,
                ),
            )
        )

        # Create a thread record representing the archive snapshot
        thread_id = f"archive_{channel_id}"
        records.append(
            ThreadRecord(
                record_type="thread",
                platform=self.platform,
                thread_id=thread_id,
                community_id=channel_id,
                title=f"Archived messages for {target.community_name}",
                author_platform_user_id=None,
                created_at=context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "source_type": "archive",
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                },
                evidence_pointer=EvidencePointer(
                    source_url=source_url,
                    fetched_at=context.observed_at,
                ),
            )
        )

        # Add message records and author records
        for message in messages:
            author_id = message.get("author_id")
            username = message.get("username", "Unknown")
            message_id = message.get("message_id", "")
            content = message.get("content", "")
            created_at = message.get("created_at", context.observed_at)

            # Skip messages without required fields
            if not message_id or not author_id:
                logger.debug(f"Skipping message with missing ID fields")
                continue

            # Filter by since timestamp if provided
            if since and created_at < since:
                continue

            # Add author records (once per user)
            if author_id not in seen_users:
                author_url = f"{source_url}#user-{author_id}"

                records.append(
                    AccountRecord(
                        record_type="account",
                        platform=self.platform,
                        platform_user_id=author_id,
                        username=username,
                        account_created_at=None,
                        first_observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(
                            source_url=author_url,
                            fetched_at=context.observed_at,
                        ),
                    )
                )

                records.append(
                    ProfileSnapshotRecord(
                        record_type="profile_snapshot",
                        platform=self.platform,
                        platform_user_id=author_id,
                        snapshot_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        fields={"display_name": username},
                        evidence_pointer=EvidencePointer(
                            source_url=author_url,
                            fetched_at=context.observed_at,
                        ),
                    )
                )

                seen_users.add(author_id)

            # Add message record
            message_url = f"{source_url}#message-{message_id}"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform=self.platform,
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=channel_id,
                    author_platform_user_id=author_id,
                    body=content,
                    created_at=created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=None,
                    metadata={
                        "source_type": "archive",
                        "guild_id": guild_id,
                        "channel_id": channel_id,
                    },
                    evidence_pointer=EvidencePointer(
                        source_url=message_url,
                        fetched_at=context.observed_at,
                        derived_from_message_id=message_id,
                    ),
                )
            )

        logger.info(f"Generated {len(records)} records from {len(messages)} messages")
        return records

    def _http_get(self, url: str) -> str | None:
        """
        Perform HTTP GET request with user-agent and error handling.

        Args:
            url: URL to fetch

        Returns:
            HTML/text content or None if request failed
        """
        # Respect request delay
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                self._last_request_time = time.time()
                content = response.read().decode("utf-8", errors="replace")
                logger.debug(f"Successfully fetched {len(content)} bytes from {url}")
                return content

        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                logger.debug(f"HTTP {e.code} from {url} - content not available")
            else:
                logger.warning(f"HTTP {e.code} error fetching {url}")
            return None

        except urllib.error.URLError as e:
            logger.warning(f"URL error fetching {url}: {e.reason}")
            return None

        except TimeoutError as e:
            logger.warning(f"Timeout fetching {url}: {e}")
            return None

        except Exception as e:
            logger.warning(f"Unexpected error fetching {url}: {e}")
            return None
