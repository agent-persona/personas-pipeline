"""
Discord User API Connector

WARNING: Automating Discord account interactions using user tokens may violate Discord's
Terms of Service. User-token automation is:
- Not officially supported by Discord
- Potentially subject to account restrictions or bans
- More visible to rate limiting than bot tokens
- Slower due to human-like delays between requests

Use this connector only with explicit understanding of these risks and in compliance
with Discord's ToS and your jurisdiction's laws.

This connector implements the CommunityConnector interface for Discord using user
authentication (user token) instead of bot tokens. It provides access to servers
the authenticated user has joined and crawls messages from channels and threads.

Key features:
- User-token based authentication
- Human-like rate limiting with random delays
- Full support for channels and threads
- Server joining via invite codes
- Profile snapshots and interaction records
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..base import CommunityConnector, CrawlContext
from ..models import (
    AccountRecord,
    CommunityRecord,
    CrawlTarget,
    EvidencePointer,
    InteractionRecord,
    MessageRecord,
    ProfileSnapshotRecord,
    Record,
    ThreadRecord,
)

logger = logging.getLogger("discord_user_api")

# Discord channel type constants
TEXT_CHANNEL_TYPES = {0, 5}
THREAD_CHANNEL_TYPES = {10, 11, 12}
FORUM_CHANNEL_TYPES = {15, 16}


def _safe_name(value: str) -> str:
    """Convert a string to a safe slug format."""
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "discord"


def _stable_user_id(guild_id: str, username: str) -> str:
    """Generate a stable user ID based on guild and username."""
    return f"discord-user-{sha256(f'{guild_id}::{username}'.encode('utf-8')).hexdigest()[:16]}"


def _parse_dt(value: str | None) -> str | None:
    """Parse Discord ISO 8601 timestamp to standard format."""
    if not value:
        return None
    # Discord returns timestamps in ISO 8601 format with Z suffix
    # Return as-is for consistency with other connectors
    return value


@dataclass(slots=True)
class _DiscordScope:
    """Context for a Discord server being crawled."""

    guild_id: str
    guild_name: str
    invite_url: str


class DiscordUserApiClient:
    """HTTP client for Discord API using user token authentication.

    Uses human-like delays between requests and proper rate limit handling.
    """

    def __init__(
        self,
        token: str,
        min_delay: float = 1.5,
        max_delay: float = 3.0,
    ) -> None:
        """Initialize Discord user API client.

        Args:
            token: User authentication token (NOT bot token)
            min_delay: Minimum random delay between requests in seconds
            max_delay: Maximum random delay between requests in seconds
        """
        self.token = token
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.api_base = "https://discord.com/api/v10"
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.last_request_time = 0.0

    def get(
        self,
        path: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object] | list:
        """Make a GET request to the Discord API.

        Args:
            path: API endpoint path (e.g., '/channels/123')
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            HTTPError: On API errors (with retry logic for 429s)
            RuntimeError: On unrecoverable errors
        """
        # Apply human-like rate limiting
        elapsed = time.time() - self.last_request_time
        delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < delay:
            time.sleep(delay - elapsed)

        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.api_base}{path}{query}"

        headers = {
            "Authorization": self.token,  # NO "Bot" prefix for user tokens
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        request = Request(url, headers=headers)
        self.last_request_time = time.time()

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                with urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    return data
            except HTTPError as e:
                if e.code == 429:
                    # Rate limited - extract retry-after header
                    retry_after = float(e.headers.get("Retry-After", 1))
                    jitter = random.uniform(0, 0.5)
                    wait_time = retry_after + jitter
                    logger.warning(
                        f"Rate limited on {path}. Waiting {wait_time:.2f}s before retry"
                    )
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                elif e.code == 401:
                    raise RuntimeError("Invalid Discord user token") from e
                elif e.code == 403:
                    raise RuntimeError(
                        f"Access forbidden to {path} - user may not have permission"
                    ) from e
                elif e.code == 404:
                    logger.warning(f"Not found: {path}")
                    return {} if "guild" in path or "channel" in path else []
                else:
                    logger.error(f"HTTP {e.code} on {path}: {e.reason}")
                    raise
            except Exception as e:
                logger.error(f"Error requesting {path}: {e}")
                raise

        raise RuntimeError(f"Max retries exceeded for {path}")

    def join_server(self, invite_code: str) -> dict:
        """Join a server using an invite code.

        Args:
            invite_code: Discord invite code (e.g., 'abc123' from discord.gg/abc123)

        Returns:
            Server information dict
        """
        path = f"/invites/{invite_code}"
        logger.info(f"Joining server with invite: {invite_code}")
        return self.get(path)

    def list_guilds(self) -> list[dict]:
        """List all servers the user is a member of.

        Returns:
            List of guild objects
        """
        path = "/users/@me/guilds"
        result = self.get(path, {"limit": 200})
        return result if isinstance(result, list) else []

    def list_channels(self, guild_id: str) -> list[dict]:
        """List all channels in a guild.

        Args:
            guild_id: Discord guild/server ID

        Returns:
            List of channel objects
        """
        path = f"/guilds/{guild_id}/channels"
        result = self.get(path)
        return result if isinstance(result, list) else []


class DiscordUserApiConnector(CommunityConnector):
    """Crawls Discord servers using user token authentication.

    This connector accesses any public channels in servers the user has joined.
    It mirrors the bot connector interface but uses user authentication and
    includes human-like rate limiting.
    """

    platform = "discord"

    def __init__(self, client: DiscordUserApiClient) -> None:
        """Initialize connector with a Discord user API client.

        Args:
            client: Configured DiscordUserApiClient instance
        """
        self.client = client

    @classmethod
    def from_env(
        cls,
        token_env: str = "DISCORD_USER_TOKEN",
        min_delay: float = 1.5,
        max_delay: float = 3.0,
    ) -> DiscordUserApiConnector:
        """Create connector from environment variable.

        Args:
            token_env: Environment variable name containing user token
            min_delay: Minimum delay between requests
            max_delay: Maximum delay between requests

        Returns:
            Configured DiscordUserApiConnector

        Raises:
            RuntimeError: If token environment variable is not set
        """
        token = os.getenv(token_env)
        if not token:
            raise RuntimeError(f"Missing Discord user token in {token_env}")
        client = DiscordUserApiClient(token, min_delay=min_delay, max_delay=max_delay)
        return cls(client)

    @classmethod
    def from_token(
        cls,
        token: str,
        min_delay: float = 1.5,
        max_delay: float = 3.0,
    ) -> DiscordUserApiConnector:
        """Create connector from explicit token.

        Args:
            token: Discord user token
            min_delay: Minimum delay between requests
            max_delay: Maximum delay between requests

        Returns:
            Configured DiscordUserApiConnector
        """
        client = DiscordUserApiClient(token, min_delay=min_delay, max_delay=max_delay)
        return cls(client)

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        """Fetch records from Discord channels and threads.

        Args:
            target: Crawl target with guild_id and channel_ids in metadata
            context: Crawl context with timing and run ID
            since: Optional ISO 8601 timestamp to fetch messages after

        Returns:
            Iterable of Record objects

        Raises:
            ValueError: If target metadata is invalid
            RuntimeError: On API or authentication errors
        """
        self.validate_target(target)

        guild_id = str(target.metadata.get("guild_id") or target.target_id)
        channel_ids = [str(value) for value in target.metadata.get("channel_ids", [])]
        thread_ids = [str(value) for value in target.metadata.get("thread_ids", [])]
        crawl_threads = bool(target.metadata.get("crawl_threads", True))
        since_value = since or str(target.metadata.get("since") or "")
        until_value = str(target.metadata.get("until") or "")
        message_limit = int(target.metadata.get("message_limit") or 200)
        invite_url = str(
            target.metadata.get("invite_url")
            or target.url
            or f"https://discord.com/channels/{guild_id}"
        )

        if not channel_ids and not thread_ids:
            raise ValueError(
                "discord_user_api requires at least one channel_id or thread_id"
            )

        logger.info(f"Fetching Discord guild {guild_id} with {len(channel_ids)} channels")

        # Fetch guild info
        guild_path = f"/guilds/{guild_id}"
        guild = self.client.get(guild_path)
        if not isinstance(guild, dict):
            raise RuntimeError(f"Invalid guild response from {guild_path}")

        scope = _DiscordScope(
            guild_id=guild_id,
            guild_name=str(guild.get("name") or target.community_name or guild_id),
            invite_url=invite_url,
        )

        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id=scope.guild_id,
                community_name=scope.guild_name,
                community_type="server",
                parent_community_id=None,
                description=str(guild.get("description") or "") or None,
                member_count=guild.get("approximate_member_count"),
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=EvidencePointer(
                    source_url=scope.invite_url,
                    fetched_at=context.observed_at,
                ),
            )
        ]

        seen_users: set[str] = set()
        channel_names: dict[str, str] = {}

        # Process explicit channels
        for channel_id in channel_ids:
            try:
                channel = self._get_channel(channel_id)
                if not channel:
                    logger.warning(f"Could not fetch channel {channel_id}")
                    continue

                channel_names[channel_id] = str(channel.get("name") or channel_id)
                records.append(
                    CommunityRecord(
                        record_type="community",
                        platform="discord",
                        community_id=channel_id,
                        community_name=channel_names[channel_id],
                        community_type="channel",
                        parent_community_id=scope.guild_id,
                        description=str(channel.get("topic") or "") or None,
                        member_count=None,
                        rules_summary=None,
                        observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(
                            source_url=f"{scope.invite_url}#channel-{channel_id}",
                            fetched_at=context.observed_at,
                        ),
                    )
                )

                # Create synthetic thread for channel timeline
                synthetic_thread_id = f"discord-thread-{channel_id}-{context.crawl_run_id}"
                records.append(
                    ThreadRecord(
                        record_type="thread",
                        platform="discord",
                        thread_id=synthetic_thread_id,
                        community_id=channel_id,
                        title=f"{channel_names[channel_id]} timeline",
                        author_platform_user_id=None,
                        created_at=since_value or context.observed_at,
                        observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        metadata={"channel_id": channel_id, "thread_mode": "timeline"},
                        evidence_pointer=EvidencePointer(
                            source_url=f"{scope.invite_url}#channel-{channel_id}",
                            fetched_at=context.observed_at,
                        ),
                    )
                )

                # Fetch and process messages
                messages = self._iter_channel_messages(
                    channel_id, since_value, until_value, limit=message_limit
                )
                records.extend(
                    self._message_records(
                        messages=messages,
                        context=context,
                        scope=scope,
                        community_id=channel_id,
                        thread_id=synthetic_thread_id,
                        seen_users=seen_users,
                    )
                )
                logger.info(
                    f"Crawled channel {channel_id}: {len(messages)} messages"
                )

            except Exception as e:
                logger.error(f"Error crawling channel {channel_id}: {e}")
                continue

        # Fetch and crawl active threads if requested
        if crawl_threads and channel_ids:
            try:
                active_threads_path = f"/guilds/{guild_id}/threads/active"
                active_threads = self.client.get(active_threads_path)
                active_candidates = (
                    active_threads.get("threads", [])
                    if isinstance(active_threads, dict)
                    else []
                )
                for item in active_candidates:
                    if isinstance(item, dict) and str(item.get("parent_id") or "") in channel_ids:
                        thread_ids.append(str(item.get("id")))
                logger.info(f"Found {len(thread_ids)} active threads")
            except Exception as e:
                logger.warning(f"Could not fetch active threads: {e}")

        # Process explicit and discovered threads
        for thread_id in dict.fromkeys(thread_ids):
            try:
                thread = self._get_channel(thread_id)
                if not thread:
                    logger.warning(f"Could not fetch thread {thread_id}")
                    continue

                parent_id = str(
                    thread.get("parent_id")
                    or target.metadata.get("default_parent_channel_id")
                    or thread_id
                )
                channel_names.setdefault(parent_id, str(thread.get("name") or parent_id))

                # Ensure parent channel record exists
                if parent_id not in {
                    record.community_id
                    for record in records
                    if isinstance(record, CommunityRecord)
                }:
                    records.append(
                        CommunityRecord(
                            record_type="community",
                            platform="discord",
                            community_id=parent_id,
                            community_name=channel_names[parent_id],
                            community_type="channel",
                            parent_community_id=scope.guild_id,
                            description=None,
                            member_count=None,
                            rules_summary=None,
                            observed_at=context.observed_at,
                            crawl_run_id=context.crawl_run_id,
                            evidence_pointer=EvidencePointer(
                                source_url=f"{scope.invite_url}#channel-{parent_id}",
                                fetched_at=context.observed_at,
                            ),
                        )
                    )

                # Create thread record
                records.append(
                    ThreadRecord(
                        record_type="thread",
                        platform="discord",
                        thread_id=thread_id,
                        community_id=parent_id,
                        title=str(thread.get("name") or thread_id),
                        author_platform_user_id=str(thread.get("owner_id") or "") or None,
                        created_at=str(
                            thread.get("thread_metadata", {}).get("create_timestamp")
                            or context.observed_at
                        ),
                        observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        metadata={
                            "channel_id": parent_id,
                            "message_count": thread.get("message_count"),
                            "member_count": thread.get("member_count"),
                        },
                        evidence_pointer=EvidencePointer(
                            source_url=f"{scope.invite_url}#channel-{parent_id}/thread/{thread_id}",
                            fetched_at=context.observed_at,
                        ),
                    )
                )

                # Fetch and process thread messages
                messages = self._iter_channel_messages(
                    thread_id, since_value, until_value, limit=message_limit
                )
                records.extend(
                    self._message_records(
                        messages=messages,
                        context=context,
                        scope=scope,
                        community_id=parent_id,
                        thread_id=thread_id,
                        seen_users=seen_users,
                    )
                )
                logger.info(f"Crawled thread {thread_id}: {len(messages)} messages")

            except Exception as e:
                logger.error(f"Error crawling thread {thread_id}: {e}")
                continue

        logger.info(f"Crawl complete: {len(records)} total records")
        return records

    def _get_guild(self, guild_id: str) -> dict:
        """Fetch guild information.

        Args:
            guild_id: Discord guild ID

        Returns:
            Guild object or empty dict on error
        """
        try:
            result = self.client.get(f"/guilds/{guild_id}")
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Error fetching guild {guild_id}: {e}")
            return {}

    def _get_channel(self, channel_id: str) -> dict:
        """Fetch channel information.

        Args:
            channel_id: Discord channel ID

        Returns:
            Channel object or empty dict on error
        """
        try:
            result = self.client.get(f"/channels/{channel_id}")
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Error fetching channel {channel_id}: {e}")
            return {}

    def _iter_channel_messages(
        self,
        channel_id: str,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Iterate through channel/thread messages with pagination.

        Fetches messages in reverse chronological order (newest first) using
        the 'before' cursor, then sorts them chronologically.

        Args:
            channel_id: Discord channel or thread ID
            since: Optional ISO 8601 timestamp (include messages after this)
            until: Optional ISO 8601 timestamp (include messages before this)
            limit: Maximum messages to fetch

        Returns:
            List of message objects sorted chronologically
        """
        messages: list[dict] = []
        before_id: str | None = None
        fetched = 0

        while fetched < limit:
            params: dict[str, object] = {"limit": min(100, limit - fetched)}
            if before_id:
                params["before"] = before_id

            try:
                payload = self.client.get(f"/channels/{channel_id}/messages", params)
            except Exception as e:
                logger.error(f"Error fetching messages from {channel_id}: {e}")
                break

            if not isinstance(payload, list) or not payload:
                break

            stop = False
            for item in payload:
                if not isinstance(item, dict):
                    continue

                timestamp = str(item.get("timestamp") or "")
                if until and timestamp > until:
                    continue
                if since and timestamp < since:
                    stop = True
                    continue

                messages.append(item)
                fetched += 1

            if stop or not payload:
                break

            before_id = str(payload[-1].get("id") or "")
            if not before_id:
                break

        messages.sort(key=lambda item: str(item.get("timestamp") or ""))
        return messages

    def _message_records(
        self,
        *,
        messages: list[dict],
        context: CrawlContext,
        scope: _DiscordScope,
        community_id: str,
        thread_id: str,
        seen_users: set[str],
    ) -> list[Record]:
        """Convert Discord messages to record objects.

        Args:
            messages: List of Discord message objects
            context: Crawl context
            scope: Discord server context
            community_id: Channel ID for these messages
            thread_id: Thread ID context
            seen_users: Set of already-seen user IDs (mutated)

        Returns:
            List of Record objects (AccountRecord, ProfileSnapshotRecord, MessageRecord, InteractionRecord)
        """
        # Build lookup tables for message authors and references
        author_lookup: dict[str, str] = {}
        message_lookup: dict[str, str] = {}

        for message in messages:
            author = message.get("author", {})
            if not isinstance(author, dict):
                continue

            author_id = str(author.get("id") or "")
            if author_id:
                author_lookup[author_id] = author_id

            message_id = str(message.get("id") or "")
            if message_id and author_id:
                message_lookup[message_id] = author_id

        records: list[Record] = []

        for message in messages:
            author = message.get("author", {})
            if not isinstance(author, dict):
                continue

            # Resolve author ID
            author_id = (
                str(author.get("id") or "")
                or _stable_user_id(scope.guild_id, str(author.get("username") or "unknown"))
            )
            username = str(author.get("username") or author.get("global_name") or author_id)

            # Create account and profile records for new users
            if author_id not in seen_users:
                user_pointer = EvidencePointer(
                    source_url=f"{scope.invite_url}#user-{author_id}",
                    fetched_at=context.observed_at,
                )
                records.append(
                    AccountRecord(
                        record_type="account",
                        platform="discord",
                        platform_user_id=author_id,
                        username=username,
                        account_created_at=None,
                        first_observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=user_pointer,
                    )
                )
                records.append(
                    ProfileSnapshotRecord(
                        record_type="profile_snapshot",
                        platform="discord",
                        platform_user_id=author_id,
                        snapshot_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        fields={
                            "display_name": str(author.get("global_name") or username)
                        },
                        evidence_pointer=user_pointer,
                    )
                )
                seen_users.add(author_id)

            # Create message record
            message_id = str(message.get("id") or "")
            message_url = f"{scope.invite_url}#channel-{community_id}/thread/{thread_id}/message/{message_id}"
            created_at = str(message.get("timestamp") or context.observed_at)

            records.append(
                MessageRecord(
                    record_type="message",
                    platform="discord",
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=author_id,
                    body=str(message.get("content") or ""),
                    created_at=created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=None,
                    metadata={
                        "attachments": len(message.get("attachments") or []),
                        "reactions": len(message.get("reactions") or []),
                    },
                    evidence_pointer=EvidencePointer(
                        source_url=message_url,
                        fetched_at=context.observed_at,
                    ),
                )
            )

            # Handle message replies
            reference = message.get("message_reference", {})
            referenced_message = message.get("referenced_message", {})
            reply_to_message_id = None
            reply_to_user_id = None

            if isinstance(reference, dict):
                reply_to_message_id = str(reference.get("message_id") or "") or None

            if isinstance(referenced_message, dict):
                referenced_author = referenced_message.get("author", {})
                if isinstance(referenced_author, dict):
                    reply_to_user_id = str(referenced_author.get("id") or "") or None

            if not reply_to_user_id and reply_to_message_id:
                reply_to_user_id = message_lookup.get(reply_to_message_id)

            # Create interaction record if this is a reply
            if reply_to_user_id:
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform="discord",
                        interaction_type="reply",
                        source_user_id=author_id,
                        target_user_id=reply_to_user_id,
                        message_id=message_id,
                        thread_id=thread_id,
                        community_id=community_id,
                        created_at=created_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(
                            source_url=message_url,
                            fetched_at=context.observed_at,
                            derived_from_message_id=message_id,
                        ),
                    )
                )

        return records
