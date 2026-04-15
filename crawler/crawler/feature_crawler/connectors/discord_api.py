from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable
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


def _safe_name(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "discord"


def _stable_user_id(guild_id: str, username: str) -> str:
    return f"discord-user-{sha256(f'{guild_id}::{username}'.encode('utf-8')).hexdigest()[:16]}"


@dataclass(slots=True)
class _DiscordScope:
    guild_id: str
    guild_name: str
    invite_url: str


class DiscordApiConnector(CommunityConnector):
    platform = "discord"
    api_base = "https://discord.com/api/v10"
    user_agent = "AgentPersonaCrawler/0.1"

    def __init__(self, *, bot_token: str | None = None, bot_token_env: str = "DISCORD_BOT_TOKEN") -> None:
        self.bot_token = bot_token
        self.bot_token_env = bot_token_env

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        self.validate_target(target)
        token = self.bot_token or os.getenv(self.bot_token_env)
        if not token:
            raise RuntimeError(f"missing Discord bot token in {self.bot_token_env}")

        guild_id = str(target.metadata.get("guild_id") or target.target_id)
        channel_ids = [str(value) for value in target.metadata.get("channel_ids", [])]
        thread_ids = [str(value) for value in target.metadata.get("thread_ids", [])]
        crawl_threads = bool(target.metadata.get("crawl_threads", True))
        since_value = since or str(target.metadata.get("since") or "")
        until_value = str(target.metadata.get("until") or "")
        invite_url = str(target.metadata.get("invite_url") or target.url or f"https://discord.com/channels/{guild_id}")

        if not channel_ids and not thread_ids:
            raise ValueError("crawl-discord requires at least one channel_id or thread_id")

        guild = self._request_json(token, f"/guilds/{guild_id}")
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
                evidence_pointer=EvidencePointer(source_url=scope.invite_url, fetched_at=context.observed_at),
            )
        ]

        seen_users: set[str] = set()
        channel_names: dict[str, str] = {}
        for channel_id in channel_ids:
            channel = self._request_json(token, f"/channels/{channel_id}")
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
            messages = self._fetch_channel_messages(token, channel_id, since_value, until_value)
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

        if crawl_threads and channel_ids:
            active_threads = self._request_json(token, f"/guilds/{guild_id}/threads/active")
            active_candidates = active_threads.get("threads", []) if isinstance(active_threads, dict) else []
            for item in active_candidates:
                if isinstance(item, dict) and str(item.get("parent_id") or "") in channel_ids:
                    thread_ids.append(str(item.get("id")))

        for thread_id in dict.fromkeys(thread_ids):
            thread = self._request_json(token, f"/channels/{thread_id}")
            parent_id = str(thread.get("parent_id") or target.metadata.get("default_parent_channel_id") or thread_id)
            channel_names.setdefault(parent_id, str(thread.get("name") or parent_id))
            if parent_id not in {record.community_id for record in records if isinstance(record, CommunityRecord)}:
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

            records.append(
                ThreadRecord(
                    record_type="thread",
                    platform="discord",
                    thread_id=thread_id,
                    community_id=parent_id,
                    title=str(thread.get("name") or thread_id),
                    author_platform_user_id=str(thread.get("owner_id") or "") or None,
                    created_at=str(thread.get("thread_metadata", {}).get("create_timestamp") or context.observed_at),
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
            messages = self._fetch_channel_messages(token, thread_id, since_value, until_value)
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

        return records

    def _message_records(
        self,
        *,
        messages: list[dict[str, object]],
        context: CrawlContext,
        scope: _DiscordScope,
        community_id: str,
        thread_id: str,
        seen_users: set[str],
    ) -> list[Record]:
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
            author_id = str(author.get("id") or "") or _stable_user_id(scope.guild_id, str(author.get("username") or "unknown"))
            username = str(author.get("username") or author.get("global_name") or author_id)
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
                        fields={"display_name": str(author.get("global_name") or username)},
                        evidence_pointer=user_pointer,
                    )
                )
                seen_users.add(author_id)

            message_id = str(message.get("id") or "")
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
                    reply_to_message_id=reply_to_message_id,
                    reply_to_user_id=reply_to_user_id,
                    metadata={
                        "attachments": len(message.get("attachments") or []),
                        "reactions": len(message.get("reactions") or []),
                    },
                    evidence_pointer=EvidencePointer(source_url=message_url, fetched_at=context.observed_at),
                )
            )
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

    def _fetch_channel_messages(
        self,
        token: str,
        channel_id: str,
        since: str | None,
        until: str | None,
    ) -> list[dict[str, object]]:
        before_id: str | None = None
        messages: list[dict[str, object]] = []
        while True:
            params = {"limit": 100}
            if before_id:
                params["before"] = before_id
            payload = self._request_json(token, f"/channels/{channel_id}/messages", params=params)
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

            before_id = str(payload[-1].get("id") or "")
            if stop or not before_id:
                break

        messages.sort(key=lambda item: str(item.get("timestamp") or ""))
        return messages

    def _request_json(
        self,
        token: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> dict[str, object] | list[dict[str, object]]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.api_base}{path}{query}",
            headers={
                "Authorization": f"Bot {token}",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
