from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable
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


TEXT_CHANNEL_TYPES = {0, 5}
THREAD_CHANNEL_TYPES = {10, 11, 12}
FORUM_CHANNEL_TYPES = {15, 16}


@dataclass(slots=True)
class DiscordTargetSpec:
    guild_id: str
    guild_name: str | None
    invite_url: str | None
    channel_ids: list[str]
    thread_ids: list[str]
    since: str | None
    until: str | None
    message_limit: int | None


@dataclass(slots=True)
class DiscordApiClient:
    token: str
    fetch_json: Any | None = None

    api_base: str = "https://discord.com/api/v10"

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        full_path = f"{path}{query}"
        if self.fetch_json is not None:
            return self.fetch_json(full_path)
        request = Request(
            f"{self.api_base}{full_path}",
            method="GET",
            headers={
                "Authorization": f"Bot {self.token}",
                "User-Agent": "AgentPersonaCrawler/0.2",
            },
        )
        while True:
            try:
                with urlopen(request, timeout=20) as response:
                    reset_after = response.headers.get("X-RateLimit-Reset-After")
                    remaining = response.headers.get("X-RateLimit-Remaining")
                    payload = json.loads(response.read().decode("utf-8"))
                    if remaining == "0" and reset_after:
                        time.sleep(float(reset_after))
                    return payload
            except HTTPError as exc:
                if exc.code == 429:
                    retry_after = json.loads(exc.read().decode("utf-8")).get("retry_after", 1)
                    time.sleep(float(retry_after))
                    continue
                raise


class DiscordApiConnector(CommunityConnector):
    platform = "discord"

    def __init__(self, client: DiscordApiClient) -> None:
        self.client = client

    @classmethod
    def from_env(cls, token_env: str = "DISCORD_BOT_TOKEN") -> "DiscordApiConnector":
        token = os.environ.get(token_env)
        if not token:
            raise ValueError(f"missing Discord bot token in ${token_env}")
        return cls(client=DiscordApiClient(token=token))

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> list[Record]:
        self.validate_target(target)
        spec = self._spec_from_target(target, since)
        guild = self._get_guild(spec.guild_id)
        if spec.channel_ids or spec.thread_ids:
            channels = {
                channel_id: self._get_channel(channel_id)
                for channel_id in [*spec.channel_ids, *spec.thread_ids]
            }
        else:
            channels = {channel["id"]: channel for channel in self._get_channels(spec.guild_id)}
        selected_channel_ids = spec.channel_ids or self._default_channel_ids(channels.values())
        selected_thread_ids = list(spec.thread_ids)

        records: list[Record] = []
        invite_url = spec.invite_url or f"https://discord.com/channels/{spec.guild_id}"
        guild_name = spec.guild_name or guild.get("name") or spec.guild_id
        pointer = EvidencePointer(source_url=invite_url, fetched_at=context.observed_at)
        records.append(
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id=spec.guild_id,
                community_name=guild_name,
                community_type="server",
                parent_community_id=None,
                description=guild.get("description"),
                member_count=guild.get("approximate_member_count"),
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            )
        )

        for channel_id in selected_channel_ids:
            channel = channels.get(channel_id)
            if not channel:
                continue
            records.extend(self._crawl_channel(spec=spec, channel=channel, channels=channels, context=context))
            if int(channel.get("type", -1)) in FORUM_CHANNEL_TYPES:
                selected_thread_ids.extend(
                    thread["id"]
                    for thread in self._iter_forum_threads(spec.guild_id, channel_id)
                    if thread.get("id")
                )

        for thread_id in selected_thread_ids:
            thread = channels.get(thread_id) or self._get_channel(thread_id)
            if not thread:
                continue
            records.extend(self._crawl_thread(spec=spec, thread=thread, channels=channels, context=context))

        return records

    def _spec_from_target(self, target: CrawlTarget, since: str | None) -> DiscordTargetSpec:
        metadata = target.metadata
        guild_id = str(metadata.get("guild_id") or target.target_id)
        return DiscordTargetSpec(
            guild_id=guild_id,
            guild_name=metadata.get("guild_name"),
            invite_url=metadata.get("invite_url"),
            channel_ids=[str(value) for value in metadata.get("channel_ids", [])],
            thread_ids=[str(value) for value in metadata.get("thread_ids", [])],
            since=since or metadata.get("since"),
            until=metadata.get("until") or metadata.get("before"),
            message_limit=int(metadata["message_limit"]) if metadata.get("message_limit") else None,
        )

    def _crawl_channel(
        self,
        *,
        spec: DiscordTargetSpec,
        channel: dict[str, Any],
        channels: dict[str, dict[str, Any]],
        context: CrawlContext,
    ) -> list[Record]:
        channel_id = str(channel["id"])
        channel_name = channel.get("name") or channel_id
        channel_url = f"https://discord.com/channels/{spec.guild_id}/{channel_id}"
        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id=channel_id,
                community_name=channel_name,
                community_type="forum" if int(channel.get("type", -1)) in FORUM_CHANNEL_TYPES else "channel",
                parent_community_id=spec.guild_id,
                description=channel.get("topic"),
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=EvidencePointer(source_url=channel_url, fetched_at=context.observed_at),
            )
        ]

        if int(channel.get("type", -1)) not in TEXT_CHANNEL_TYPES:
            return records

        thread_id = f"discord-channel-window-{channel_id}-{(spec.since or 'all').replace(':', '').replace('-', '')}"
        records.append(
            ThreadRecord(
                record_type="thread",
                platform="discord",
                thread_id=thread_id,
                community_id=channel_id,
                title=f"#{channel_name} history",
                author_platform_user_id=None,
                created_at=spec.since or context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "channel_id": channel_id,
                    "channel_type": channel.get("type"),
                    "source_kind": "channel-history",
                },
                evidence_pointer=EvidencePointer(source_url=channel_url, fetched_at=context.observed_at),
            )
        )
        records.extend(
            self._message_records(
                channel_id=channel_id,
                community_id=channel_id,
                thread_id=thread_id,
                thread_url=channel_url,
                messages=self._iter_channel_messages(channel_id, spec.since, spec.until, spec.message_limit),
                context=context,
            )
        )
        return records

    def _crawl_thread(
        self,
        *,
        spec: DiscordTargetSpec,
        thread: dict[str, Any],
        channels: dict[str, dict[str, Any]],
        context: CrawlContext,
    ) -> list[Record]:
        thread_id = str(thread["id"])
        parent_id = str(thread.get("parent_id") or spec.guild_id)
        parent = channels.get(parent_id) or {}
        parent_name = parent.get("name") or parent_id
        parent_url = f"https://discord.com/channels/{spec.guild_id}/{parent_id}"
        thread_url = f"{parent_url}/{thread_id}"
        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id=parent_id,
                community_name=parent_name,
                community_type="forum" if int(parent.get("type", -1)) in FORUM_CHANNEL_TYPES else "channel",
                parent_community_id=spec.guild_id,
                description=parent.get("topic"),
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=EvidencePointer(source_url=parent_url, fetched_at=context.observed_at),
            ),
            ThreadRecord(
                record_type="thread",
                platform="discord",
                thread_id=thread_id,
                community_id=parent_id,
                title=thread.get("name") or thread_id,
                author_platform_user_id=None,
                created_at=thread.get("thread_metadata", {}).get("create_timestamp") or spec.since or context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "channel_id": parent_id,
                    "thread_type": thread.get("type"),
                    "archived": thread.get("thread_metadata", {}).get("archived"),
                },
                evidence_pointer=EvidencePointer(source_url=thread_url, fetched_at=context.observed_at),
            ),
        ]
        records.extend(
            self._message_records(
                channel_id=parent_id,
                community_id=parent_id,
                thread_id=thread_id,
                thread_url=thread_url,
                messages=self._iter_channel_messages(thread_id, spec.since, spec.until, spec.message_limit),
                context=context,
            )
        )
        return records

    def _message_records(
        self,
        *,
        channel_id: str,
        community_id: str,
        thread_id: str,
        thread_url: str,
        messages: Iterable[dict[str, Any]],
        context: CrawlContext,
    ) -> list[Record]:
        records: list[Record] = []
        seen_users: set[str] = set()

        for message in messages:
            author = message.get("author") or {}
            author_id = author.get("id")
            if not author_id:
                continue
            username = author.get("global_name") or author.get("username") or author_id
            if author_id not in seen_users:
                user_url = f"{thread_url}#user-{author_id}"
                records.extend(
                    [
                        AccountRecord(
                            record_type="account",
                            platform="discord",
                            platform_user_id=author_id,
                            username=username,
                            account_created_at=None,
                            first_observed_at=context.observed_at,
                            crawl_run_id=context.crawl_run_id,
                            evidence_pointer=EvidencePointer(source_url=user_url, fetched_at=context.observed_at),
                        ),
                        ProfileSnapshotRecord(
                            record_type="profile_snapshot",
                            platform="discord",
                            platform_user_id=author_id,
                            snapshot_at=context.observed_at,
                            crawl_run_id=context.crawl_run_id,
                            fields={"display_name": username},
                            evidence_pointer=EvidencePointer(source_url=user_url, fetched_at=context.observed_at),
                        ),
                    ]
                )
                seen_users.add(author_id)

            message_id = str(message["id"])
            reply_to_message_id = message.get("message_reference", {}).get("message_id")
            referenced = message.get("referenced_message") or {}
            reply_to_user_id = referenced.get("author", {}).get("id")
            content = message.get("content") or self._attachment_body(message.get("attachments", []))
            if not content:
                continue
            message_url = f"{thread_url}/{message_id}"
            created_at = message.get("timestamp") or context.observed_at
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="discord",
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=author_id,
                    body=content,
                    created_at=created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=reply_to_message_id,
                    reply_to_user_id=reply_to_user_id,
                    metadata={
                        "channel_id": channel_id,
                        "attachments": len(message.get("attachments", [])),
                        "reactions": len(message.get("reactions", [])),
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

    def _attachment_body(self, attachments: list[dict[str, Any]]) -> str:
        urls = [attachment.get("url") for attachment in attachments if attachment.get("url")]
        return "\n".join(urls)

    def _default_channel_ids(self, channels: Iterable[dict[str, Any]]) -> list[str]:
        return [
            str(channel["id"])
            for channel in channels
            if int(channel.get("type", -1)) in (TEXT_CHANNEL_TYPES | FORUM_CHANNEL_TYPES)
        ]

    def _iter_channel_messages(
        self,
        channel_id: str,
        since: str | None,
        until: str | None,
        message_limit: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        after: str | None = None
        since_dt = _parse_dt(since)
        until_dt = _parse_dt(until)
        yielded = 0
        while True:
            params = {"limit": "100"}
            if after:
                params["after"] = after
            batch = self._request_json("GET", f"/channels/{channel_id}/messages", params=params)
            if not isinstance(batch, list) or not batch:
                return
            stop = False
            for message in batch:
                created_at = _parse_dt(message.get("timestamp"))
                if until_dt and created_at and created_at >= until_dt:
                    continue
                if since_dt and created_at and created_at < since_dt:
                    stop = True
                    continue
                yield message
                yielded += 1
                if message_limit is not None and yielded >= message_limit:
                    return
            after = batch[-1]["id"]
            if stop or len(batch) < 100:
                return

    def _iter_forum_threads(self, guild_id: str, channel_id: str) -> Iterable[dict[str, Any]]:
        active = self._request_json("GET", f"/guilds/{guild_id}/threads/active")
        if isinstance(active, dict):
            for thread in active.get("threads", []):
                if isinstance(thread, dict) and str(thread.get("parent_id")) == channel_id:
                    yield thread
        archived = self._request_json(
            "GET",
            f"/channels/{channel_id}/threads/archived/public",
            params={"limit": "100"},
        )
        if isinstance(archived, dict):
            for thread in archived.get("threads", []):
                if isinstance(thread, dict):
                    yield thread

    def _get_guild(self, guild_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/guilds/{guild_id}")

    def _get_channels(self, guild_id: str) -> list[dict[str, Any]]:
        payload = self._request_json("GET", f"/guilds/{guild_id}/channels")
        return payload if isinstance(payload, list) else []

    def _get_channel(self, channel_id: str) -> dict[str, Any]:
        payload = self._request_json("GET", f"/channels/{channel_id}")
        return payload if isinstance(payload, dict) else {}

    def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
    ) -> Any:
        if method != "GET":
            raise ValueError(f"unsupported method: {method}")
        return self.client.get(path, params=params)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


LiveDiscordConnector = DiscordApiConnector
