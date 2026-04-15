from __future__ import annotations

from dataclasses import dataclass

from ..pipeline import CommunityConnector, CrawlTarget
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


@dataclass(frozen=True)
class DiscordSeedMessage:
    message_id: str
    author_id: str
    username: str
    body: str
    created_at: str
    observed_at: str
    reply_to_message_id: str | None = None
    reply_to_user_id: str | None = None


@dataclass(frozen=True)
class DiscordSeed:
    run_id: str
    observed_at: str
    server_id: str
    server_name: str
    channel_id: str
    channel_name: str
    thread_id: str
    thread_title: str
    thread_author_id: str
    thread_created_at: str
    invite_url: str
    messages: list[DiscordSeedMessage]


class DiscordSeedConnector(CommunityConnector):
    platform = "discord"

    def __init__(self, seed: DiscordSeed) -> None:
        self.seed = seed

    def fetch(self, target: CrawlTarget, since: str | None = None) -> list[BronzeRecord]:
        del since
        run_id = self.seed.run_id
        observed_at = self.seed.observed_at
        server_url = self.seed.invite_url
        channel_url = f"{server_url}#channel-{self.seed.channel_id}"
        thread_url = f"{channel_url}/thread/{self.seed.thread_id}"

        records: list[BronzeRecord] = [
            CommunityRecord(
                record_type="community",
                platform=self.platform,
                crawl_run_id=run_id,
                community_id=self.seed.server_id,
                community_name=self.seed.server_name,
                community_type="server",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=observed_at,
                evidence_pointer=EvidencePointer(source_url=server_url, fetched_at=observed_at),
            ),
            CommunityRecord(
                record_type="community",
                platform=self.platform,
                crawl_run_id=run_id,
                community_id=self.seed.channel_id,
                community_name=self.seed.channel_name,
                community_type="channel",
                parent_community_id=self.seed.server_id,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=observed_at,
                evidence_pointer=EvidencePointer(source_url=channel_url, fetched_at=observed_at),
            ),
            ThreadRecord(
                record_type="thread",
                platform=self.platform,
                crawl_run_id=run_id,
                thread_id=self.seed.thread_id,
                community_id=self.seed.channel_id,
                title=self.seed.thread_title,
                author_platform_user_id=self.seed.thread_author_id,
                created_at=self.seed.thread_created_at,
                observed_at=observed_at,
                metadata={"server_id": self.seed.server_id, "channel_id": self.seed.channel_id},
                evidence_pointer=EvidencePointer(source_url=thread_url, fetched_at=observed_at),
            ),
        ]

        seen_users: set[str] = set()
        for message in self.seed.messages:
            user_url = f"{server_url}#user-{message.author_id}"
            if message.author_id not in seen_users:
                records.extend(
                    [
                        AccountRecord(
                            record_type="account",
                            platform=self.platform,
                            crawl_run_id=run_id,
                            platform_user_id=message.author_id,
                            username=message.username,
                            account_created_at=None,
                            first_observed_at=observed_at,
                            evidence_pointer=EvidencePointer(source_url=user_url, fetched_at=observed_at),
                        ),
                        ProfileSnapshotRecord(
                            record_type="profile_snapshot",
                            platform=self.platform,
                            crawl_run_id=run_id,
                            platform_user_id=message.author_id,
                            snapshot_at=observed_at,
                            fields={"display_name": message.username},
                            evidence_pointer=EvidencePointer(source_url=user_url, fetched_at=observed_at),
                        ),
                    ]
                )
                seen_users.add(message.author_id)

            message_url = f"{thread_url}/message/{message.message_id}"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform=self.platform,
                    crawl_run_id=run_id,
                    message_id=message.message_id,
                    thread_id=self.seed.thread_id,
                    community_id=self.seed.channel_id,
                    author_platform_user_id=message.author_id,
                    body=message.body,
                    created_at=message.created_at,
                    observed_at=message.observed_at,
                    reply_to_message_id=message.reply_to_message_id,
                    reply_to_user_id=message.reply_to_user_id,
                    metadata={"channel_id": self.seed.channel_id},
                    evidence_pointer=EvidencePointer(source_url=message_url, fetched_at=observed_at),
                )
            )
            if message.reply_to_user_id:
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform=self.platform,
                        crawl_run_id=run_id,
                        interaction_type="reply",
                        source_user_id=message.author_id,
                        target_user_id=message.reply_to_user_id,
                        message_id=message.message_id,
                        thread_id=self.seed.thread_id,
                        community_id=self.seed.channel_id,
                        created_at=message.created_at,
                        evidence_pointer=EvidencePointer(
                            source_url=message_url,
                            fetched_at=observed_at,
                            derived_from_message_id=message.message_id,
                        ),
                    )
                )
        return records
