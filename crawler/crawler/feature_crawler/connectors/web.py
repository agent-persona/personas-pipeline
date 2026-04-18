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
class WebSeedComment:
    message_id: str
    author_id: str
    author_name: str
    body: str
    created_at: str
    observed_at: str
    reply_to_message_id: str | None = None
    reply_to_user_id: str | None = None


@dataclass(frozen=True)
class WebSeed:
    run_id: str
    observed_at: str
    site_id: str
    site_name: str
    source_url: str
    thread_id: str
    thread_title: str
    thread_author_id: str
    thread_created_at: str
    comments: list[WebSeedComment]


class WebSeedConnector(CommunityConnector):
    platform = "web"

    def __init__(self, seed: WebSeed) -> None:
        self.seed = seed

    def fetch(self, target: CrawlTarget, since: str | None = None) -> list[BronzeRecord]:
        del since
        run_id = self.seed.run_id
        observed_at = self.seed.observed_at
        base_url = self.seed.source_url

        records: list[BronzeRecord] = [
            CommunityRecord(
                record_type="community",
                platform=self.platform,
                crawl_run_id=run_id,
                community_id=self.seed.site_id,
                community_name=self.seed.site_name,
                community_type="site",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=observed_at,
                evidence_pointer=EvidencePointer(source_url=base_url, fetched_at=observed_at),
            ),
            ThreadRecord(
                record_type="thread",
                platform=self.platform,
                crawl_run_id=run_id,
                thread_id=self.seed.thread_id,
                community_id=self.seed.site_id,
                title=self.seed.thread_title,
                author_platform_user_id=self.seed.thread_author_id,
                created_at=self.seed.thread_created_at,
                observed_at=observed_at,
                metadata={"source_type": "approved-web"},
                evidence_pointer=EvidencePointer(source_url=base_url, fetched_at=observed_at),
            ),
        ]

        seen_users: set[str] = set()
        for comment in self.seed.comments:
            author_url = f"{base_url}#author-{comment.author_id}"
            if comment.author_id not in seen_users:
                records.extend(
                    [
                        AccountRecord(
                            record_type="account",
                            platform=self.platform,
                            crawl_run_id=run_id,
                            platform_user_id=comment.author_id,
                            username=comment.author_name,
                            account_created_at=None,
                            first_observed_at=observed_at,
                            evidence_pointer=EvidencePointer(source_url=author_url, fetched_at=observed_at),
                        ),
                        ProfileSnapshotRecord(
                            record_type="profile_snapshot",
                            platform=self.platform,
                            crawl_run_id=run_id,
                            platform_user_id=comment.author_id,
                            snapshot_at=observed_at,
                            fields={"display_name": comment.author_name},
                            evidence_pointer=EvidencePointer(source_url=author_url, fetched_at=observed_at),
                        ),
                    ]
                )
                seen_users.add(comment.author_id)

            message_url = f"{base_url}#comment-{comment.message_id}"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform=self.platform,
                    crawl_run_id=run_id,
                    message_id=comment.message_id,
                    thread_id=self.seed.thread_id,
                    community_id=self.seed.site_id,
                    author_platform_user_id=comment.author_id,
                    body=comment.body,
                    created_at=comment.created_at,
                    observed_at=comment.observed_at,
                    reply_to_message_id=comment.reply_to_message_id,
                    reply_to_user_id=comment.reply_to_user_id,
                    metadata={"source_type": "approved-web"},
                    evidence_pointer=EvidencePointer(source_url=message_url, fetched_at=observed_at),
                )
            )
            if comment.reply_to_user_id:
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform=self.platform,
                        crawl_run_id=run_id,
                        interaction_type="reply",
                        source_user_id=comment.author_id,
                        target_user_id=comment.reply_to_user_id,
                        message_id=comment.message_id,
                        thread_id=self.seed.thread_id,
                        community_id=self.seed.site_id,
                        created_at=comment.created_at,
                        evidence_pointer=EvidencePointer(
                            source_url=message_url,
                            fetched_at=observed_at,
                            derived_from_message_id=comment.message_id,
                        ),
                    )
                )
        return records
