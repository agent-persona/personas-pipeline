from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Iterable
from urllib.request import Request, urlopen

from ...core.base import CommunityConnector, CrawlContext
from ...core.models import (
    AccountRecord,
    CommunityRecord,
    CrawlTarget,
    EvidencePointer,
    MessageRecord,
    ProfileSnapshotRecord,
    Record,
    ThreadRecord,
)


def _clean_space(value: str | None) -> str | None:
    if value is None:
        return None
    clean = " ".join(value.split()).strip()
    return clean or None


@dataclass(slots=True)
class LinkedInOidcClient:
    access_token: str
    userinfo_endpoint: str = "https://api.linkedin.com/v2/userinfo"
    fetch_json: Any | None = None

    def userinfo(self) -> dict[str, Any]:
        if self.fetch_json is not None:
            return self.fetch_json(self.userinfo_endpoint)
        request = Request(
            self.userinfo_endpoint,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


class LinkedInOfficialConnector(CommunityConnector):
    platform = "linkedin"

    def __init__(self, client: LinkedInOidcClient) -> None:
        self.client = client

    @classmethod
    def from_env(
        cls,
        *,
        access_token_env: str = "LINKEDIN_ACCESS_TOKEN",
    ) -> "LinkedInOfficialConnector":
        access_token = _clean_space(os.environ.get(access_token_env))
        if not access_token:
            raise ValueError(f"missing LinkedIn access token; expected ${access_token_env}")
        return cls(LinkedInOidcClient(access_token=access_token))

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        del since
        self.validate_target(target)
        payload = self.client.userinfo()

        subject = _clean_space(str(payload.get("sub") or "")) or target.target_id
        name = _clean_space(str(payload.get("name") or "")) or target.community_name or subject
        given_name = _clean_space(str(payload.get("given_name") or ""))
        family_name = _clean_space(str(payload.get("family_name") or ""))
        picture = _clean_space(str(payload.get("picture") or ""))
        email = _clean_space(str(payload.get("email") or ""))
        locale = _clean_space(str(payload.get("locale") or ""))
        source_url = target.url or "https://api.linkedin.com/v2/userinfo"
        pointer = EvidencePointer(source_url=source_url, fetched_at=context.observed_at)
        community_id = target.target_id or subject
        platform_user_id = f"linkedin-user-{subject}"
        thread_id = f"linkedin-official-{subject}"

        fields = {
            "display_name": name,
            "given_name": given_name,
            "family_name": family_name,
            "image_url": picture,
            "email": email,
            "locale": locale,
            "source_mode": "official-oidc",
            "subject": subject,
        }
        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="linkedin",
                community_id=community_id,
                community_name=target.community_name or name,
                community_type="account",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            ),
            AccountRecord(
                record_type="account",
                platform="linkedin",
                platform_user_id=platform_user_id,
                username=name,
                account_created_at=None,
                first_observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            ),
            ProfileSnapshotRecord(
                record_type="profile_snapshot",
                platform="linkedin",
                platform_user_id=platform_user_id,
                snapshot_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                fields=fields,
                evidence_pointer=pointer,
            ),
            ThreadRecord(
                record_type="thread",
                platform="linkedin",
                thread_id=thread_id,
                community_id=community_id,
                title=f"LinkedIn OIDC profile: {name}",
                author_platform_user_id=platform_user_id,
                created_at=context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={"source_mode": "official-oidc"},
                evidence_pointer=pointer,
            ),
        ]

        ordinal = 0
        for source_kind, body in (
            ("name", name),
            ("email", email),
            ("locale", locale),
        ):
            if not body:
                continue
            ordinal += 1
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="linkedin",
                    message_id=f"linkedin-official-message-{subject}-{ordinal}",
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=platform_user_id,
                    body=body,
                    created_at=context.observed_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=None,
                    metadata={"source_kind": source_kind, "ordinal": ordinal},
                    evidence_pointer=pointer,
                )
            )
        return records
