from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ...core.base import CommunityConnector, CrawlContext
from ...core.models import (
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


def _stable_id(*parts: str) -> str:
    return sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def _to_iso(value: float | int | None, fallback: str) -> str:
    if value is None:
        return fallback
    return (
        datetime.fromtimestamp(float(value), tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(slots=True)
class RedditApiClient:
    client_id: str
    client_secret: str
    user_agent: str
    fetch_json: Any | None = None
    token_base: str = "https://www.reddit.com"
    api_base: str = "https://oauth.reddit.com"

    def _request_token(self) -> str:
        encoded = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("ascii")
        request = Request(
            f"{self.token_base}/api/v1/access_token",
            method="POST",
            data=urlencode({"grant_type": "client_credentials"}).encode("utf-8"),
            headers={
                "Authorization": f"Basic {encoded}",
                "User-Agent": self.user_agent,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        token = str(payload.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("reddit access token missing from OAuth response")
        return token

    def get(self, path: str, params: dict[str, object] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        full_path = f"{path}{query}"
        if self.fetch_json is not None:
            return self.fetch_json(full_path)
        token = self._request_token()
        request = Request(
            f"{self.api_base}{full_path}",
            method="GET",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass(slots=True)
class RedditPublicJsonClient:
    user_agent: str
    fetch_json: Any | None = None
    base_url: str = "https://www.reddit.com"

    def get(self, path: str, params: dict[str, object] | None = None) -> Any:
        query_params = {"raw_json": 1}
        if params:
            query_params.update(params)
        full_path = self._json_path(path)
        query = urlencode(query_params)
        if self.fetch_json is not None:
            return self.fetch_json(f"{full_path}?{query}")
        request = Request(
            f"{self.base_url}{full_path}?{query}",
            method="GET",
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                raise RuntimeError("reddit public-json endpoint rate-limited; retry later or use OAuth") from exc
            raise

    def _json_path(self, path: str) -> str:
        clean = path if path.startswith("/") else f"/{path}"
        if clean.endswith(".json"):
            return clean
        if clean.endswith("/about"):
            return f"{clean}.json"
        if "/comments/" in clean:
            return f"{clean.rstrip('/')}.json"
        return f"{clean.rstrip('/')}.json"


class RedditApiConnector(CommunityConnector):
    platform = "reddit"

    def __init__(self, client: RedditApiClient) -> None:
        self.client = client

    @classmethod
    def from_env(
        cls,
        client_id_env: str = "REDDIT_CLIENT_ID",
        client_secret_env: str = "REDDIT_CLIENT_SECRET",
        user_agent_env: str = "REDDIT_USER_AGENT",
    ) -> "RedditApiConnector":
        client_id = os.environ.get(client_id_env)
        client_secret = os.environ.get(client_secret_env)
        user_agent = os.environ.get(user_agent_env) or "agent-personas-crawler/0.1"
        if not client_id or not client_secret:
            raise ValueError(
                f"missing Reddit credentials; expected ${client_id_env} and ${client_secret_env}"
            )
        return cls(RedditApiClient(client_id=client_id, client_secret=client_secret, user_agent=user_agent))

    @classmethod
    def from_public_json(
        cls,
        user_agent_env: str = "REDDIT_USER_AGENT",
    ) -> "RedditApiConnector":
        user_agent = os.environ.get(user_agent_env) or "agent-personas-crawler/0.1"
        return cls(RedditPublicJsonClient(user_agent=user_agent))

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        self.validate_target(target)
        subreddit = str(target.metadata.get("subreddit") or target.target_id)
        sort = str(target.metadata.get("sort") or "new")
        limit = int(target.metadata.get("limit") or 25)
        comment_limit = int(target.metadata.get("comment_limit") or 128)
        until = str(target.metadata.get("until") or "") or None

        about = self._unwrap_data(self.client.get(f"/r/{subreddit}/about"))
        community_name = str(about.get("display_name_prefixed") or f"r/{subreddit}")
        community_url = f"https://www.reddit.com/r/{subreddit}/"
        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="reddit",
                community_id=subreddit,
                community_name=community_name,
                community_type="subreddit",
                parent_community_id=None,
                description=about.get("public_description") or about.get("title"),
                member_count=about.get("subscribers"),
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=EvidencePointer(source_url=community_url, fetched_at=context.observed_at),
            )
        ]

        seen_accounts: set[str] = set()
        listing_payload = self.client.get(
            f"/r/{subreddit}/{sort}",
            params={"limit": limit, "raw_json": 1},
        )
        for child in self._iter_listing_children(listing_payload):
            submission = self._unwrap_data(child)
            created_at = _to_iso(submission.get("created_utc"), context.observed_at)
            if since and created_at < since:
                continue
            if until and created_at > until:
                continue
            records.extend(
                self._records_for_submission(
                    subreddit=subreddit,
                    submission=submission,
                    context=context,
                    seen_accounts=seen_accounts,
                    comment_limit=comment_limit,
                )
            )

        return records

    def _records_for_submission(
        self,
        *,
        subreddit: str,
        submission: dict[str, Any],
        context: CrawlContext,
        seen_accounts: set[str],
        comment_limit: int,
    ) -> list[Record]:
        submission_id = str(submission.get("id") or "")
        if not submission_id:
            return []
        permalink = str(submission.get("permalink") or f"/r/{subreddit}/comments/{submission_id}/")
        thread_url = f"https://www.reddit.com{permalink}"
        thread_id = f"reddit-thread-{submission_id}"
        thread_author_id = self._author_id(submission.get("author"))
        thread_created_at = _to_iso(submission.get("created_utc"), context.observed_at)
        title = str(submission.get("title") or thread_url)
        records: list[Record] = []
        records.extend(
            self._maybe_account_records(
                author_name=submission.get("author"),
                author_id=thread_author_id,
                source_url=thread_url,
                context=context,
                seen_accounts=seen_accounts,
            )
        )
        records.append(
            ThreadRecord(
                record_type="thread",
                platform="reddit",
                thread_id=thread_id,
                community_id=subreddit,
                title=title,
                author_platform_user_id=thread_author_id,
                created_at=thread_created_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "permalink": permalink,
                    "score": submission.get("score"),
                    "sort_hint": submission.get("suggested_sort"),
                    "num_comments": submission.get("num_comments"),
                    "source_url": thread_url,
                },
                evidence_pointer=EvidencePointer(source_url=thread_url, fetched_at=context.observed_at),
            )
        )

        submission_body = str(submission.get("selftext") or submission.get("url") or "").strip()
        op_message_id = f"t3_{submission_id}"
        if submission_body:
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="reddit",
                    message_id=op_message_id,
                    thread_id=thread_id,
                    community_id=subreddit,
                    author_platform_user_id=thread_author_id,
                    body=submission_body,
                    created_at=thread_created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=None,
                    metadata={"source_kind": "submission"},
                    evidence_pointer=EvidencePointer(source_url=thread_url, fetched_at=context.observed_at),
                )
            )

        comments_payload = self.client.get(
            f"/r/{subreddit}/comments/{submission_id}",
            params={"limit": comment_limit, "depth": 16, "raw_json": 1, "sort": "top"},
        )
        comment_author_ids: dict[str, str] = {}
        for comment in self._iter_comment_nodes(comments_payload):
            if str(comment.get("kind")) != "t1":
                continue
            data = self._unwrap_data(comment)
            comment_name = str(data.get("name") or "")
            comment_id = str(data.get("id") or "")
            if not comment_name or not comment_id:
                continue
            author_id = self._author_id(data.get("author"))
            comment_author_ids[comment_name] = author_id or ""
            records.extend(
                self._maybe_account_records(
                    author_name=data.get("author"),
                    author_id=author_id,
                    source_url=f"{thread_url}{comment_id}/",
                    context=context,
                    seen_accounts=seen_accounts,
                )
            )
            body = str(data.get("body") or "").strip()
            if not body:
                continue
            created_at = _to_iso(data.get("created_utc"), context.observed_at)
            parent_fullname = str(data.get("parent_id") or "")
            reply_to_message_id = parent_fullname if parent_fullname.startswith("t1_") else op_message_id
            reply_to_user_id = None
            if parent_fullname.startswith("t1_"):
                parent_author_id = comment_author_ids.get(parent_fullname)
                reply_to_user_id = parent_author_id or None
            elif parent_fullname.startswith("t3_"):
                reply_to_user_id = thread_author_id

            comment_url = f"{thread_url}{comment_id}/"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="reddit",
                    message_id=comment_name,
                    thread_id=thread_id,
                    community_id=subreddit,
                    author_platform_user_id=author_id,
                    body=body,
                    created_at=created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=reply_to_message_id,
                    reply_to_user_id=reply_to_user_id,
                    metadata={
                        "source_kind": "comment",
                        "depth": data.get("depth"),
                        "score": data.get("score"),
                    },
                    evidence_pointer=EvidencePointer(source_url=comment_url, fetched_at=context.observed_at),
                )
            )
            if author_id and reply_to_user_id:
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform="reddit",
                        interaction_type="reply",
                        source_user_id=author_id,
                        target_user_id=reply_to_user_id,
                        message_id=comment_name,
                        thread_id=thread_id,
                        community_id=subreddit,
                        created_at=created_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(
                            source_url=comment_url,
                            fetched_at=context.observed_at,
                            derived_from_message_id=comment_name,
                        ),
                    )
                )
        return records

    def _maybe_account_records(
        self,
        *,
        author_name: object,
        author_id: str | None,
        source_url: str,
        context: CrawlContext,
        seen_accounts: set[str],
    ) -> list[Record]:
        if not author_id or author_id in seen_accounts:
            return []
        username = str(author_name or "").strip()
        if not username:
            return []
        seen_accounts.add(author_id)
        pointer = EvidencePointer(source_url=source_url, fetched_at=context.observed_at)
        return [
            AccountRecord(
                record_type="account",
                platform="reddit",
                platform_user_id=author_id,
                username=username,
                account_created_at=None,
                first_observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            ),
            ProfileSnapshotRecord(
                record_type="profile_snapshot",
                platform="reddit",
                platform_user_id=author_id,
                snapshot_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                fields={"display_name": username},
                evidence_pointer=pointer,
            ),
        ]

    def _unwrap_data(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload if isinstance(payload, dict) else {}

    def _iter_listing_children(self, payload: Any) -> Iterable[dict[str, Any]]:
        data = self._unwrap_data(payload)
        children = data.get("children", [])
        if not isinstance(children, list):
            return []
        return [child for child in children if isinstance(child, dict)]

    def _iter_comment_nodes(self, payload: Any) -> Iterable[dict[str, Any]]:
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        comment_listing = payload[1]
        stack = list(reversed(self._iter_listing_children(comment_listing)))
        nodes: list[dict[str, Any]] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            replies = self._unwrap_data(node).get("replies")
            if isinstance(replies, dict):
                stack.extend(reversed(self._iter_listing_children(replies)))
        return nodes

    def _author_id(self, author_name: object) -> str | None:
        username = str(author_name or "").strip()
        if not username or username == "[deleted]":
            return None
        return f"reddit-user-{_stable_id(username)}"
