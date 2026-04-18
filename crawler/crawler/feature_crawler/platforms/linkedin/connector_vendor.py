from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from typing import Any, Iterable
from urllib.parse import urlencode, urlparse
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


def _clean_space(value: Any) -> str | None:
    if value is None:
        return None
    clean = " ".join(str(value).split()).strip()
    return clean or None


def _first(*values: Any) -> str | None:
    for value in values:
        clean = _clean_space(value)
        if clean:
            return clean
    return None


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _public_identifier(url: str, fallback: str) -> str:
    path = urlparse(url).path
    marker = "/in/"
    if marker in path:
        return path.split(marker, 1)[1].split("/", 1)[0].split("?", 1)[0] or fallback
    return fallback


def _normalize_iso(value: Any, fallback: str) -> str:
    clean = _clean_space(value)
    if not clean:
        return fallback
    candidate = clean.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_in_window(value: str, start: str | None, end: str | None) -> bool:
    current = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if start:
        lower = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if current < lower:
            return False
    if end:
        upper = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if current > upper:
            return False
    return True


class _HttpJsonClient:
    def __init__(self, fetch_json: Any | None = None) -> None:
        self.fetch_json = fetch_json

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: Any | None = None,
    ) -> Any:
        if self.fetch_json is not None:
            return self.fetch_json(method, url, headers, json_body)
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers = {**headers, "Content-Type": "application/json"}
        request = Request(url, method=method, headers=headers, data=data)
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


class ApifyVendorClient(_HttpJsonClient):
    def __init__(
        self,
        token: str,
        *,
        profile_actor: str,
        posts_actor: str | None = None,
        network_actor: str | None = None,
        fetch_json: Any | None = None,
        base_url: str = "https://api.apify.com/v2",
    ) -> None:
        super().__init__(fetch_json)
        self.token = token
        self.profile_actor = profile_actor
        self.posts_actor = posts_actor
        self.network_actor = network_actor
        self.base_url = base_url.rstrip("/")

    def fetch_profile(self, profile_url: str, _: str) -> dict[str, Any]:
        items = self._run_actor(self.profile_actor, {"profileUrls": [profile_url], "maxProfiles": 1})
        return items[0] if items else {}

    def fetch_posts(self, profile_url: str, _: str, *, limit: int, page_limit: int) -> list[dict[str, Any]]:
        if not self.posts_actor:
            return []
        items = self._run_actor(self.posts_actor, {"profileUrls": [profile_url], "maxPosts": limit, "pageLimit": page_limit})
        return [item for item in items if isinstance(item, dict)]

    def fetch_connections(self, profile_url: str, _: str, *, limit: int, page_limit: int) -> list[dict[str, Any]]:
        if not self.network_actor:
            return []
        items = self._run_actor(self.network_actor, {"profileUrls": [profile_url], "maxItems": limit, "pageLimit": page_limit})
        return [item for item in items if isinstance(item, dict)]

    def _run_actor(self, actor_id: str, run_input: dict[str, Any]) -> list[Any]:
        actor_path = actor_id.replace("/", "~")
        run_url = f"{self.base_url}/acts/{actor_path}/runs?token={self.token}&waitForFinish=120"
        run = self._request_json(method="POST", url=run_url, headers={"Accept": "application/json"}, json_body=run_input)
        dataset_id = (
            run.get("data", {}).get("defaultDatasetId")
            if isinstance(run, dict)
            else None
        ) or run.get("defaultDatasetId")
        if not dataset_id:
            return []
        items_url = f"{self.base_url}/datasets/{dataset_id}/items?token={self.token}&clean=true"
        items = self._request_json(method="GET", url=items_url, headers={"Accept": "application/json"})
        return items if isinstance(items, list) else []


class BrightDataVendorClient(_HttpJsonClient):
    def __init__(
        self,
        api_key: str,
        *,
        profile_dataset_id: str,
        posts_dataset_id: str | None = None,
        network_dataset_id: str | None = None,
        fetch_json: Any | None = None,
        base_url: str = "https://api.brightdata.com/datasets/v3",
    ) -> None:
        super().__init__(fetch_json)
        self.api_key = api_key
        self.profile_dataset_id = profile_dataset_id
        self.posts_dataset_id = posts_dataset_id
        self.network_dataset_id = network_dataset_id
        self.base_url = base_url.rstrip("/")

    def fetch_profile(self, profile_url: str, _: str) -> dict[str, Any]:
        items = self._scrape(self.profile_dataset_id, [{"url": profile_url}])
        return items[0] if items else {}

    def fetch_posts(self, profile_url: str, _: str, *, limit: int, page_limit: int) -> list[dict[str, Any]]:
        if not self.posts_dataset_id:
            return []
        return self._scrape(self.posts_dataset_id, [{"url": profile_url, "limit": limit, "page_limit": page_limit}])

    def fetch_connections(self, profile_url: str, _: str, *, limit: int, page_limit: int) -> list[dict[str, Any]]:
        if not self.network_dataset_id:
            return []
        return self._scrape(self.network_dataset_id, [{"url": profile_url, "limit": limit, "page_limit": page_limit}])

    def _scrape(self, dataset_id: str, body: list[dict[str, Any]]) -> list[dict[str, Any]]:
        url = f"{self.base_url}/scrape?dataset_id={dataset_id}&format=json"
        payload = self._request_json(
            method="POST",
            url=url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"},
            json_body=body,
        )
        return payload if isinstance(payload, list) else []


class LinkdApiVendorClient(_HttpJsonClient):
    def __init__(
        self,
        api_key: str,
        *,
        fetch_json: Any | None = None,
        base_url: str = "https://linkdapi.com/api/v1",
    ) -> None:
        super().__init__(fetch_json)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def fetch_profile(self, _: str, public_identifier: str) -> dict[str, Any]:
        return self._get("/profile/full", {"username": public_identifier})

    def fetch_posts(self, _: str, public_identifier: str, *, limit: int, page_limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for page in range(1, page_limit + 1):
            payload = self._get("/profile/posts", {"username": public_identifier, "page": page, "limit": limit})
            page_items = _listify(payload.get("posts") if isinstance(payload, dict) else payload)
            if not page_items:
                break
            items.extend(item for item in page_items if isinstance(item, dict))
            if len(items) >= limit:
                break
        return items[:limit]

    def fetch_connections(self, _: str, public_identifier: str, *, limit: int, page_limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for page in range(1, page_limit + 1):
            payload = self._get("/profile/connections", {"username": public_identifier, "page": page, "limit": limit})
            page_items = _listify(payload.get("connections") if isinstance(payload, dict) else payload)
            if not page_items:
                break
            items.extend(item for item in page_items if isinstance(item, dict))
            if len(items) >= limit:
                break
        return items[:limit]

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        query = urlencode(params)
        return self._request_json(
            method="GET",
            url=f"{self.base_url}{path}?{query}",
            headers={"Accept": "application/json", "x-api-key": self.api_key},
        )


class LinkedInVendorConnector(CommunityConnector):
    platform = "linkedin"

    def __init__(self, vendor: str, client: Any) -> None:
        self.vendor = vendor
        self.client = client

    @classmethod
    def from_env(cls, *, vendor: str) -> "LinkedInVendorConnector":
        if vendor == "apify":
            token = os.environ.get("APIFY_TOKEN")
            if not token:
                raise ValueError("missing $APIFY_TOKEN")
            profile_actor = os.environ.get("LINKEDIN_APIFY_PROFILE_ACTOR", "apimaestro/linkedin-profile-scraper")
            posts_actor = os.environ.get("LINKEDIN_APIFY_POSTS_ACTOR")
            network_actor = os.environ.get("LINKEDIN_APIFY_NETWORK_ACTOR")
            return cls(vendor, ApifyVendorClient(token, profile_actor=profile_actor, posts_actor=posts_actor, network_actor=network_actor))
        if vendor == "brightdata":
            api_key = os.environ.get("BRIGHTDATA_API_KEY")
            dataset_id = os.environ.get("LINKEDIN_BRIGHTDATA_PROFILE_DATASET_ID")
            if not api_key or not dataset_id:
                raise ValueError("missing $BRIGHTDATA_API_KEY or $LINKEDIN_BRIGHTDATA_PROFILE_DATASET_ID")
            return cls(
                vendor,
                BrightDataVendorClient(
                    api_key,
                    profile_dataset_id=dataset_id,
                    posts_dataset_id=os.environ.get("LINKEDIN_BRIGHTDATA_POSTS_DATASET_ID"),
                    network_dataset_id=os.environ.get("LINKEDIN_BRIGHTDATA_NETWORK_DATASET_ID"),
                ),
            )
        if vendor == "linkdapi":
            api_key = os.environ.get("LINKDAPI_API_KEY")
            if not api_key:
                raise ValueError("missing $LINKDAPI_API_KEY")
            return cls(vendor, LinkdApiVendorClient(api_key))
        raise ValueError(f"unsupported vendor: {vendor}")

    def fetch(self, target: CrawlTarget, context: CrawlContext, since: str | None = None) -> Iterable[Record]:
        self.validate_target(target)
        public_identifier = _public_identifier(target.url, target.target_id)
        include_posts = bool(target.metadata.get("include_posts"))
        include_network = bool(target.metadata.get("include_network"))
        post_limit = int(target.metadata.get("post_limit") or 25)
        comment_limit = int(target.metadata.get("comment_limit") or 64)
        network_limit = int(target.metadata.get("network_limit") or 50)
        page_limit = int(target.metadata.get("page_limit") or 3)
        until = _clean_space(target.metadata.get("until"))

        profile_payload = self.client.fetch_profile(target.url, public_identifier)
        profile = self._normalize_profile(profile_payload, target.url, public_identifier)
        profile_user_id = f"linkedin-user-{profile['public_identifier']}"
        pointer = EvidencePointer(source_url=profile["profile_url"], fetched_at=context.observed_at)

        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="linkedin",
                community_id=target.target_id or profile["public_identifier"],
                community_name=target.community_name or profile["name"],
                community_type="account",
                parent_community_id=None,
                description=profile.get("headline") or profile.get("about"),
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            ),
            AccountRecord(
                record_type="account",
                platform="linkedin",
                platform_user_id=profile_user_id,
                username=profile["name"],
                account_created_at=None,
                first_observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            ),
            ProfileSnapshotRecord(
                record_type="profile_snapshot",
                platform="linkedin",
                platform_user_id=profile_user_id,
                snapshot_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                fields={
                    "display_name": profile["name"],
                    "headline": profile.get("headline"),
                    "about": profile.get("about"),
                    "location": profile.get("location"),
                    "image_url": profile.get("image_url"),
                    "profile_url": profile["profile_url"],
                    "public_identifier": profile["public_identifier"],
                    "experience": profile.get("experience", []),
                    "source_mode": self.vendor,
                },
                evidence_pointer=pointer,
            ),
        ]

        records.extend(self._profile_messages(profile, profile_user_id, target.target_id or profile["public_identifier"], context, pointer))

        if include_posts:
            posts = self.client.fetch_posts(target.url, public_identifier, limit=post_limit, page_limit=page_limit)
            records.extend(self._post_records(posts, community_id=target.target_id or profile["public_identifier"], profile_user_id=profile_user_id, context=context, source_url=profile["profile_url"], since=since, until=until, comment_limit=comment_limit))

        if include_network:
            connections = self.client.fetch_connections(target.url, public_identifier, limit=network_limit, page_limit=page_limit)
            records.extend(self._network_records(connections, community_id=target.target_id or profile["public_identifier"], source_user_id=profile_user_id, context=context, source_url=profile["profile_url"]))

        return records

    def _profile_messages(self, profile: dict[str, Any], profile_user_id: str, community_id: str, context: CrawlContext, pointer: EvidencePointer) -> list[Record]:
        thread_id = f"linkedin-profile-{profile['public_identifier']}"
        records: list[Record] = [
            ThreadRecord(
                record_type="thread",
                platform="linkedin",
                thread_id=thread_id,
                community_id=community_id,
                title=f"LinkedIn profile: {profile['name']}",
                author_platform_user_id=profile_user_id,
                created_at=context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={"source_mode": self.vendor},
                evidence_pointer=pointer,
            )
        ]
        ordinal = 0
        for source_kind, body in (
            ("headline", profile.get("headline")),
            ("about", profile.get("about")),
            *[("experience", item) for item in profile.get("experience", [])],
        ):
            clean = _clean_space(body)
            if not clean:
                continue
            ordinal += 1
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="linkedin",
                    message_id=f"linkedin-profile-message-{profile['public_identifier']}-{ordinal}",
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=profile_user_id,
                    body=clean,
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

    def _post_records(self, posts: list[dict[str, Any]], *, community_id: str, profile_user_id: str, context: CrawlContext, source_url: str, since: str | None, until: str | None, comment_limit: int) -> list[Record]:
        records: list[Record] = []
        for post in posts:
            post_id = _first(post.get("id"), post.get("urn"), post.get("activityId")) or _stable_id(json.dumps(post, sort_keys=True))
            body = _first(post.get("text"), post.get("body"), post.get("content"), post.get("commentary")) or ""
            created_at = _normalize_iso(_first(post.get("createdAt"), post.get("postedAt"), post.get("date")), context.observed_at)
            if not _iso_in_window(created_at, since, until):
                continue
            thread_id = f"linkedin-post-{post_id}"
            message_id = f"linkedin-post-message-{post_id}"
            post_url = _first(post.get("url"), post.get("postUrl"), source_url) or source_url
            pointer = EvidencePointer(source_url=post_url, fetched_at=context.observed_at)
            records.append(
                ThreadRecord(
                    record_type="thread",
                    platform="linkedin",
                    thread_id=thread_id,
                    community_id=community_id,
                    title=_first(post.get("title"), body[:80], f"LinkedIn post {post_id}") or f"LinkedIn post {post_id}",
                    author_platform_user_id=profile_user_id,
                    created_at=created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    metadata={"source_mode": self.vendor, "post_id": post_id},
                    evidence_pointer=pointer,
                )
            )
            if body:
                records.append(
                    MessageRecord(
                        record_type="message",
                        platform="linkedin",
                        message_id=message_id,
                        thread_id=thread_id,
                        community_id=community_id,
                        author_platform_user_id=profile_user_id,
                        body=body,
                        created_at=created_at,
                        observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        reply_to_message_id=None,
                        reply_to_user_id=None,
                        metadata={"source_kind": "post", "post_id": post_id},
                        evidence_pointer=pointer,
                    )
                )
            comments = _listify(post.get("comments"))[:comment_limit]
            author_by_message: dict[str, str] = {message_id: profile_user_id}
            for index, comment in enumerate(comments, start=1):
                comment_id = _first(comment.get("id"), comment.get("urn"), f"{post_id}-comment-{index}") or f"{post_id}-comment-{index}"
                comment_author_name = _first(comment.get("authorName"), comment.get("author"), comment.get("username"), "unknown") or "unknown"
                comment_author_identifier = _first(comment.get("authorPublicIdentifier"), comment.get("publicIdentifier"), comment_author_name) or comment_author_name
                comment_author_id = f"linkedin-user-{comment_author_identifier}"
                comment_body = _first(comment.get("text"), comment.get("body"), comment.get("content"))
                if not comment_body:
                    continue
                comment_created_at = _normalize_iso(_first(comment.get("createdAt"), comment.get("date")), created_at)
                parent_message_id = _first(comment.get("replyToMessageId")) or message_id
                parent_user_id = author_by_message.get(parent_message_id, profile_user_id)
                comment_message_id = f"linkedin-comment-{comment_id}"
                author_by_message[comment_message_id] = comment_author_id
                records.append(
                    MessageRecord(
                        record_type="message",
                        platform="linkedin",
                        message_id=comment_message_id,
                        thread_id=thread_id,
                        community_id=community_id,
                        author_platform_user_id=comment_author_id,
                        body=comment_body,
                        created_at=comment_created_at,
                        observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        reply_to_message_id=parent_message_id,
                        reply_to_user_id=parent_user_id,
                        metadata={"source_kind": "comment", "post_id": post_id},
                        evidence_pointer=pointer,
                    )
                )
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform="linkedin",
                        interaction_type="reply",
                        source_user_id=comment_author_id,
                        target_user_id=parent_user_id,
                        message_id=comment_message_id,
                        thread_id=thread_id,
                        community_id=community_id,
                        created_at=comment_created_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(source_url=post_url, fetched_at=context.observed_at, derived_from_message_id=comment_message_id),
                    )
                )
        return records

    def _network_records(self, connections: list[dict[str, Any]], *, community_id: str, source_user_id: str, context: CrawlContext, source_url: str) -> list[Record]:
        thread_id = f"linkedin-network-{community_id}"
        pointer = EvidencePointer(source_url=source_url, fetched_at=context.observed_at)
        records: list[Record] = [
            ThreadRecord(
                record_type="thread",
                platform="linkedin",
                thread_id=thread_id,
                community_id=community_id,
                title=f"LinkedIn network: {community_id}",
                author_platform_user_id=source_user_id,
                created_at=context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={"source_mode": self.vendor, "scope": "network"},
                evidence_pointer=pointer,
            )
        ]
        for index, connection in enumerate(connections, start=1):
            name = _first(connection.get("name"), connection.get("fullName"), connection.get("headline"), "unknown") or "unknown"
            public_identifier = _first(connection.get("publicIdentifier"), connection.get("username"), connection.get("slug"), _stable_id(name, str(index))) or _stable_id(name, str(index))
            target_user_id = f"linkedin-user-{public_identifier}"
            body = _first(connection.get("headline"), connection.get("occupation"), name) or name
            message_id = f"linkedin-network-message-{public_identifier}"
            records.append(
                AccountRecord(
                    record_type="account",
                    platform="linkedin",
                    platform_user_id=target_user_id,
                    username=name,
                    account_created_at=None,
                    first_observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    evidence_pointer=pointer,
                )
            )
            records.append(
                ProfileSnapshotRecord(
                    record_type="profile_snapshot",
                    platform="linkedin",
                    platform_user_id=target_user_id,
                    snapshot_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    fields={"display_name": name, "headline": _first(connection.get("headline"), connection.get("occupation")), "public_identifier": public_identifier, "source_mode": self.vendor},
                    evidence_pointer=pointer,
                )
            )
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="linkedin",
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=source_user_id,
                    body=body,
                    created_at=context.observed_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=target_user_id,
                    metadata={"source_kind": "connection", "ordinal": index},
                    evidence_pointer=pointer,
                )
            )
            records.append(
                InteractionRecord(
                    record_type="interaction",
                    platform="linkedin",
                    interaction_type="connection",
                    source_user_id=source_user_id,
                    target_user_id=target_user_id,
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    created_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    evidence_pointer=EvidencePointer(source_url=source_url, fetched_at=context.observed_at, derived_from_message_id=message_id),
                )
            )
        return records

    def _normalize_profile(self, payload: dict[str, Any], source_url: str, public_identifier: str) -> dict[str, Any]:
        experience = _listify(payload.get("experience") or payload.get("positions"))
        exp_texts = [
            _first(item.get("title"), item.get("headline"), item.get("companyName"), item.get("company"))
            for item in experience
            if isinstance(item, dict)
        ]
        return {
            "public_identifier": _first(payload.get("publicIdentifier"), payload.get("username"), public_identifier) or public_identifier,
            "name": _first(payload.get("fullName"), payload.get("name"), payload.get("firstName")) or public_identifier,
            "headline": _first(payload.get("headline"), payload.get("occupation")),
            "about": _first(payload.get("summary"), payload.get("about"), payload.get("description")),
            "location": _first(payload.get("location"), payload.get("geo")),
            "image_url": _first(payload.get("profilePicture"), payload.get("profilePictureUrl"), payload.get("image")),
            "profile_url": _first(payload.get("profileUrl"), payload.get("url"), source_url) or source_url,
            "experience": [item for item in exp_texts if item],
        }
