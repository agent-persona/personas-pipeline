from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from json import loads
from os import getenv
from typing import Iterable
from urllib import robotparser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
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


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"


def _stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return sha256(joined.encode("utf-8")).hexdigest()[:16]


def _normalize_iso8601(value: str | None) -> str | None:
    if not value:
        return None
    clean = value.strip()
    if not clean:
        return None
    return clean if clean.endswith("Z") or "+" in clean else f"{clean}Z"


def _topic_json_url(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    if "/t/" not in parsed.path:
        return None
    if parsed.path.endswith(".json"):
        return raw_url
    path = parsed.path.rstrip("/")
    return parsed._replace(path=f"{path}.json", query="", fragment="").geturl()


@dataclass(slots=True)
class ParsedWebPage:
    title: str = ""
    author: str | None = None
    description: str | None = None
    published_at: str | None = None
    paragraphs: list[str] = field(default_factory=list)


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.page = ParsedWebPage()
        self._capture_title = False
        self._capture_text = False
        self._capture_depth = 0
        self._buffer: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value for key, value in attrs}
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._capture_title = True
        if tag in {"p", "li"}:
            self._capture_text = True
            self._capture_depth += 1
        if tag == "meta":
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            content = (attr_map.get("content") or "").strip()
            if not content:
                return
            if name in {"author", "article:author"}:
                self.page.author = content
            elif name in {"description", "og:description"} and not self.page.description:
                self.page.description = content
            elif name in {"article:published_time", "date", "pubdate", "publishdate"}:
                self.page.published_at = self.page.published_at or content
        if tag == "time" and not self.page.published_at:
            datetime_value = (attr_map.get("datetime") or "").strip()
            if datetime_value:
                self.page.published_at = datetime_value

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if tag == "title":
            self._capture_title = False
        if tag in {"p", "li"} and self._capture_depth > 0:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                self._capture_text = False
                text = " ".join(piece.strip() for piece in self._buffer if piece.strip())
                if len(text) >= 40:
                    self.page.paragraphs.append(text)
                self._buffer.clear()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._capture_title and not self.page.title:
            self.page.title = clean
        if self._capture_text:
            self._buffer.append(clean)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if clean:
            self.parts.append(clean)


def _clean_html_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(unescape(value))
    return " ".join(parser.parts).strip()


class ApprovedWebConnector(CommunityConnector):
    platform = "web"
    user_agent = "AgentPersonaCrawler/0.1"

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        self.validate_target(target)
        self._assert_robots(target.url)

        since_value = since or _normalize_iso8601(str(target.metadata.get("since") or "")) or None
        until_value = _normalize_iso8601(str(target.metadata.get("until") or "")) or None
        thread_mode = str(target.metadata.get("thread_mode") or "auto")
        if thread_mode not in {"auto", "discourse", "generic"}:
            raise ValueError(f"unsupported thread_mode: {thread_mode}")

        if thread_mode in {"auto", "discourse"}:
            topic_url = _topic_json_url(target.url)
            if topic_url:
                try:
                    payload = self._fetch_json(topic_url)
                except (HTTPError, URLError, ValueError):
                    payload = None
                if payload is not None:
                    return self._records_from_discourse(
                        target=target,
                        context=context,
                        payload=payload,
                        since=since_value,
                        until=until_value,
                    )
                if thread_mode == "discourse":
                    raise RuntimeError(f"failed to fetch discourse topic JSON for {target.url}")

        html = self._fetch_page_html(target)
        parsed = self._parse(html)
        source_url = target.url
        pointer = EvidencePointer(source_url=source_url, fetched_at=context.observed_at)
        parsed_url = urlparse(target.url)
        domain = parsed_url.netloc or target.community_name
        community_id = _slug(target.target_id or domain)
        thread_id = f"web-thread-{_stable_id(target.url)}"
        author_id = None
        author_name = parsed.author

        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="web",
                community_id=community_id,
                community_name=target.community_name,
                community_type="site",
                parent_community_id=None,
                description=parsed.description,
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            )
        ]

        if author_name:
            author_id = f"web-user-{_stable_id(domain, author_name)}"
            records.append(
                AccountRecord(
                    record_type="account",
                    platform="web",
                    platform_user_id=author_id,
                    username=author_name,
                    account_created_at=None,
                    first_observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    evidence_pointer=pointer,
                )
            )
            records.append(
                ProfileSnapshotRecord(
                    record_type="profile_snapshot",
                    platform="web",
                    platform_user_id=author_id,
                    snapshot_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    fields={"display_name": author_name},
                    evidence_pointer=pointer,
                )
            )

        records.append(
            ThreadRecord(
                record_type="thread",
                platform="web",
                thread_id=thread_id,
                community_id=community_id,
                title=parsed.title or target.metadata.get("title", target.url),
                author_platform_user_id=author_id,
                created_at=_normalize_iso8601(parsed.published_at) or context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "url": target.url,
                    "paragraph_count": len(parsed.paragraphs),
                    "collection_basis": target.collection_basis,
                    "thread_mode": "generic",
                },
                evidence_pointer=pointer,
            )
        )

        for index, paragraph in enumerate(parsed.paragraphs):
            message_id = f"web-message-{_stable_id(target.url, str(index), paragraph)}"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="web",
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=author_id,
                    body=paragraph,
                    created_at=context.observed_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=None,
                    metadata={"ordinal": index, "source_kind": "paragraph", "thread_mode": "generic"},
                    evidence_pointer=pointer,
                )
            )
        return records

    def _fetch_page_html(self, target: CrawlTarget) -> str:
        render_js = bool(target.metadata.get("render_js"))
        fetch_mode = str(target.metadata.get("fetch_mode") or "auto")
        if fetch_mode not in {"auto", "http", "playwright"}:
            raise ValueError(f"unsupported fetch_mode: {fetch_mode}")

        last_error: Exception | None = None
        if fetch_mode in {"auto", "http"}:
            try:
                html = self._fetch_html(target.url)
                if self._looks_like_robot_gate(html) and fetch_mode == "auto":
                    raise RuntimeError("encountered anti-bot or JS gate")
                return html
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if fetch_mode == "http" and not render_js:
                    raise

        if fetch_mode == "playwright" or render_js or last_error is not None:
            return self._fetch_html_with_playwright(target.url)

        raise RuntimeError(f"failed to fetch {target.url}: {last_error}")

    def _fetch_html(self, url: str) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urlopen(request, timeout=20) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding, errors="replace")

    def _fetch_json(self, url: str) -> dict[str, object]:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=20) as response:
            return loads(response.read().decode("utf-8"))

    def _assert_robots(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
        except OSError:
            return
        if parser.default_entry is None and not parser.entries:
            return
        if not parser.can_fetch(self.user_agent, url):
            raise PermissionError(f"robots.txt blocks crawling for {url}")

    def _parse(self, html: str) -> ParsedWebPage:
        parser = _PageParser()
        parser.feed(html)
        return parser.page

    def _looks_like_robot_gate(self, html: str) -> bool:
        lowered = html.lower()
        return "verify that you're not a robot" in lowered or "javascript is disabled" in lowered

    def _fetch_html_with_playwright(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("playwright is not installed") from exc

        executable_path = getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, executable_path=executable_path)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30_000)
                html = page.content()
                browser.close()
                return html
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "playwright fallback failed; run `python3 -m playwright install chromium`"
            ) from exc

    def _records_from_discourse(
        self,
        *,
        target: CrawlTarget,
        context: CrawlContext,
        payload: dict[str, object],
        since: str | None,
        until: str | None,
    ) -> list[Record]:
        source_url = target.url.rstrip("/")
        pointer = EvidencePointer(source_url=source_url, fetched_at=context.observed_at)
        parsed_url = urlparse(target.url)
        domain = parsed_url.netloc or target.community_name
        community_id = _slug(target.target_id or domain)
        topic_id = str(payload.get("id") or _stable_id(source_url))
        thread_id = f"web-thread-{topic_id}"
        post_stream = payload.get("post_stream")
        posts = post_stream.get("posts", []) if isinstance(post_stream, dict) else []
        if not isinstance(posts, list):
            posts = []

        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="web",
                community_id=community_id,
                community_name=target.community_name,
                community_type="site",
                parent_community_id=None,
                description=None,
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            )
        ]

        author_ids_by_username: dict[str, str] = {}
        message_ids_by_post_number: dict[int, str] = {}
        selected_posts: list[dict[str, object]] = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            created_at = _normalize_iso8601(str(post.get("created_at") or "")) or context.observed_at
            if since and created_at < since:
                continue
            if until and created_at > until:
                continue
            selected_posts.append(post)
            post_number = int(post.get("post_number") or 0)
            message_ids_by_post_number[post_number] = f"web-message-{post.get('id')}"
            username = str(post.get("username") or "").strip()
            if username:
                author_ids_by_username[username] = f"web-user-{_stable_id(domain, username)}"

        if not selected_posts and posts:
            thread_created = _normalize_iso8601(str(payload.get("created_at") or "")) or context.observed_at
            if (since and thread_created < since) or (until and thread_created > until):
                return records

        thread_author = None
        if posts and isinstance(posts[0], dict):
            first_username = str(posts[0].get("username") or "").strip()
            if first_username:
                thread_author = author_ids_by_username.get(first_username) or f"web-user-{_stable_id(domain, first_username)}"

        records.append(
            ThreadRecord(
                record_type="thread",
                platform="web",
                thread_id=thread_id,
                community_id=community_id,
                title=str(payload.get("title") or target.metadata.get("title") or target.url),
                author_platform_user_id=thread_author,
                created_at=_normalize_iso8601(str(payload.get("created_at") or "")) or context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "url": target.url,
                    "posts_count": int(payload.get("posts_count") or len(posts)),
                    "reply_count": int(payload.get("reply_count") or 0),
                    "thread_mode": "discourse",
                },
                evidence_pointer=pointer,
            )
        )

        seen_users: set[str] = set()
        for post in selected_posts:
            username = str(post.get("username") or "").strip()
            author_name = str(post.get("name") or username).strip() or username
            author_id = author_ids_by_username.get(username) or f"web-user-{_stable_id(domain, username or author_name)}"
            post_id = str(post.get("id"))
            post_number = int(post.get("post_number") or 0)
            created_at = _normalize_iso8601(str(post.get("created_at") or "")) or context.observed_at
            message_id = f"web-message-{post_id}"
            body = _clean_html_text(str(post.get("cooked") or post.get("raw") or ""))
            if not body:
                continue

            if author_id not in seen_users:
                user_pointer = EvidencePointer(
                    source_url=f"{source_url}/{post_number}",
                    fetched_at=context.observed_at,
                )
                records.append(
                    AccountRecord(
                        record_type="account",
                        platform="web",
                        platform_user_id=author_id,
                        username=author_name,
                        account_created_at=None,
                        first_observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=user_pointer,
                    )
                )
                records.append(
                    ProfileSnapshotRecord(
                        record_type="profile_snapshot",
                        platform="web",
                        platform_user_id=author_id,
                        snapshot_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        fields={"display_name": author_name},
                        evidence_pointer=user_pointer,
                    )
                )
                seen_users.add(author_id)

            reply_post_number = int(post.get("reply_to_post_number") or 0) or None
            reply_to_message_id = (
                message_ids_by_post_number.get(reply_post_number) if reply_post_number is not None else None
            )
            reply_username = str(post.get("reply_to_user") or "").strip()
            reply_to_user_id = author_ids_by_username.get(reply_username) if reply_username else None
            message_url = f"{source_url}/{post_number}"
            message_pointer = EvidencePointer(source_url=message_url, fetched_at=context.observed_at)
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="web",
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=author_id,
                    body=body,
                    created_at=created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=reply_to_message_id,
                    reply_to_user_id=reply_to_user_id,
                    metadata={
                        "post_number": post_number,
                        "thread_mode": "discourse",
                        "reaction_count": len(post.get("actions_summary") or []),
                    },
                    evidence_pointer=message_pointer,
                )
            )
            if reply_to_user_id:
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform="web",
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
