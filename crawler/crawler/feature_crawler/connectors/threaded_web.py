from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib import robotparser
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


def _normalize_iso(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    candidate = value.strip()
    if not candidate:
        return fallback
    candidate = candidate.replace("Z", "+00:00")
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


@dataclass(slots=True)
class HtmlNode:
    tag: str
    attrs: dict[str, str]
    children: list["HtmlNode"] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    def add_text(self, text: str) -> None:
        clean = " ".join(text.split())
        if clean:
            self.text_parts.append(clean)

    @property
    def text(self) -> str:
        parts = [*self.text_parts]
        for child in self.children:
            if child.text:
                parts.append(child.text)
        return " ".join(parts).strip()


class _DomBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root = HtmlNode(tag="document", attrs={})
        self.stack = [self.root]
        self.title = ""
        self._in_title = False
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        if tag == "meta":
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            content = attr_map.get("content", "").strip()
            if name and content:
                self.meta[name] = content
            return
        if tag == "title":
            self._in_title = True
        node = HtmlNode(tag=tag, attrs=attr_map)
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if len(self.stack) > 1:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            clean = " ".join(data.split())
            if clean and not self.title:
                self.title = clean
        self.stack[-1].add_text(data)


@dataclass(slots=True)
class ParsedPost:
    message_id: str
    author_id: str
    author_name: str
    body: str
    created_at: str
    reply_to_message_id: str | None = None
    reply_to_user_id: str | None = None


@dataclass(slots=True)
class ParsedThreadPage:
    title: str
    description: str | None
    posts: list[ParsedPost]
    author_platform_user_id: str | None
    created_at: str


class ThreadAwareWebConnector(CommunityConnector):
    platform = "web"
    user_agent = "AgentPersonaCrawler/0.2"

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        self.validate_target(target)
        self._assert_robots(target.url)
        html = self._fetch_html(target.url, render_mode=target.metadata.get("render_mode", "http"))
        page = self._parse_thread_page(
            html=html,
            target=target,
            observed_at=context.observed_at,
            since=since or target.metadata.get("since"),
            until=target.metadata.get("until"),
        )
        source_url = target.url
        pointer = EvidencePointer(source_url=source_url, fetched_at=context.observed_at)

        parsed_url = urlparse(target.url)
        domain = parsed_url.netloc or target.community_name
        community_id = _slug(target.target_id or domain)
        thread_id = f"web-thread-{_stable_id(target.url)}"

        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="web",
                community_id=community_id,
                community_name=target.community_name,
                community_type="site",
                parent_community_id=None,
                description=page.description,
                member_count=None,
                rules_summary=None,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                evidence_pointer=pointer,
            ),
            ThreadRecord(
                record_type="thread",
                platform="web",
                thread_id=thread_id,
                community_id=community_id,
                title=page.title,
                author_platform_user_id=page.author_platform_user_id,
                created_at=page.created_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "url": target.url,
                    "post_count": len(page.posts),
                    "render_mode": target.metadata.get("render_mode", "http"),
                    "thread_aware": True,
                    "collection_basis": target.collection_basis,
                },
                evidence_pointer=pointer,
            ),
        ]

        seen_users: set[str] = set()
        for post in page.posts:
            author_url = f"{source_url}#author-{post.author_id}"
            if post.author_id not in seen_users:
                account_pointer = EvidencePointer(source_url=author_url, fetched_at=context.observed_at)
                records.append(
                    AccountRecord(
                        record_type="account",
                        platform="web",
                        platform_user_id=post.author_id,
                        username=post.author_name,
                        account_created_at=None,
                        first_observed_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=account_pointer,
                    )
                )
                records.append(
                    ProfileSnapshotRecord(
                        record_type="profile_snapshot",
                        platform="web",
                        platform_user_id=post.author_id,
                        snapshot_at=context.observed_at,
                        crawl_run_id=context.crawl_run_id,
                        fields={"display_name": post.author_name},
                        evidence_pointer=account_pointer,
                    )
                )
                seen_users.add(post.author_id)

            message_url = f"{source_url}#post-{post.message_id}"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="web",
                    message_id=post.message_id,
                    thread_id=thread_id,
                    community_id=community_id,
                    author_platform_user_id=post.author_id,
                    body=post.body,
                    created_at=post.created_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=post.reply_to_message_id,
                    reply_to_user_id=post.reply_to_user_id,
                    metadata={"source_kind": "thread-post"},
                    evidence_pointer=EvidencePointer(
                        source_url=message_url,
                        fetched_at=context.observed_at,
                    ),
                )
            )
            if post.reply_to_user_id:
                records.append(
                    InteractionRecord(
                        record_type="interaction",
                        platform="web",
                        interaction_type="reply",
                        source_user_id=post.author_id,
                        target_user_id=post.reply_to_user_id,
                        message_id=post.message_id,
                        thread_id=thread_id,
                        community_id=community_id,
                        created_at=post.created_at,
                        crawl_run_id=context.crawl_run_id,
                        evidence_pointer=EvidencePointer(
                            source_url=message_url,
                            fetched_at=context.observed_at,
                            derived_from_message_id=post.message_id,
                        ),
                    )
                )
        return records

    def _fetch_html(self, url: str, *, render_mode: str) -> str:
        if render_mode == "playwright":
            return self._render_playwright(url)
        try:
            return self._fetch_http(url)
        except Exception:
            if render_mode != "auto":
                raise
            return self._render_playwright(url)

    def _fetch_http(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=30) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding, errors="replace")

    def _render_playwright(self, url: str) -> str:
        py_cmd = [
            "python3",
            "-c",
            (
                "from playwright.sync_api import sync_playwright\n"
                "import sys\n"
                "u = sys.argv[1]\n"
                "with sync_playwright() as p:\n"
                "    browser = p.chromium.launch(headless=True)\n"
                "    page = browser.new_page()\n"
                "    page.goto(u, wait_until='networkidle', timeout=30000)\n"
                "    print(page.content())\n"
                "    browser.close()\n"
            ),
            url,
        ]
        py_result = subprocess.run(py_cmd, capture_output=True, text=True)
        if py_result.returncode == 0 and py_result.stdout.strip():
            return py_result.stdout

        node_script = """
            const url = process.argv[1];
            import('playwright').then(async ({ chromium }) => {
              const browser = await chromium.launch({ headless: true });
              const page = await browser.newPage();
              await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
              process.stdout.write(await page.content());
              await browser.close();
            }).catch((error) => {
              console.error(String(error));
              process.exit(1);
            });
        """
        node_result = subprocess.run(["node", "-e", node_script, url], capture_output=True, text=True)
        if node_result.returncode != 0 or not node_result.stdout.strip():
            detail = (py_result.stderr or node_result.stderr).strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Playwright fallback unavailable{suffix}")
        return node_result.stdout

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

    def _parse_thread_page(
        self,
        *,
        html: str,
        target: CrawlTarget,
        observed_at: str,
        since: str | None,
        until: str | None,
    ) -> ParsedThreadPage:
        dom = _DomBuilder()
        dom.feed(html)
        posts = self._extract_posts(dom.root, target.url, observed_at)
        filtered_posts = [post for post in posts if _iso_in_window(post.created_at, since, until)]
        if not filtered_posts:
            filtered_posts = posts
        author_id = filtered_posts[0].author_id if filtered_posts else None
        created_at = filtered_posts[0].created_at if filtered_posts else observed_at
        title = dom.meta.get("og:title") or dom.title or target.metadata.get("title") or target.url
        description = dom.meta.get("description") or dom.meta.get("og:description")
        return ParsedThreadPage(
            title=title,
            description=description,
            posts=filtered_posts or self._paragraph_fallback(dom.root, target.url, observed_at),
            author_platform_user_id=author_id,
            created_at=created_at,
        )

    def _extract_posts(self, root: HtmlNode, source_url: str, observed_at: str) -> list[ParsedPost]:
        candidates = [node for node in self._walk(root) if self._is_post_container(node)]
        posts: list[ParsedPost] = []
        seen_ids: set[str] = set()
        post_to_author: dict[str, str] = {}

        for node in candidates:
            body = self._extract_body(node)
            if len(body) < 24:
                continue
            message_id = self._extract_node_id(node, source_url, body)
            if message_id in seen_ids:
                continue
            author_name = self._extract_author(node) or "unknown"
            author_id = f"web-user-{_stable_id(urlparse(source_url).netloc, author_name)}"
            created_at = _normalize_iso(self._extract_timestamp(node), observed_at)
            reply_to_message_id = self._extract_reply_to(node)
            reply_to_user_id = post_to_author.get(reply_to_message_id)
            posts.append(
                ParsedPost(
                    message_id=message_id,
                    author_id=author_id,
                    author_name=author_name,
                    body=body,
                    created_at=created_at,
                    reply_to_message_id=reply_to_message_id,
                    reply_to_user_id=reply_to_user_id,
                )
            )
            post_to_author[message_id] = author_id
            seen_ids.add(message_id)
        return posts

    def _paragraph_fallback(self, root: HtmlNode, source_url: str, observed_at: str) -> list[ParsedPost]:
        domain = urlparse(source_url).netloc or "web"
        author_name = "site-author"
        author_id = f"web-user-{_stable_id(domain, author_name)}"
        posts: list[ParsedPost] = []
        for index, node in enumerate(self._walk(root)):
            if node.tag not in {"p", "li"}:
                continue
            text = node.text.strip()
            if len(text) < 40:
                continue
            posts.append(
                ParsedPost(
                    message_id=f"web-message-{_stable_id(source_url, str(index), text)}",
                    author_id=author_id,
                    author_name=author_name,
                    body=text,
                    created_at=observed_at,
                )
            )
        return posts

    def _walk(self, node: HtmlNode) -> Iterable[HtmlNode]:
        for child in node.children:
            yield child
            yield from self._walk(child)

    def _is_post_container(self, node: HtmlNode) -> bool:
        if node.tag not in {"article", "div", "section", "li"}:
            return False
        attrs = " ".join(node.attrs.values()).lower()
        post_markers = [
            "comment",
            "reply",
            "post",
            "message",
            "topic-post",
            "discussion",
            "forum",
        ]
        return any(marker in attrs for marker in post_markers) or any(
            key in node.attrs for key in ("data-post-id", "data-comment-id", "data-reply-to-id")
        )

    def _extract_body(self, node: HtmlNode) -> str:
        body_nodes = [
            child
            for child in self._walk(node)
            if child.tag in {"p", "blockquote", "li"} and len(child.text.strip()) >= 20
        ]
        if body_nodes:
            return "\n\n".join(child.text.strip() for child in body_nodes)
        return node.text.strip()

    def _extract_author(self, node: HtmlNode) -> str | None:
        for child in self._walk(node):
            attrs = " ".join(child.attrs.values()).lower()
            if any(marker in attrs for marker in ("author", "username", "user", "poster", "display-name")):
                if child.text.strip():
                    return child.text.strip()
            if child.attrs.get("rel") == "author" and child.text.strip():
                return child.text.strip()
        return None

    def _extract_timestamp(self, node: HtmlNode) -> str | None:
        for child in self._walk(node):
            if child.tag == "time":
                return child.attrs.get("datetime") or child.attrs.get("title") or child.text
            for key in ("data-time", "data-created-at", "datetime"):
                if child.attrs.get(key):
                    return child.attrs[key]
        text = node.text
        match = re.search(r"20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", text)
        return match.group(0) if match else None

    def _extract_reply_to(self, node: HtmlNode) -> str | None:
        for key in ("data-reply-to-id", "data-reply-to-post-number"):
            if node.attrs.get(key):
                return node.attrs[key]
        for child in self._walk(node):
            href = child.attrs.get("href", "")
            if "#" in href:
                fragment = href.rsplit("#", 1)[-1]
                if fragment:
                    return fragment.replace("post-", "")
        return None

    def _extract_node_id(self, node: HtmlNode, source_url: str, body: str) -> str:
        for key in ("data-post-id", "data-comment-id", "data-id", "id"):
            value = node.attrs.get(key)
            if value:
                return str(value).replace("post-", "")
        return f"web-message-{_stable_id(source_url, body)}"
