from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from typing import Iterable
from urllib import robotparser
from urllib.parse import unquote, urlparse
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


def _stable_id(*parts: str) -> str:
    return sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"


def _clean_space(value: str | None) -> str | None:
    if value is None:
        return None
    clean = " ".join(value.split()).strip()
    return clean or None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = _clean_space(item)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _selector_blob(attrs: dict[str, str]) -> str:
    keys = ("id", "class", "data-section", "data-test-id", "aria-label", "data-view-name")
    return " ".join(attrs.get(key, "") for key in keys).lower()


def _looks_like_linkedin_profile_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return True
    return parsed.netloc.endswith("linkedin.com") and "/in/" in parsed.path


def _public_identifier(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/in/([^/?#]+)/?", parsed.path)
    if match:
        return _slug(unquote(match.group(1)))
    return _slug(fallback)


def _cookie_from_env(cookie_env: str, li_at_env: str, jsessionid_env: str) -> str | None:
    li_at = _clean_space(os.environ.get(li_at_env))
    jsessionid = _clean_space(os.environ.get(jsessionid_env))
    cookie_parts: list[str] = []
    if li_at:
        cookie_parts.append(f"li_at={li_at}")
    if jsessionid:
        cookie_parts.append(f"JSESSIONID={jsessionid}")
    if cookie_parts:
        return "; ".join(cookie_parts)
    raw_cookie = _clean_space(os.environ.get(cookie_env))
    if raw_cookie:
        return raw_cookie
    return None


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
        self.meta: dict[str, str] = {}
        self.title = ""
        self._in_title = False
        self._capture_json_ld = False
        self._json_ld_buffer: list[str] = []
        self.json_ld_blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        if tag == "meta":
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            content = (attr_map.get("content") or "").strip()
            if name and content:
                self.meta[name] = content
            return
        if tag == "script" and attr_map.get("type", "").lower() == "application/ld+json":
            self._capture_json_ld = True
            self._json_ld_buffer.clear()
            return
        if tag == "title":
            self._in_title = True
        node = HtmlNode(tag=tag, attrs=attr_map)
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._capture_json_ld:
            block = "".join(self._json_ld_buffer).strip()
            if block:
                self.json_ld_blocks.append(block)
            self._capture_json_ld = False
            self._json_ld_buffer.clear()
            return
        if tag == "title":
            self._in_title = False
        if len(self.stack) > 1:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self._capture_json_ld:
            self._json_ld_buffer.append(data)
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title and not self.title:
            self.title = clean
        self.stack[-1].add_text(clean)


@dataclass(slots=True)
class ParsedLinkedInProfile:
    source_mode: str
    source_url: str
    public_identifier: str
    name: str
    headline: str | None
    about: str | None
    location: str | None
    image_url: str | None
    profile_url: str
    experience: list[str]
    activity: list[str]


class LinkedInProfileConnector(CommunityConnector):
    platform = "linkedin"
    user_agent = "AgentPersonaCrawler/0.3"

    def __init__(self, *, mode: str = "public-html", session_cookie: str | None = None) -> None:
        if mode not in {"public-html", "session-html"}:
            raise ValueError(f"unsupported linkedin mode: {mode}")
        if mode == "session-html" and not session_cookie:
            raise ValueError("session-html mode requires a LinkedIn session cookie")
        self.mode = mode
        self.session_cookie = session_cookie

    @classmethod
    def from_env(
        cls,
        *,
        mode: str,
        cookie_env: str = "LINKEDIN_COOKIE",
        li_at_env: str = "LINKEDIN_SESSION_COOKIE_LI_AT",
        jsessionid_env: str = "LINKEDIN_SESSION_COOKIE_JSESSIONID",
    ) -> "LinkedInProfileConnector":
        session_cookie = None
        if mode == "session-html":
            session_cookie = _cookie_from_env(cookie_env, li_at_env, jsessionid_env)
        return cls(mode=mode, session_cookie=session_cookie)

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        del since
        self.validate_target(target)
        if not _looks_like_linkedin_profile_url(target.url):
            raise ValueError(f"linkedin connector expects a profile URL or file URI, got: {target.url}")
        self._assert_robots(target.url)
        html = self._fetch_html(target.url)
        profile = self._parse_profile(html=html, source_url=target.url)
        pointer = EvidencePointer(source_url=target.url, fetched_at=context.observed_at)
        community_id = target.target_id or profile.public_identifier
        username = profile.name or profile.public_identifier
        platform_user_id = f"linkedin-user-{profile.public_identifier}"
        thread_id = f"linkedin-profile-{profile.public_identifier}"

        profile_fields = {
            "display_name": profile.name,
            "headline": profile.headline,
            "about": profile.about,
            "location": profile.location,
            "image_url": profile.image_url,
            "profile_url": profile.profile_url,
            "public_identifier": profile.public_identifier,
            "experience": profile.experience,
            "activity": profile.activity,
            "source_mode": profile.source_mode,
        }
        records: list[Record] = [
            CommunityRecord(
                record_type="community",
                platform="linkedin",
                community_id=community_id,
                community_name=target.community_name or username,
                community_type="account",
                parent_community_id=None,
                description=profile.headline or profile.about,
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
                username=username,
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
                fields=profile_fields,
                evidence_pointer=pointer,
            ),
            ThreadRecord(
                record_type="thread",
                platform="linkedin",
                thread_id=thread_id,
                community_id=community_id,
                title=f"LinkedIn profile: {username}",
                author_platform_user_id=platform_user_id,
                created_at=context.observed_at,
                observed_at=context.observed_at,
                crawl_run_id=context.crawl_run_id,
                metadata={
                    "source_mode": profile.source_mode,
                    "profile_url": profile.profile_url,
                    "public_identifier": profile.public_identifier,
                    "experience_count": len(profile.experience),
                    "activity_count": len(profile.activity),
                },
                evidence_pointer=pointer,
            ),
        ]

        ordinal = 0
        for source_kind, body in self._message_bodies(profile):
            ordinal += 1
            message_id = f"linkedin-message-{profile.public_identifier}-{ordinal}"
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="linkedin",
                    message_id=message_id,
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

    def _message_bodies(self, profile: ParsedLinkedInProfile) -> list[tuple[str, str]]:
        messages: list[tuple[str, str]] = []
        if profile.headline:
            messages.append(("headline", profile.headline))
        if profile.about:
            messages.append(("about", profile.about))
        messages.extend(("experience", item) for item in profile.experience)
        messages.extend(("activity", item) for item in profile.activity)
        return messages

    def _fetch_html(self, source_url: str) -> str:
        parsed = urlparse(source_url)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).read_text(encoding="utf-8")

        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}
        if self.session_cookie:
            headers["Cookie"] = self.session_cookie
        request = Request(source_url, headers=headers)
        with urlopen(request, timeout=30) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding, errors="replace")

    def _assert_robots(self, source_url: str) -> None:
        parsed = urlparse(source_url)
        if parsed.scheme == "file" or self.mode == "session-html":
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
        if not parser.can_fetch(self.user_agent, source_url):
            raise PermissionError(f"robots.txt blocks crawling for {source_url}")

    def _parse_profile(self, *, html: str, source_url: str) -> ParsedLinkedInProfile:
        dom = _DomBuilder()
        dom.feed(html)
        person = self._person_from_json_ld(dom.json_ld_blocks)
        name = (
            person.get("name")
            or self._first_tag_text(dom.root, "h1")
            or dom.meta.get("og:title")
            or dom.title
            or "unknown"
        )
        headline = (
            person.get("description")
            or self._first_matching_text(dom.root, ("headline", "text-body-medium", "top-card"))
            or dom.meta.get("description")
        )
        about = self._section_text(dom.root, "about")
        location = self._first_matching_text(dom.root, ("location", "top-card-location"))
        image_url = person.get("image") or dom.meta.get("og:image")
        profile_url = person.get("url") or dom.meta.get("og:url") or source_url
        public_identifier = _public_identifier(profile_url, fallback=name)
        experience = self._section_items(dom.root, "experience")
        activity = self._section_items(dom.root, "activity")
        if not about and headline and dom.meta.get("description") and dom.meta.get("description") != headline:
            about = dom.meta.get("description")
        return ParsedLinkedInProfile(
            source_mode=self.mode,
            source_url=source_url,
            public_identifier=public_identifier,
            name=_clean_space(name) or public_identifier,
            headline=_clean_space(headline),
            about=_clean_space(about),
            location=_clean_space(location),
            image_url=_clean_space(image_url),
            profile_url=profile_url,
            experience=experience,
            activity=activity,
        )

    def _person_from_json_ld(self, blocks: list[str]) -> dict[str, str]:
        for block in blocks:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            for node in self._iter_json_nodes(payload):
                node_type = str(node.get("@type") or "").lower()
                if node_type != "person":
                    continue
                return {
                    "name": _clean_space(str(node.get("name") or "")) or "",
                    "description": _clean_space(str(node.get("description") or "")) or "",
                    "image": _clean_space(str(node.get("image") or "")) or "",
                    "url": _clean_space(str(node.get("url") or "")) or "",
                }
        return {}

    def _iter_json_nodes(self, payload: object) -> Iterable[dict[str, object]]:
        if isinstance(payload, dict):
            yield payload
            graph = payload.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    if isinstance(item, dict):
                        yield from self._iter_json_nodes(item)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield from self._iter_json_nodes(item)

    def _walk(self, root: HtmlNode) -> Iterable[HtmlNode]:
        yield root
        for child in root.children:
            yield from self._walk(child)

    def _first_tag_text(self, root: HtmlNode, tag: str) -> str | None:
        for node in self._walk(root):
            if node.tag == tag and node.text:
                return node.text
        return None

    def _first_matching_text(self, root: HtmlNode, markers: tuple[str, ...]) -> str | None:
        for node in self._walk(root):
            blob = _selector_blob(node.attrs)
            if any(marker in blob for marker in markers):
                return node.text or None
        return None

    def _section_text(self, root: HtmlNode, keyword: str) -> str | None:
        section = self._find_section(root, keyword)
        if section is None:
            return None
        texts = self._candidate_texts(section, minimum_words=8)
        if not texts:
            return None
        return texts[0]

    def _section_items(self, root: HtmlNode, keyword: str) -> list[str]:
        section = self._find_section(root, keyword)
        if section is None:
            return []
        items: list[str] = []
        for node in self._walk(section):
            if node.tag in {"li", "article"} and len(node.text.split()) >= 4:
                items.append(node.text)
        if not items:
            items = self._candidate_texts(section, minimum_words=4)
        return _dedupe(items)

    def _find_section(self, root: HtmlNode, keyword: str) -> HtmlNode | None:
        keyword_lower = keyword.lower()
        for node in self._walk(root):
            if node.tag not in {"section", "div"}:
                continue
            blob = _selector_blob(node.attrs)
            if keyword_lower in blob:
                return node
            header_texts = [child.text.lower() for child in node.children if child.tag in {"h2", "h3", "span"}]
            if any(keyword_lower in header for header in header_texts):
                return node
        return None

    def _candidate_texts(self, section: HtmlNode, *, minimum_words: int) -> list[str]:
        texts: list[str] = []
        for node in self._walk(section):
            if node.tag in {"p", "span", "div", "li", "article"}:
                text = _clean_space(node.text)
                if text and len(text.split()) >= minimum_words:
                    texts.append(text)
        return _dedupe(texts)
