from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
import logging
import re
from typing import Any, Iterable

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
from .connector_profile import LinkedInProfileConnector, _clean_space, _cookie_from_env
from .headless_visible_profiles import (
    _is_retryable_profile_error,
    _scroll_profile_page,
    _wait_for_profile_ready,
)

logger = logging.getLogger(__name__)


def _stable_id(*parts: str) -> str:
    return sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def _public_identifier_from_url(url: str, fallback: str) -> str:
    match = re.search(r"/in/([^/?#]+)/?", url)
    if match:
        return match.group(1)
    return fallback


def _parse_connection_count(text: str | None) -> int | None:
    clean = _clean_space(text)
    if not clean:
        return None
    match = re.search(r"(\d[\d,]*)\s+connections\b", clean, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


@dataclass(slots=True)
class BrowserProfilePayload:
    """Result of Playwright-based profile page extraction."""

    profile_url: str
    name: str
    headline: str | None
    about: str | None
    location: str | None
    experience: list[str]
    activity: list[str]
    education: list[str]
    skills: list[str]
    section_map: dict[str, str]


def _profile_extraction_script() -> str:
    """JS that extracts profile sections, individual items, and activity posts."""
    return r"""
JSON.stringify((() => {
  const trim = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const wanted = new Set([
    'About','Experience','Education','Skills','Licenses & certifications',
    'Volunteer experience','Projects','Publications','Honors & awards',
    'Organizations','Courses','Recommendations','Interests','Featured','Languages'
  ]);
  const sectionMap = {};
  const sectionItems = {};

  for (const sec of document.querySelectorAll('main section')) {
    const heading = trim(sec.querySelector('h2,h3')?.textContent || '');
    const text = trim(sec.innerText || '');
    if (!text) continue;
    if (heading && wanted.has(heading) && !(heading in sectionMap)) {
      sectionMap[heading] = text.slice(0, 10000);
      const items = [...sec.querySelectorAll('li, article')]
        .map(el => trim(el.innerText))
        .filter(t => t && t.split(' ').length >= 4);
      const seen = new Set();
      const unique = items.filter(t => { if (seen.has(t)) return false; seen.add(t); return true; });
      if (unique.length > 0) sectionItems[heading] = unique.slice(0, 50);
    }
  }

  const activityItems = [];
  for (const sec of document.querySelectorAll('main section')) {
    const heading = trim(sec.querySelector('h2,h3')?.textContent || '');
    if (heading !== 'Activity') continue;
    const posts = sec.querySelectorAll('li, article, [data-urn]');
    const seen = new Set();
    for (const post of posts) {
      const t = trim(post.innerText);
      if (t && t.split(' ').length >= 6 && !seen.has(t)) {
        seen.add(t);
        activityItems.push(t.slice(0, 5000));
      }
    }
    break;
  }

  const headlineText = trim(
    (document.querySelector('.text-body-medium') ||
     document.querySelector('[data-test-id="headline"]') ||
     document.querySelector('.top-card-layout__headline'))?.textContent || ''
  );
  const locationText = trim(
    (document.querySelector('.text-body-small[class*="location"]') ||
     document.querySelector('[class*="top-card"][class*="location"]'))?.textContent || ''
  );

  return {
    fetched_at: new Date().toISOString(),
    profile_url: window.location.href,
    title: document.title,
    name: trim(document.querySelector('h1')?.textContent || ''),
    headline: headlineText,
    location: locationText,
    section_map: sectionMap,
    section_items: sectionItems,
    activity_items: activityItems,
  };
})())
""".strip()


def _parse_browser_profile(raw: dict[str, Any]) -> BrowserProfilePayload:
    """Convert raw JS extraction result into a structured payload."""
    section_map = raw.get("section_map") or {}
    section_items = raw.get("section_items") or {}
    activity_items = raw.get("activity_items") or []

    experience = section_items.get("Experience", [])
    if not experience and section_map.get("Experience"):
        experience = _split_section_blob(section_map["Experience"])

    education = section_items.get("Education", [])
    if not education and section_map.get("Education"):
        education = _split_section_blob(section_map["Education"])

    skills = section_items.get("Skills", [])
    if not skills and section_map.get("Skills"):
        skills = _split_section_blob(section_map["Skills"])

    activity = activity_items or section_items.get("Activity", [])
    if not activity and section_map.get("Activity"):
        activity = _split_section_blob(section_map["Activity"])

    about = section_map.get("About")
    headline = _clean_space(raw.get("headline"))
    if about and headline and about.startswith(headline):
        pass  # keep about as-is even if it starts with the headline

    return BrowserProfilePayload(
        profile_url=raw.get("profile_url") or "",
        name=_clean_space(raw.get("name")) or "unknown",
        headline=headline,
        about=_clean_space(about),
        location=_clean_space(raw.get("location")),
        experience=experience,
        activity=activity,
        education=education,
        skills=skills,
        section_map=section_map,
    )


def _split_section_blob(blob: str) -> list[str]:
    """Split a raw section text blob into individual items heuristically."""
    lines = [line.strip() for line in blob.split("\n") if line.strip()]
    if len(lines) <= 1:
        return [blob.strip()] if blob.strip() else []
    items: list[str] = []
    current: list[str] = []
    for line in lines:
        if len(line.split()) >= 3 and current and (
            line[0].isupper() or line.startswith("·")
        ):
            combined = " ".join(current)
            if len(combined.split()) >= 4:
                items.append(combined)
            current = [line]
        else:
            current.append(line)
    if current:
        combined = " ".join(current)
        if len(combined.split()) >= 4:
            items.append(combined)
    return items if items else ([blob.strip()] if len(blob.split()) >= 4 else [])


@dataclass(slots=True)
class BrowserConnectionsPayload:
    source_url: str
    connection_count: int | None
    connections: list[dict[str, Any]]
    viewer_public_identifier: str | None = None
    viewer_name: str | None = None


class LinkedInBrowserSessionClient:
    def __init__(self, session_cookie: str, *, timeout_ms: int = 60_000, settle_ms: int = 8_000) -> None:
        self.session_cookie = session_cookie
        self.timeout_ms = timeout_ms
        self.settle_ms = settle_ms

    def fetch_connections(self) -> BrowserConnectionsPayload:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - exercised only in runtime envs without playwright
            raise RuntimeError("session-browser mode requires playwright to be installed") from exc

        li_at, jsessionid = self._cookie_values()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1440, "height": 1200},
            )
            context.add_cookies(
                [
                    {
                        "name": "li_at",
                        "value": li_at,
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": True,
                        "secure": True,
                        "sameSite": "None",
                    },
                    {
                        "name": "JSESSIONID",
                        "value": jsessionid,
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": False,
                        "secure": True,
                        "sameSite": "Lax",
                    },
                    {
                        "name": "liap",
                        "value": "true",
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": False,
                        "secure": True,
                        "sameSite": "Lax",
                    },
                ]
            )
            page = context.new_page()
            page.goto(
                "https://www.linkedin.com/mynetwork/invite-connect/connections/",
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
            page.wait_for_timeout(self.settle_ms)
            self._scroll_connections(page)
            body_text = page.locator("body").inner_text(timeout=self.timeout_ms)
            connection_count = _parse_connection_count(body_text)
            connections = page.evaluate(
                """
                () => {
                  const root =
                    document.querySelector('[componentkey="ConnectionsPage_ConnectionsList"]') ||
                    document.querySelector('[data-sdui-screen="com.linkedin.sdui.flagshipnav.mynetwork.Connections"]');
                  if (!root) return [];
                  const anchors = Array.from(root.querySelectorAll('a[href*="/in/"]'));
                  const seen = new Set();
                  return anchors
                    .map((anchor) => {
                      const href = anchor.href || anchor.getAttribute('href') || '';
                      if (!href || seen.has(href)) return null;
                      seen.add(href);
                      const card = anchor.closest('li, article, div') || anchor;
                      const textNodes = Array.from(card.querySelectorAll('span, p, div'))
                        .map((node) => (node.innerText || '').trim())
                        .filter(Boolean);
                      const name = (anchor.innerText || '').trim() || textNodes[0] || '';
                      const headline = textNodes.find((item) => item && item !== name) || '';
                      return { profile_url: href, name, headline };
                    })
                    .filter(Boolean);
                }
                """
            )
            browser.close()
        clean_connections = [item for item in connections if isinstance(item, dict)]
        viewer_public_identifier = clean_connections[0].get("viewer_public_identifier") if clean_connections else None
        return BrowserConnectionsPayload(
            source_url="https://www.linkedin.com/mynetwork/invite-connect/connections/",
            connection_count=connection_count,
            connections=clean_connections,
            viewer_public_identifier=_clean_space(viewer_public_identifier),
        )

    def _cookie_values(self) -> tuple[str, str]:
        li_at_match = re.search(r"(?:^|;\s*)li_at=([^;]+)", self.session_cookie)
        jsessionid_match = re.search(r"(?:^|;\s*)JSESSIONID=([^;]+)", self.session_cookie)
        if not li_at_match or not jsessionid_match:
            raise RuntimeError("session-browser mode requires li_at and JSESSIONID cookies")
        return li_at_match.group(1), jsessionid_match.group(1)

    def fetch_profile(self, profile_url: str) -> BrowserProfilePayload:
        """Use Playwright to load a profile page with full JS rendering and extract data."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("session-browser mode requires playwright to be installed") from exc

        li_at, jsessionid = self._cookie_values()
        extract_script = _profile_extraction_script()

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1440, "height": 1200},
            )
            context.add_cookies(
                [
                    {
                        "name": "li_at",
                        "value": li_at,
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": True,
                        "secure": True,
                        "sameSite": "None",
                    },
                    {
                        "name": "JSESSIONID",
                        "value": jsessionid,
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": False,
                        "secure": True,
                        "sameSite": "Lax",
                    },
                    {
                        "name": "liap",
                        "value": "true",
                        "domain": ".linkedin.com",
                        "path": "/",
                        "httpOnly": False,
                        "secure": True,
                        "sameSite": "Lax",
                    },
                ]
            )
            page = context.new_page()
            last_error: Exception | None = None
            raw: dict[str, Any] | None = None
            for _ in range(3):
                try:
                    page.goto(profile_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    _wait_for_profile_ready(page)
                    page.wait_for_timeout(self.settle_ms)
                    _scroll_profile_page(page, iterations=4, pause_ms=1200)
                    _wait_for_profile_ready(page)
                    result = page.evaluate(extract_script)
                    raw = json.loads(result)
                    break
                except Exception as exc:
                    last_error = exc
                    if not _is_retryable_profile_error(exc):
                        break
                    page.wait_for_timeout(1_500)
            browser.close()

        if raw is None:
            raise RuntimeError(
                f"browser profile extraction failed for {profile_url}: {last_error}"
            ) from last_error
        return _parse_browser_profile(raw)

    def _scroll_connections(self, page: Any) -> None:
        for _ in range(3):
            page.mouse.wheel(0, 2400)
            page.wait_for_timeout(1_200)


class LinkedInBrowserConnector(CommunityConnector):
    platform = "linkedin"

    def __init__(
        self,
        *,
        profile_connector: LinkedInProfileConnector,
        browser_client: LinkedInBrowserSessionClient,
    ) -> None:
        self.profile_connector = profile_connector
        self.browser_client = browser_client

    @classmethod
    def from_env(
        cls,
        *,
        cookie_env: str = "LINKEDIN_COOKIE",
        li_at_env: str = "LINKEDIN_SESSION_COOKIE_LI_AT",
        jsessionid_env: str = "LINKEDIN_SESSION_COOKIE_JSESSIONID",
    ) -> "LinkedInBrowserConnector":
        session_cookie = _cookie_from_env(cookie_env, li_at_env, jsessionid_env)
        if not session_cookie:
            raise ValueError("session-browser mode requires a LinkedIn session cookie")
        return cls(
            profile_connector=LinkedInProfileConnector(mode="session-html", session_cookie=session_cookie),
            browser_client=LinkedInBrowserSessionClient(session_cookie=session_cookie),
        )

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        include_network = bool(target.metadata.get("include_network"))
        records: list[Record] = []

        # Always use browser rendering for the profile page (the whole point of session-browser)
        try:
            profile_payload = self.browser_client.fetch_profile(target.url)
            records = self._browser_profile_records(
                profile_payload, target=target, context=context,
            )
        except Exception as exc:
            logger.warning(
                "linkedin session-browser profile fetch failed; falling back to HTTP: %s",
                exc,
            )
            try:
                records = list(self.profile_connector.fetch(target=target, context=context, since=since))
            except Exception as fallback_exc:
                logger.warning(
                    "linkedin session-html fallback also failed: %s",
                    fallback_exc,
                )
                records = []

        if include_network:
            payload = self.browser_client.fetch_connections()
            profile_snapshot = next(
                (
                    record
                    for record in records
                    if isinstance(record, ProfileSnapshotRecord)
                ),
                None,
            )
            source_public_identifier = (
                profile_snapshot.fields.get("public_identifier")
                if isinstance(profile_snapshot, ProfileSnapshotRecord)
                else target.target_id
            ) or target.target_id or _public_identifier_from_url(target.url, "linkedin-viewer")
            source_user_id = f"linkedin-user-{source_public_identifier}"
            community_id = target.target_id or source_public_identifier
            records.extend(
                self._network_records(
                    payload.connections,
                    connection_count=payload.connection_count,
                    community_id=community_id,
                    source_user_id=source_user_id,
                    context=context,
                    source_url=payload.source_url,
                )
            )
        return records

    def _browser_profile_records(
        self,
        profile: BrowserProfilePayload,
        *,
        target: CrawlTarget,
        context: CrawlContext,
    ) -> list[Record]:
        """Convert browser-extracted profile data into canonical Records."""
        public_identifier = _public_identifier_from_url(
            profile.profile_url, profile.name,
        )
        community_id = target.target_id or public_identifier
        platform_user_id = f"linkedin-user-{public_identifier}"
        thread_id = f"linkedin-profile-{public_identifier}"
        username = profile.name or public_identifier
        pointer = EvidencePointer(source_url=target.url, fetched_at=context.observed_at)

        profile_fields: dict[str, Any] = {
            "display_name": profile.name,
            "headline": profile.headline,
            "about": profile.about,
            "location": profile.location,
            "profile_url": profile.profile_url,
            "public_identifier": public_identifier,
            "experience": profile.experience,
            "activity": profile.activity,
            "education": profile.education,
            "skills": profile.skills,
            "source_mode": "session-browser",
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
                    "source_mode": "session-browser",
                    "profile_url": profile.profile_url,
                    "public_identifier": public_identifier,
                    "experience_count": len(profile.experience),
                    "activity_count": len(profile.activity),
                },
                evidence_pointer=pointer,
            ),
        ]

        ordinal = 0
        message_bodies: list[tuple[str, str]] = []
        if profile.headline:
            message_bodies.append(("headline", profile.headline))
        if profile.about:
            message_bodies.append(("about", profile.about))
        message_bodies.extend(("experience", item) for item in profile.experience)
        message_bodies.extend(("activity", item) for item in profile.activity)
        message_bodies.extend(("education", item) for item in profile.education)
        message_bodies.extend(("skills", item) for item in profile.skills)

        for source_kind, body in message_bodies:
            ordinal += 1
            records.append(
                MessageRecord(
                    record_type="message",
                    platform="linkedin",
                    message_id=f"linkedin-message-{public_identifier}-{ordinal}",
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

    def _network_records(
        self,
        connections: list[dict[str, Any]],
        *,
        connection_count: int | None,
        community_id: str,
        source_user_id: str,
        context: CrawlContext,
        source_url: str,
    ) -> list[Record]:
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
                metadata={
                    "source_mode": "session-browser",
                    "scope": "network",
                    "connection_count": connection_count,
                },
                evidence_pointer=pointer,
            )
        ]
        for index, connection in enumerate(connections, start=1):
            name = _clean_space(connection.get("name")) or "unknown"
            profile_url = _clean_space(connection.get("profile_url")) or source_url
            public_identifier = _clean_space(connection.get("public_identifier")) or _public_identifier_from_url(
                profile_url,
                _stable_id(name, str(index)),
            )
            target_user_id = f"linkedin-user-{public_identifier}"
            headline = _clean_space(connection.get("headline"))
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
                    fields={
                        "display_name": name,
                        "headline": headline,
                        "public_identifier": public_identifier,
                        "profile_url": profile_url,
                        "source_mode": "session-browser",
                    },
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
                    body=headline or name,
                    created_at=context.observed_at,
                    observed_at=context.observed_at,
                    crawl_run_id=context.crawl_run_id,
                    reply_to_message_id=None,
                    reply_to_user_id=target_user_id,
                    metadata={"source_kind": "connection", "ordinal": index, "profile_url": profile_url},
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
                    evidence_pointer=EvidencePointer(
                        source_url=source_url,
                        fetched_at=context.observed_at,
                        derived_from_message_id=message_id,
                    ),
                )
            )
        return records
