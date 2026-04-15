from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crawler.feature_crawler.platforms.linkedin.headless_visible_profiles import (
    _is_retryable_profile_error,
    _normalize_same_site,
    _playwright_cookie_from_cookiejar,
    _load_existing_profiles,
    merge_seed_profile_result,
)


class _CookieStub:
    def __init__(
        self,
        *,
        name: str,
        value: str,
        domain: str = ".linkedin.com",
        path: str = "/",
        expires: float = 0,
        secure: bool = True,
        rest: dict[str, str] | None = None,
        http_only: bool = False,
    ) -> None:
        self.name = name
        self.value = value
        self.domain = domain
        self.path = path
        self.expires = expires
        self.secure = secure
        self._rest = rest or {}
        self._http_only = http_only

    def has_nonstandard_attr(self, attr: str) -> bool:
        return attr == "HttpOnly" and self._http_only


class LinkedInHeadlessVisibleProfilesTest(unittest.TestCase):
    def test_normalize_same_site(self) -> None:
        self.assertEqual(_normalize_same_site("strict"), "Strict")
        self.assertEqual(_normalize_same_site("none"), "None")
        self.assertEqual(_normalize_same_site(""), "Lax")
        self.assertEqual(_normalize_same_site(None), "Lax")

    def test_playwright_cookie_from_cookiejar(self) -> None:
        cookie = _CookieStub(
            name="li_at",
            value="token-123",
            expires=123.0,
            rest={"SameSite": "None"},
        )
        result = _playwright_cookie_from_cookiejar(cookie)
        self.assertEqual(result["name"], "li_at")
        self.assertEqual(result["sameSite"], "None")
        self.assertTrue(result["httpOnly"])
        self.assertTrue(result["secure"])

    def test_merge_seed_profile_result_derives_section_fields(self) -> None:
        seed = {
            "name": "Jane Builder",
            "headline": "Founder",
            "connected_on": "April 1, 2026",
        }
        profile = {
            "profile_url": "https://www.linkedin.com/in/jane-builder/",
            "title": "Jane Builder | LinkedIn",
            "name": "",
            "section_map": {
                "Experience": "Experience Example Co",
                "Education": "Education State University",
                "Skills": "Skills Python",
            },
        }
        merged = merge_seed_profile_result(seed, profile)
        self.assertEqual(merged["name"], "Jane Builder")
        self.assertEqual(merged["experience"], "Experience Example Co")
        self.assertEqual(merged["education"], "Education State University")
        self.assertEqual(merged["skills"], "Skills Python")
        self.assertEqual(merged["seed_headline"], "Founder")

    def test_is_retryable_profile_error_matches_nav_flakes(self) -> None:
        self.assertTrue(_is_retryable_profile_error(RuntimeError("Execution context was destroyed")))
        self.assertTrue(_is_retryable_profile_error(RuntimeError("net::ERR_ABORTED")))
        self.assertFalse(_is_retryable_profile_error(RuntimeError("headless LinkedIn auth invalid")))

    def test_load_existing_profiles_indexed_by_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "profiles": [
                    {"profile_url": "https://www.linkedin.com/in/a/", "name": "A"},
                    {"profile_url": "https://www.linkedin.com/in/b/", "name": "B"},
                ]
            }
            path = Path(tmpdir) / "profiles.json"
            path.write_text(json.dumps(payload))
            indexed = _load_existing_profiles(path)
            self.assertEqual(set(indexed), {"https://www.linkedin.com/in/a/", "https://www.linkedin.com/in/b/"})
