from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


WANTED_SECTION_HEADINGS = {
    "About",
    "Experience",
    "Education",
    "Skills",
    "Licenses & certifications",
    "Volunteer experience",
    "Projects",
    "Publications",
    "Honors & awards",
    "Organizations",
    "Courses",
    "Recommendations",
    "Interests",
    "Featured",
    "Languages",
}


def _is_retryable_profile_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        needle in message
        for needle in (
            "Execution context was destroyed",
            "net::ERR_ABORTED",
            "Target page, context or browser has been closed",
        )
    )


def _normalize_same_site(raw_value: str | None) -> str:
    value = (raw_value or "").lower()
    if value == "strict":
        return "Strict"
    if value == "none":
        return "None"
    return "Lax"


def _playwright_cookie_from_cookiejar(cookie: Any) -> dict[str, Any]:
    return {
        "name": cookie.name,
        "value": cookie.value,
        "domain": cookie.domain,
        "path": cookie.path or "/",
        "expires": float(cookie.expires) if cookie.expires else -1,
        "httpOnly": bool(
            cookie.has_nonstandard_attr("HttpOnly")
            or cookie.name in {"li_at", "JSESSIONID", "PLAY_SESSION", "__cf_bm", "li_rm"}
        ),
        "secure": bool(cookie.secure),
        "sameSite": _normalize_same_site(getattr(cookie, "_rest", {}).get("SameSite")),
    }


def build_storage_state_from_brave(*, domain_name: str = "linkedin.com") -> dict[str, Any]:
    try:
        import browser_cookie3
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("install browser-cookie3 to build LinkedIn auth state from Brave") from exc

    cookies = [
        _playwright_cookie_from_cookiejar(cookie)
        for cookie in browser_cookie3.brave(domain_name=domain_name)
    ]
    if not cookies:
        raise RuntimeError(f"no {domain_name} cookies found in Brave")
    return {"cookies": cookies, "origins": []}


def merge_seed_profile_result(seed: dict[str, Any], profile_result: dict[str, Any]) -> dict[str, Any]:
    merged = dict(profile_result)
    merged["seed_name"] = seed.get("name")
    merged["seed_headline"] = seed.get("headline")
    merged["seed_connected_on"] = seed.get("connected_on")
    section_map = merged.get("section_map") or {}
    merged["about"] = section_map.get("About")
    merged["experience"] = section_map.get("Experience")
    merged["education"] = section_map.get("Education")
    merged["skills"] = section_map.get("Skills")
    merged["licenses_and_certifications"] = section_map.get("Licenses & certifications")
    merged["languages"] = section_map.get("Languages")
    if not merged.get("name"):
        merged["name"] = seed.get("name")
    return merged


def _extract_profile_script() -> str:
    return r"""
JSON.stringify((() => {
  const trim = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const wanted = new Set([
    'About','Experience','Education','Skills','Licenses & certifications','Volunteer experience',
    'Projects','Publications','Honors & awards','Organizations','Courses','Recommendations',
    'Interests','Featured','Languages'
  ]);
  const sectionMap = {};
  for (const sec of document.querySelectorAll('main section')) {
    const heading = trim(sec.querySelector('h2,h3')?.textContent || '');
    const text = trim(sec.innerText || '');
    if (!text) continue;
    if (heading && wanted.has(heading) && !(heading in sectionMap)) {
      sectionMap[heading] = text.slice(0, 10000);
    }
  }
  const headings = [...new Set([...document.querySelectorAll('main h2, main h3')].map(x => trim(x.textContent)).filter(Boolean))];
  const title = document.title;
  const h1 = trim(document.querySelector('h1')?.textContent || '');
  return {
    fetched_at: new Date().toISOString(),
    profile_url: location.href,
    title,
    name: h1,
    headings,
    section_map: sectionMap
  };
})())
""".strip()


def _wait_for_profile_ready(page: Any) -> None:
    page.wait_for_load_state("domcontentloaded", timeout=45_000)
    page.locator("main#workspace").wait_for(timeout=45_000)
    page.wait_for_timeout(1_000)


def _scroll_profile_page(page: Any, *, iterations: int = 4, pause_ms: int = 1200) -> None:
    container = page.locator("main#workspace")
    for _ in range(iterations):
        container.evaluate(
            """
            (el) => {
              if (el) {
                el.scrollTo(0, el.scrollHeight);
              }
            }
            """
        )
        page.wait_for_timeout(pause_ms)


def _load_existing_profiles(output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}
    payload = json.loads(output_path.read_text())
    results: dict[str, dict[str, Any]] = {}
    for item in payload.get("profiles", []):
        profile_url = item.get("profile_url")
        if profile_url:
            results[profile_url] = item
    return results


def _write_checkpoint(
    *,
    output_path: Path,
    input_connections_file: Path,
    profiles: list[dict[str, Any]],
) -> None:
    payload = {
        "source": "linkedin-authenticated-visible-profile-harvest",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_connections_file": str(input_connections_file),
        "profiles": profiles,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def _extract_profile_result(
    *,
    page: Any,
    profile_url: str,
    extract_script: str,
    settle_ms: int,
    scroll_iterations: int,
    scroll_pause_ms: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            page.goto(profile_url, wait_until="domcontentloaded", timeout=45_000)
            _wait_for_profile_ready(page)
            page.wait_for_timeout(settle_ms)
            _scroll_profile_page(page, iterations=scroll_iterations, pause_ms=scroll_pause_ms)
            _wait_for_profile_ready(page)
            raw = page.evaluate(extract_script)
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - runtime timing behavior
            last_error = exc
            if not _is_retryable_profile_error(exc):
                raise
            page.wait_for_timeout(1_500)
    if last_error is not None:
        raise last_error
    raise RuntimeError("profile extraction failed without a captured exception")


def run_visible_profile_harvest(
    *,
    connections_path: Path,
    output_path: Path,
    state_path: Path,
    max_profiles: int | None = None,
    checkpoint_every: int = 5,
    settle_ms: int = 4500,
    scroll_iterations: int = 4,
    scroll_pause_ms: int = 1200,
) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("install playwright to run LinkedIn headless harvest") from exc

    input_connections = json.loads(connections_path.read_text())
    seeds = input_connections["connections"]
    if max_profiles is not None:
        seeds = seeds[:max_profiles]

    existing = _load_existing_profiles(output_path)
    completed_urls = set(existing)
    results = list(existing.values())

    if not state_path.exists():
        state_path.write_text(json.dumps(build_storage_state_from_brave()))

    extract_script = _extract_profile_script()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(state_path),
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1440, "height": 1200},
        )
        auth_page = context.new_page()
        auth_page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=45_000)
        auth_page.wait_for_timeout(2_000)
        if "linkedin.com/feed/" not in auth_page.url:
            raise RuntimeError(f"headless LinkedIn auth invalid; landed at {auth_page.url}")
        auth_page.close()

        for index, seed in enumerate(seeds, start=1):
            profile_url = seed["profile_url"]
            if profile_url in completed_urls:
                print(f"[{index}/{len(seeds)}] skip {seed.get('name') or profile_url}", flush=True)
                continue

            result: dict[str, Any] | None = None
            last_error: Exception | None = None
            for _ in range(3):
                page = context.new_page()
                try:
                    result = merge_seed_profile_result(
                        seed,
                        _extract_profile_result(
                            page=page,
                            profile_url=profile_url,
                            extract_script=extract_script,
                            settle_ms=settle_ms,
                            scroll_iterations=scroll_iterations,
                            scroll_pause_ms=scroll_pause_ms,
                        ),
                    )
                    break
                except Exception as exc:  # pragma: no cover - runtime timing behavior
                    last_error = exc
                    if not _is_retryable_profile_error(exc):
                        break
                    page.wait_for_timeout(1_500)
                finally:
                    page.close()

            if result is None:
                result = {
                    "profile_url": profile_url,
                    "name": seed.get("name"),
                    "seed_name": seed.get("name"),
                    "seed_headline": seed.get("headline"),
                    "seed_connected_on": seed.get("connected_on"),
                    "error": str(last_error) if last_error is not None else "unknown profile extraction failure",
                }
                print(
                    f"[{index}/{len(seeds)}] error {seed.get('name') or profile_url}: {result['error']}",
                    flush=True,
                )
            results.append(result)
            completed_urls.add(profile_url)

            if len(results) % checkpoint_every == 0 or len(completed_urls) == len(seeds):
                _write_checkpoint(
                    output_path=output_path,
                    input_connections_file=connections_path,
                    profiles=results,
                )
            if "error" not in result:
                print(f"[{index}/{len(seeds)}] ok {result.get('name') or seed.get('name') or profile_url}", flush=True)

        browser.close()

    _write_checkpoint(output_path=output_path, input_connections_file=connections_path, profiles=results)
    return len(results)
