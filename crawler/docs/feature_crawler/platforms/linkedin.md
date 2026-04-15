# LinkedIn Platform

## Scope

First LinkedIn pass supports two HTML-backed transports that map into the same canonical records:

- `official-oidc`: official LinkedIn member identity fetch via access token
- `public-html`: direct profile fetch for public profile pages or local exported HTML
- `session-html`: authenticated profile fetch using an existing LinkedIn cookie/session
- `session-browser`: authenticated Playwright-backed crawl for full profile rendering (JS-rendered pages with lazy-loaded sections) AND first-degree connections in My Network
- `headless-visible-profiles`: authenticated Playwright crawl for detailed sections on already harvested visible connections
- `apify`, `brightdata`, `linkdapi`: vendor-backed crawl modes for richer posts and network data

Current crawl unit:

- community: LinkedIn account/profile
- thread: profile snapshot page
- message: headline, about, experience item, or activity item

## Runtime shape

Code lives under:

- `feature_crawler/crawler/platforms/linkedin/connector_auth.py`
- `feature_crawler/crawler/platforms/linkedin/connector_official.py`
- `feature_crawler/crawler/platforms/linkedin/connector_profile.py`
- `feature_crawler/crawler/platforms/linkedin/connector_vendor.py`
- `feature_crawler/crawler/platforms/linkedin/runner.py`

Shared runtime stays in:

- `feature_crawler/crawler/core/`

## Env

Optional for public mode:

- none

Required for official mode:

- `LINKEDIN_ACCESS_TOKEN`

Required for auth code exchange:

- `LINKEDIN_CLIENT_ID`
- `LINKEDIN_CLIENT_SECRET`

Required for session mode:

- `LINKEDIN_COOKIE`

or:

- `LINKEDIN_SESSION_COOKIE_LI_AT`
- `LINKEDIN_SESSION_COOKIE_JSESSIONID`

Required for headless visible-profile helper:

- local Brave profile with an active LinkedIn session
- `browser-cookie3`
- `playwright`

Vendor mode env:

- `APIFY_TOKEN`
- `BRIGHTDATA_API_KEY`
- `LINKDAPI_API_KEY`
- `LINKEDIN_BRIGHTDATA_PROFILE_DATASET_ID`
- optional actor/dataset env vars for posts/network expansion

## Smoke

Public HTML:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url https://www.linkedin.com/in/example-person/ \
  --output-dir feature_crawler/data \
  --mode public-html \
  --collection-basis consented
```

Official OIDC:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url https://api.linkedin.com/v2/userinfo \
  --output-dir feature_crawler/data \
  --mode official-oidc \
  --collection-basis consented
```

Auth URL bootstrap:

```bash
python3 -m feature_crawler.crawler.cli linkedin-auth-url \
  --client-id "$LINKEDIN_CLIENT_ID" \
  --redirect-uri http://127.0.0.1:8080/callback
```

Code exchange:

```bash
python3 -m feature_crawler.crawler.cli linkedin-exchange-code \
  --code "$LINKEDIN_AUTH_CODE" \
  --redirect-uri http://127.0.0.1:8080/callback
```

Session-backed HTML:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url https://www.linkedin.com/in/example-person/details/experience/ \
  --output-dir feature_crawler/data \
  --mode session-html \
  --collection-basis consented \
  --allow-persona-inference
```

Session-backed browser profile crawl (full JS-rendered profile with lazy-loaded sections):

```bash
python -m crawler.feature_crawler crawl-linkedin \
  --url https://www.linkedin.com/in/username/ \
  --mode session-browser \
  --output-dir data/linkedin/username \
  --collection-basis consented
```

Session-backed browser network crawl:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url https://www.linkedin.com/in/example-person/ \
  --output-dir feature_crawler/data \
  --mode session-browser \
  --collection-basis consented \
  --scope profile,network
```

Headless visible-profile harvest from a saved connections file:

```bash
PYTHONPATH=crawler python crawler/scripts/feature_crawler/linkedin_headless_visible_profiles.py \
  --connections-file crawler/data/linkedin_connections_live_2026-04-13.json \
  --output-file crawler/data/linkedin_visible_profile_details_2026-04-13.json \
  --state-file crawler/data/linkedin_headless_state.json \
  --refresh-auth-state
```

Repo snapshots captured so far:

- `crawler/data/linkedin_connections_live_2026-04-13.json`: 120 visible first-degree connection cards
- `crawler/data/linkedin_visible_profile_details_2026-04-13.json`: 100 headless profile-detail harvest results with section extracts
- keep `crawler/data/linkedin_headless_state.json` local only; it contains reusable authenticated browser state

Vendor crawl with activity + network:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url https://www.linkedin.com/in/example-person/ \
  --output-dir feature_crawler/data \
  --mode linkdapi \
  --collection-basis consented \
  --scope profile,activity,network \
  --activity-limit 25 \
  --comment-limit 64 \
  --network-limit 50 \
  --max-pages 3 \
  --cursor-store feature_crawler/data/cursors
```

Fixture smoke:

```bash
python3 -m feature_crawler.crawler.cli crawl-linkedin \
  --url feature_crawler/tests/fixtures/linkedin_profile_sample.html \
  --output-dir /tmp/feature-crawler-linkedin \
  --mode public-html \
  --collection-basis consented \
  --allow-persona-inference
```

## Policy notes

- public profile crawling still needs explicit source/terms review
- official OIDC mode is limited to consented basic member identity/profile fields
- auth URL helper uses the current OIDC authorization endpoint and scopes `openid profile email` by default
- vendor modes are the intended place for deeper posts/comments/network crawl
- persona inference is only allowed for `owned` or `consented` collection basis
- cross-linking remains off
- session mode assumes a human-provided cookie/session and should use a dedicated account, not a primary identity
- `session-browser` uses Playwright for both full profile page rendering and My Network crawling; use the viewer's own profile URL to avoid target/viewer mismatch

## Session-browser profile flow

When `--mode session-browser` targets an individual profile URL, the crawler uses Playwright to render the full profile page with JavaScript enabled. The flow is:

1. Launch a Playwright browser with the authenticated LinkedIn session (cookies injected from `LINKEDIN_COOKIE` or `LINKEDIN_SESSION_COOKIE_LI_AT` + `LINKEDIN_SESSION_COOKIE_JSESSIONID`).
2. Navigate to the profile URL and wait for the main profile content to load.
3. Scroll the page progressively to trigger lazy-loaded sections (experience, education, skills, activity, etc.).
4. Extract individual items from each section: experience entries, education entries, skills, and activity posts.
5. Emit canonical `message` records for each extracted item, consistent with the other modes.

If browser-based extraction fails for any reason (timeout, element changes, network errors), the crawler falls back to HTTP-based extraction via `session-html` so that a partial result is still returned.

This flow closes the JS-render gap that previously limited `session-html` on profile pages: sections that require JavaScript to appear (e.g., dynamically loaded experience cards, activity feed) are now fully accessible through `session-browser`.

## Gaps

- no full callback web app yet; current auth flow is CLI + local callback helper
- no generic token refresh path yet
- HTML session mode still does not paginate posts/comments/network
- browser session mode depends on Playwright plus a live authenticated LinkedIn viewer session
- headless visible-profile helper depends on exporting cookies from the local Brave session into Playwright storage state
- vendor modes depend on human-provided actor IDs / dataset IDs / API keys
- no delete/tombstone propagation yet
- `session-browser` now fills the JS rendering gap for individual profiles; the previous `session-html` JS-render limitation on profile pages is resolved when using `session-browser` mode
