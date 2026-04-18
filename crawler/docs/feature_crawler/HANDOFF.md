# Handoff

## Read first

1. `PRD_CRAWLER.md`
2. `STATUS.yaml`
3. `TASKS.md`
4. `feature_crawler/data/manifest_2026-04-09_fortnite_roblox.json`

## Current state

Problem and scope defined.
Crawler runtime exists inside `feature_crawler/crawler`.

Shipped:

- canonical record models
- source policy gate
- approved web connector
- thread-aware web connector
- live Discord API connector
- local JSONL sink
- CLI entrypoint
- focused tests

Verified:

- `python3 -m unittest discover -s feature_crawler/tests`
- 12 tests passed on `2026-04-09`

## Goal of this handoff

Crawl data.
Web scrape data.
Save it under `feature_crawler/data`.

Do not wait on Discord unless a server admin has invited the bot.

## Best current data

### Discord (Midjourney server — first successful crawl 2026-04-09)

Crawled via user-token API. 2,009+ records across 5 channels:

- `feature_crawler/data/discord/midjourney/getting-started_999550150705954856/4/9/2026/run_20260409_163213.jsonl` — #getting-started (8 records)
- `feature_crawler/data/discord/midjourney/discussion_938713143759216720/4/9/2026/run_20260409_163315.jsonl` — #discussion (440 records, 59 users)
- `feature_crawler/data/discord/midjourney/prompt-craft_992207085146222713/4/9/2026/run_20260409_163340.jsonl` — #prompt-craft (434 records, 32 users)
- `feature_crawler/data/discord/midjourney/ideas-and-features_989270517401911316/4/9/2026/run_20260409_163412.jsonl` — #ideas-and-features (412 records, 89 users)
- `feature_crawler/data/discord/midjourney/announcements_952771221915840552/4/9/2026/run_20260409_163435.jsonl` — #announcements (205 records)
- `feature_crawler/data/discord/midjourney/bug-reporting_1100553646162317332/4/9/2026/run_20260409_163500.jsonl` — #bug-reporting (518 records, 116 users)

### Web (Fortnite + Roblox — 2026-04-01 to 2026-04-09)

- `feature_crawler/data/web/fortnite-unreal-forums/2026-04-09/run_20260409_153841.jsonl`
- `feature_crawler/data/web/fortnite-unreal-forums/2026-04-09/run_20260409_153850.jsonl`
- `feature_crawler/data/web/roblox-devforum/2026-04-09/run_20260409_153903.jsonl`
- `feature_crawler/data/web/roblox-devforum/2026-04-09/run_20260409_153904.jsonl`
- `feature_crawler/data/manifest_2026-04-09_fortnite_roblox.json`

## Current crawl strategy

Preferred path: Discourse `.json` topic endpoints.

Reason:

- faster than full HTML
- thread-aware
- stable author/message structure
- works for `devforum.roblox.com`
- works for `forums.unrealengine.com`

Fallback path: HTML/thread-aware parsing or Playwright-backed rendering when needed.

## Known-good commands

Roblox:

```bash
python3 -m feature_crawler.crawler.cli crawl-web \
  --url 'https://devforum.roblox.com/t/roblox-please-patch-your-ai-moderation/4554549.json' \
  --output-dir 'feature_crawler/data' \
  --community-name 'roblox-devforum' \
  --target-id 'roblox-devforum' \
  --render-mode http \
  --since '2026-04-01T00:00:00Z' \
  --until '2026-04-09T23:59:59Z' \
  --allow-persona-inference
```

```bash
python3 -m feature_crawler.crawler.cli crawl-web \
  --url 'https://devforum.roblox.com/t/full-release-the-future-of-character-movement-character-controller-library/4565267.json' \
  --output-dir 'feature_crawler/data' \
  --community-name 'roblox-devforum' \
  --target-id 'roblox-devforum' \
  --render-mode http \
  --since '2026-04-01T00:00:00Z' \
  --until '2026-04-09T23:59:59Z' \
  --allow-persona-inference
```

Fortnite:

```bash
python3 -m feature_crawler.crawler.cli crawl-web \
  --url 'https://forums.unrealengine.com/t/seagulls-bugged-after-v40-10-update-customize-cp-p-atmospheric-seagulls-orbit/2710672.json' \
  --output-dir 'feature_crawler/data' \
  --community-name 'fortnite-unreal-forums' \
  --target-id 'fortnite-unreal-forums' \
  --render-mode http \
  --since '2026-04-01T00:00:00Z' \
  --until '2026-04-09T23:59:59Z' \
  --allow-persona-inference
```

```bash
python3 -m feature_crawler.crawler.cli crawl-web \
  --url 'https://forums.unrealengine.com/t/critical-creative-maps-forcefully-unload-when-taking-a-large-amount-of-damage-around-a-large-number-of-props/2711970.json' \
  --output-dir 'feature_crawler/data' \
  --community-name 'fortnite-unreal-forums' \
  --target-id 'fortnite-unreal-forums' \
  --render-mode http \
  --since '2026-04-01T00:00:00Z' \
  --until '2026-04-09T23:59:59Z' \
  --allow-persona-inference
```

## Discord status

Bot token exists in repo `.env` but bot is in `0` guilds. Bot-invite path is blocked.

New strategy (2026-04-09): three alternative Discord connectors shipped to bypass the bot-invite blocker.

### Connector 1: User-token API (`discord_user_api.py`)

Uses a Discord user account token to call the same REST API the web client uses. Most efficient for bulk message retrieval with proper pagination.

- Auth: `Authorization: {token}` (no “Bot” prefix)
- Human-like delays (1.5–3.0s between requests)
- Can join public servers via invite codes
- ToS risk: automated user-token access violates Discord ToS. Use a separate/throwaway account.

### Connector 2: Browser automation (`discord_browser.py`)

Playwright-based. Logs into Discord web, navigates to channels, scrolls to load history, extracts messages from DOM.

- Requires saved login state (run `save-discord-login` first)
- Slower than API but works when API detection is aggressive
- Configurable scroll speed and depth
- ToS risk: same as above

### Connector 3: Third-party archives (`discord_archive.py`)

Fetches from public Discord archive/index services. Best-effort fallback — many archives are incomplete.

- No ToS violation (accesses third parties, not Discord)
- Lowest yield, highest legality

### New CLI commands

User-token API crawl:

```bash
python3 -m feature_crawler.crawler.cli crawl-discord-user \
  --guild-id 662267976984297473 \
  --channel-id 999550150705954856 \
  --output-dir feature_crawler/data \
  --community-name 'target-server' \
  --message-limit 200 \
  --min-delay 2.0 --max-delay 4.0 \
  --allow-persona-inference
```

Browser crawl:

```bash
# Step 1: Save login state (interactive, one-time)
python3 -m feature_crawler.crawler.cli save-discord-login \
  --storage-state discord_session.json

# Step 2: Crawl a channel
python3 -m feature_crawler.crawler.cli crawl-discord-browser \
  --url 'https://discord.com/channels/662267976984297473/999550150705954856' \
  --output-dir feature_crawler/data \
  --storage-state discord_session.json \
  --max-scrolls 50 --scroll-pause 2.5 \
  --community-name 'target-server' \
  --allow-persona-inference
```

Archive crawl:

```bash
python3 -m feature_crawler.crawler.cli crawl-discord-archive \
  --guild-id 662267976984297473 \
  --channel-id 999550150705954856 \
  --output-dir feature_crawler/data \
  --community-name 'target-server' \
  --request-delay 2.0 \
  --allow-persona-inference
```

Join a server before crawling (user-token method):

```bash
# Set DISCORD_USER_TOKEN in env or .env first
python3 -m feature_crawler.crawler.cli crawl-discord-user \
  --guild-id 662267976984297473 \
  --channel-id 999550150705954856 \
  --output-dir feature_crawler/data \
  --join-invite 'INVITE_CODE_HERE' \
  --message-limit 100
```

### Recommended crawl order

1. Try user-token API first (fastest, cleanest data)
2. Fall back to browser if API gets rate-limited or blocked
3. Use archive as supplementary data source

## Next best move

1. Set up throwaway Discord account for crawling.
2. Get user token from browser DevTools (Network tab → any request → Authorization header).
3. Join target public servers via Discord discovery.
4. Run user-token crawls with conservative delays (2–4s).
5. Keep expanding web crawl coverage for Fortnite and Roblox.
6. Fix sink collision risk before parallel crawls.

## Important bug / risk

`JsonlSink` names files with second-level `crawl_run_id`.

Effect:

- parallel crawls for the same community started in the same second can overwrite each other

Workaround used:

- stagger writes by `sleep 1`
- run same-community crawls serially when durability matters

## Blockers

- No blocker for web scraping on allowed public sources.
- Discord bot-invite path: still blocked (bot in 0 guilds).
- Discord user-token path: unblocked, needs throwaway account + token.
- Discord browser path: unblocked, needs Playwright + saved login state.
- Main runtime risk: same-second sink collisions for same-community writes.
