# Running the Discord Crawler

## Prerequisites

- Python 3.10+
- A Discord account (use a throwaway — automated access violates Discord ToS)
- The account must have joined the target server(s)

## Getting your Discord user token

1. Log in to Discord in your browser (Chrome recommended)
2. Open DevTools: `F12` or `Cmd+Opt+I`
3. Go to the **Network** tab
4. Click on any channel in Discord (to trigger an API request)
5. In the Network tab, find any request to `discord.com/api`
6. Click the request → **Headers** tab → find `Authorization`
7. Copy the value (long string like `MTQ5MT...iiXA`)

## Setting the token

Option A — environment variable:

```bash
export DISCORD_USER_TOKEN='your_token_here'
```

Option B — project .env file:

```bash
echo 'DISCORD_USER_TOKEN=your_token_here' >> .env
```

Option C — Doppler (preferred for team use):

```bash
doppler secrets set DISCORD_USER_TOKEN 'your_token_here' --project api_keys --config dev
```

## Quick test

Verify the token works:

```bash
python3 -c "
import json, os
from urllib.request import Request, urlopen
token = os.environ.get('DISCORD_USER_TOKEN') or open('.env').read().split('DISCORD_USER_TOKEN=')[1].split('\n')[0].strip()
req = Request('https://discord.com/api/v10/users/@me', headers={'Authorization': token})
me = json.loads(urlopen(req).read())
print(f'Logged in as: {me[\"username\"]} (ID: {me[\"id\"]})')
guilds = json.loads(urlopen(Request('https://discord.com/api/v10/users/@me/guilds', headers={'Authorization': token})).read())
print(f'Joined {len(guilds)} server(s):')
for g in guilds: print(f'  {g[\"id\"]} — {g[\"name\"]}')
"
```

## Crawl commands

### Method 1: CLI (simplest)

Single channel:

```bash
cd AgentPersonasProject

python3 -m feature_crawler.crawler.cli crawl-discord-user \
  --guild-id 662267976984297473 \
  --channel-id 938713143759216720 \
  --output-dir feature_crawler/data \
  --community-name 'midjourney' \
  --message-limit 200 \
  --min-delay 2.5 --max-delay 4.0 \
  --allow-persona-inference
```

Multiple channels (run sequentially to avoid sink collisions):

```bash
for CHANNEL in 938713143759216720 992207085146222713 989270517401911316 1100553646162317332; do
  python3 -m feature_crawler.crawler.cli crawl-discord-user \
    --guild-id 662267976984297473 \
    --channel-id "$CHANNEL" \
    --output-dir feature_crawler/data \
    --community-name 'midjourney' \
    --message-limit 200 \
    --min-delay 2.5 --max-delay 4.0 \
    --allow-persona-inference
  sleep 2
done
```

### Method 2: Runner script (multi-channel, manifest output)

```bash
python3 feature_crawler/scripts/discord_crawl_runner.py \
  --guild 662267976984297473 \
  --channel 938713143759216720 \
  --channel 992207085146222713 \
  --channel 989270517401911316 \
  --channel 1100553646162317332 \
  --community-name midjourney \
  --message-limit 200 \
  --min-delay 2.5 --max-delay 5.0 \
  -v
```

### Method 3: Browser automation (Playwright fallback)

If the API path gets rate-limited or blocked:

```bash
# One-time: save login state
python3 -m feature_crawler.crawler.cli save-discord-login \
  --storage-state discord_session.json

# Crawl
python3 -m feature_crawler.crawler.cli crawl-discord-browser \
  --url 'https://discord.com/channels/662267976984297473/938713143759216720' \
  --output-dir feature_crawler/data \
  --storage-state discord_session.json \
  --max-scrolls 50 --scroll-pause 2.5 \
  --community-name midjourney
```

## Joining a new server

Via browser: navigate to the server's invite link and click Join.

Via API (if you have an invite code):

```bash
python3 -m feature_crawler.crawler.cli crawl-discord-user \
  --guild-id TARGET_GUILD_ID \
  --channel-id SOME_CHANNEL_ID \
  --output-dir feature_crawler/data \
  --join-invite INVITE_CODE \
  --message-limit 50
```

## Discovering channels in a server

```bash
python3 -c "
import json, os
from urllib.request import Request, urlopen
token = os.environ['DISCORD_USER_TOKEN']
GUILD = '662267976984297473'  # change this
headers = {'Authorization': token}
channels = json.loads(urlopen(Request(f'https://discord.com/api/v10/guilds/{GUILD}/channels', headers=headers)).read())
for c in sorted(channels, key=lambda x: x.get('position', 999)):
    if c['type'] in (0, 5, 15):
        kind = 'FORUM' if c['type'] == 15 else 'TEXT'
        print(f'  {kind} #{c[\"name\"]:40s} id={c[\"id\"]}')
"
```

## Automation (macOS launchd)

```bash
chmod +x feature_crawler/scripts/setup_automation.sh
./feature_crawler/scripts/setup_automation.sh
```

This installs a launchd job that runs daily at 6:00 AM.

Manual trigger: `launchctl start com.agentpersonas.discord-crawl`

Logs: `tail -f feature_crawler/logs/crawl_stdout.log`

Uninstall: `launchctl unload ~/Library/LaunchAgents/com.agentpersonas.discord-crawl.plist`

## Output structure

Records land in:

```
feature_crawler/data/discord/{server_name}/{channel_name}_{channel_id}/{month}/{day}/{year}/{crawl_run_id}.jsonl
```

Each JSONL file contains typed records: `community`, `thread`, `account`, `profile_snapshot`, `message`, `interaction`.

## Slow crawl first

- Keep the first pass single-account, low volume, low concurrency.
- Start with one server and a short channel list.
- Keep `--message-limit` small enough to verify foldering and record quality before widening the crawl.

## Multi-account crawl research note

Research on large Discord scraping operations points to the same broad pattern: operators shard work across many low-trust accounts, keep per-account rate state, distribute server coverage, and retire tokens that get blocked or expire. This repo does **not** implement that pattern right now. Treat it as external context only, not the default operating mode.

For this codebase:

- default path: one account, one slow runner, one channel folder per run
- next safe step: widen channel coverage gradually inside the same server
- future design note: if scale becomes necessary, document a coordinator separately and get explicit policy review first

Research trail from the 2026-04-09 crawl note:

- Spy.pet / 404 Media reporting
- DataBreaches and Malwarebytes summaries of the Spy.pet scrape
- `discum` on PyPI as a commonly referenced user-token library
- Discord support policy on self-bots: https://support.discord.com/hc/en-us/articles/115002192352-Automated-User-Accounts-Self-Bots

## Known Midjourney channels (high-value for personas)

| Channel | ID | Category | Signal type |
|---|---|---|---|
| #discussion | 938713143759216720 | Chat | General conversation, opinions |
| #prompt-craft | 992207085146222713 | Chat | Technique sharing, expertise |
| #ideas-and-features | 989270517401911316 | Feedback | Feature requests, pain points |
| #bug-reporting | 1100553646162317332 | Feedback | Bug reports, technical detail |
| #announcements | 952771221915840552 | Info | Official updates (1 author) |
| #support | 958069758211797092 | Support | Help-seeking behavior |
| #superusers-prompt-craft | 1183983460306653275 | Super-user | Expert community |

## Rate limiting and safety

- Default delays: 2.5–5.0 seconds between API requests
- Discord rate limit (429) is handled automatically with retry + jitter
- Never run parallel crawls against the same server
- Token can be revoked by Discord at any time — refresh if crawls return 401
- Keep `--message-limit` reasonable (200–500 per channel per run)

## Troubleshooting

**401 Unauthorized**: Token expired or revoked. Get a new one from DevTools.

**403 Forbidden**: Account doesn't have access to that channel. Some channels are role-restricted.

**429 Too Many Requests**: Rate limited. The crawler handles this automatically, but increase `--min-delay` if it happens often.

**0 records**: Channel might be empty, or all messages are from bots (filtered by default).
