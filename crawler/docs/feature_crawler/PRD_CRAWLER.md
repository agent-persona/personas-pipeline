## Community Crawler — owner: Max

---

### Problem

Persona quality is bottlenecked by behavioral signal density. The persona pipeline (ingestion → segmentation → synthesis → twin runtime) needs **longitudinal, structured, replayable evidence** from the platforms where real community discourse happens. The crawler's job is to capture who said what, where, when, in reply to whom, and with what behavioral signals — then land those records in a schema that downstream can trace, replay, cluster, and synthesize into stable personas.

The crawler does NOT own the persona. It owns the evidence stream.

---

### Goals

1. **Structured evidence as canonical output.** Every crawl produces typed records (`account`, `profile_snapshot`, `community`, `thread`, `message`, `interaction`) in a normalized schema. These are the source of truth — everything else (Obsidian vaults, dashboards, exports) is a projection.
2. **User identity anchoring.** Stable `account` records keyed by `{platform, platform_user_id}`. Profile-mutable fields captured as versioned `profile_snapshot` records so persona pipeline can track drift (bio changes, display name changes, community membership over time).
3. **Community-first hierarchy.** Crawl targets are communities (subreddits, servers, channels, groups), not individual users. The natural structure is: platform → community → thread → message → user. Users emerge from conversations, not the other way around.
4. **Time-aware by default.** Every record carries `observed_at`, `created_at` (platform timestamp), and `crawl_run_id`. Incremental crawls use `since` cursors. Deletion/edit events propagate as tombstone records. This gives downstream the primitives for recency windows, decay, and freshness.
5. **Approved-source by default.** Phase 1 focuses on opt-in and policy-clear sources: Discord communities with explicit access, owned community properties, and public web sources whose terms permit crawling, storage, and persona inference. Restricted platforms remain blocked until legal and policy sign-off.
6. **Resilient to platform constraints.** Rate-limit-aware, auth-refresh-capable, ToS-compliant. Graceful degradation when APIs restrict access.
7. **New platform connector ships in < 1 week.**

---

### Canonical Record Types

These are the contracts between crawler and ingestion. Ingestion validates shape; segmentation and synthesis consume downstream.

#### `account`
The identity anchor. One per `{platform, platform_user_id}`. Immutable ID, mutable fields captured via snapshots.

```jsonc
{
  "record_type": "account",
  "platform": "reddit",
  "platform_user_id": "t2_abc123",       // platform's stable ID
  "username": "spez",                     // display handle (mutable)
  "account_created_at": "2005-06-05T00:00:00Z",
  "first_observed_at": "2026-03-15T10:00:00Z",
  "crawl_run_id": "run_20260408_001",
  "evidence_pointer": {
    "source_url": "https://reddit.com/user/spez",
    "fetched_at": "2026-04-08T12:00:00Z"
  }
}
```

#### `profile_snapshot`
Point-in-time capture of mutable account fields. Enables persona drift detection.

```jsonc
{
  "record_type": "profile_snapshot",
  "platform": "reddit",
  "platform_user_id": "t2_abc123",
  "snapshot_at": "2026-04-08T12:00:00Z",
  "crawl_run_id": "run_20260408_001",
  "fields": {
    "display_name": "spez",
    "bio": "CEO of Reddit",
    "karma": 123456,
    "avatar_url": "https://...",
    "communities_active": ["wallstreetbets", "LocalLLaMA"],
    "follower_count": null,           // null if platform doesn't expose
    "account_age_days": 7613
  }
}
```

#### `community`
Metadata about a crawl target community.

```jsonc
{
  "record_type": "community",
  "platform": "reddit",
  "community_id": "t5_2th52",
  "community_name": "wallstreetbets",
  "community_type": "subreddit",        // subreddit | server | channel | group | page | account
  "parent_community_id": null,          // for discord channels → server
  "description": "Like 4chan found a Bloomberg terminal",
  "member_count": 15200000,
  "rules_summary": "...",
  "observed_at": "2026-04-08T12:00:00Z",
  "crawl_run_id": "run_20260408_001",
  "evidence_pointer": {
    "source_url": "https://reddit.com/r/wallstreetbets",
    "fetched_at": "2026-04-08T12:00:00Z"
  }
}
```

#### `thread`
A conversation container (Reddit post, X tweet with replies, Discord thread, Twitch stream chat session).

```jsonc
{
  "record_type": "thread",
  "platform": "reddit",
  "thread_id": "t3_abc123",
  "community_id": "t5_2th52",
  "title": "GME Squeeze Discussion",
  "author_platform_user_id": "t2_abc123",
  "created_at": "2026-04-01T14:00:00Z",
  "observed_at": "2026-04-08T12:00:00Z",
  "crawl_run_id": "run_20260408_001",
  "metadata": {
    "score": 15420,
    "upvote_ratio": 0.94,
    "flair": "Discussion",
    "is_pinned": false,
    "comment_count": 1247
  },
  "evidence_pointer": {
    "source_url": "https://reddit.com/r/wallstreetbets/comments/abc123",
    "fetched_at": "2026-04-08T12:00:00Z"
  }
}
```

#### `message`
An individual utterance within a thread. The core behavioral signal.

```jsonc
{
  "record_type": "message",
  "platform": "reddit",
  "message_id": "t1_xyz789",
  "thread_id": "t3_abc123",
  "community_id": "t5_2th52",
  "author_platform_user_id": "t2_abc123",
  "body": "This is the actual comment text...",
  "created_at": "2026-04-01T14:32:00Z",
  "observed_at": "2026-04-08T12:00:00Z",
  "crawl_run_id": "run_20260408_001",
  "reply_to_message_id": null,          // null if top-level
  "reply_to_user_id": null,
  "metadata": {
    "score": 42,
    "is_edited": false,
    "awards": [],
    "depth": 0                          // nesting level in thread
  },
  "evidence_pointer": {
    "source_url": "https://reddit.com/r/wallstreetbets/comments/abc123/comment/xyz789",
    "fetched_at": "2026-04-08T12:00:00Z"
  }
}
```

#### `interaction`
Derived edge between two users. Built from reply chains, mentions, quotes, reactions.

```jsonc
{
  "record_type": "interaction",
  "platform": "reddit",
  "interaction_type": "reply",          // reply | mention | quote | react
  "source_user_id": "t2_abc123",
  "target_user_id": "t2_def456",
  "message_id": "t1_xyz789",
  "thread_id": "t3_abc123",
  "community_id": "t5_2th52",
  "created_at": "2026-04-01T14:32:00Z",
  "crawl_run_id": "run_20260408_001",
  "evidence_pointer": {
    "source_url": "https://reddit.com/r/wallstreetbets/comments/abc123/comment/xyz789",
    "fetched_at": "2026-04-08T12:00:00Z",
    "derived_from_message_id": "t1_xyz789"
  }
}
```

#### `tombstone`
Deletion/edit propagation. When content is removed or modified on-platform.

```jsonc
{
  "record_type": "tombstone",
  "platform": "reddit",
  "tombstone_type": "deleted",          // deleted | edited | suspended
  "target_record_type": "message",
  "target_record_id": "t1_xyz789",
  "observed_at": "2026-04-09T08:00:00Z",
  "crawl_run_id": "run_20260409_001",
  "previous_body_hash": "sha256:...",   // proof we had it, without retaining content
  "reason": "author_deleted",           // author_deleted | mod_removed | admin_removed | edited
  "evidence_pointer": {
    "source_url": "https://reddit.com/r/wallstreetbets/comments/abc123/comment/xyz789",
    "fetched_at": "2026-04-09T08:00:00Z"
  }
}
```

---

### Success Metrics

| Metric | Target | Why |
|---|---|---|
| Successful fetch rate per connector | > 99% | Reliability baseline |
| Record schema validity | 100% (enforced at write) | Ingestion contract |
| User identity accuracy (same platform_user_id = same account) | 100% within platform | Persona stability depends on this |
| Incremental crawl correctness (no duplicates, no gaps) | Verified by run-to-run diff | Replay semantics |
| Tombstone propagation latency | < 1 crawl cycle | Deletion compliance |
| New connector dev time | < 5 engineering days | Velocity |
| **Persona-level (eval, owned jointly with synthesis):** | | |
| Held-out same-person recognition | > 80% accuracy | If crawler signals are too thin, personas are indistinguishable |
| Cross-run persona stability (same user, two runs → same persona cluster) | > 90% match rate | Evidence must be consistent enough for stable clustering |
| Persona drift precision | > 80% precision on held-out drift labels | Persona updates should reflect real new evidence, not crawl noise |
| Grounded-quote recall (synthesis can cite actual messages) | > 85% of persona claims have evidence_pointer | evidence_pointer chain must survive the full pipeline |

---

### Platform Hierarchy Mapping

Each platform's native structure maps to the canonical types:

| Platform | Community | Sub-community | Thread | Message | User ID |
|---|---|---|---|---|---|
| **Reddit** | Subreddit (`r/X`) | — | Post | Comment | `t2_` ID |
| **X (Twitter)** | Topic cluster / List | — | Tweet (root) | Reply / QT | Numeric user ID |
| **Discord** | Server (guild) | Channel | Thread (or time-window chunk) | Message | Snowflake user ID |
| **Twitch** | Channel | — | Stream session (VOD) | Chat message | Twitch user ID |
| **Facebook** | Group / Page | — | Post | Comment | FB user ID |
| **Instagram** | Account / Hashtag cluster | — | Post | Comment | IG user ID |
| **Web (approved)** | Site / forum / comment area | Section / board | Page / discussion | Comment / paragraph / post | Site-local author ID or canonical author key |

---

### Per-Platform Policy Matrix

Legal/policy gates are first-class. No connector ships without clearing these.

| Platform | Collection Basis | Allowed to Collect | Allowed to Infer (persona traits) | Allowed to Cross-link (off-platform) | Storage Retention | Deletion SLA | Commercial Status | Action Required |
|---|---|---|---|---|---|---|---|---|
| **Reddit** | blocked | Public posts/comments via approved API access only | Within-platform behavioral inference OK, subject to Reddit policy restrictions | **PROHIBITED** — Reddit policy explicitly bans matching Reddit data with off-platform identifiers | Must delete when user/mod deletes (tombstone) | Same crawl cycle | **Requires permission/contract for commercial use.** Free tier = non-commercial only. Model training on Reddit content needs explicit consent (updated March 2026). | Legal review before Day 1 launch. Pursue Reddit Data API commercial license. Do not use Pushshift except for approved moderator tooling. |
| **X** | blocked | Public tweets via API v2 (pay-per-use) | Inference OK per API ToS | Not explicitly prohibited but risky | Standard retention | Best-effort (no push notification of deletes) | Pay-per-use is commercial by default | Budget approval. Track Developer Console pricing (brittle — $0.005/read is current but subject to change). |
| **Discord** | consented | Messages in servers where bot is invited, with MESSAGE_CONTENT_INTENT | Inference OK within bot scope | Not prohibited for invited bots | Standard retention | Tombstone on deletion events (gateway) | Free (bot API) | Bot must be verified if in >75 servers. MESSAGE_CONTENT_INTENT requires verification for bots in 100+ servers. |
| **Twitch** | public-permitted | Public chat via IRC, channel metadata via Helix | Inference OK | Not explicitly addressed | Standard retention | Best-effort | Free (API) | Register application at dev.twitch.tv |
| **Facebook** | blocked | Public page posts. Group posts require app review + member consent. | Limited — Meta restricts behavioral profiling | **PROHIBITED** under Meta Platform Terms | Must honor deletion requests | Per Meta policy | Requires app review for group data. Pages = public. | Start app review process early (slow). |
| **Instagram** | blocked | Public post comments via Graph API | Limited — same Meta restrictions | **PROHIBITED** under Meta Platform Terms | Must honor deletion requests | Per Meta policy | Requires app review | Start app review process early. |
| **Web (approved)** | owned / consented / public-permitted | Public pages, forums, blogs, comments only where terms and robots allow crawling, storage, and persona inference | Allowed where site terms permit | Site-specific; default to no cross-linking unless explicitly permitted | Per site policy | Best-effort unless source exposes deletion/update feed | Commercial use depends on source license/terms | Maintain source allowlist with terms review before activation. |

**Gate rule:** A connector is `blocked` until its policy row is reviewed and signed off by project lead. The `commercial_status` column determines whether we can use the data for persona generation at all.

**Enforcement rule:** No Playwright, proxy, or scraper fallback may be used to bypass a `blocked` collection basis. Fallback crawling is only allowed for `owned`, `consented`, or `public-permitted` sources.

---

### Time Semantics

The crawler is the clock for the evidence system. Downstream depends on these guarantees:

**Timestamps on every record:**
- `created_at` — when the content was created on-platform (platform's timestamp)
- `observed_at` — when the crawler first saw this record
- `crawl_run_id` — which run produced this record (enables replay, diffing)

**Incremental crawl:**
- Each connector maintains a `since` cursor per `{community, record_type}`
- Cursors stored in Postgres, updated atomically at run completion
- On failure: cursor not advanced → next run retries from same point (at-least-once delivery, deduplicated by `{platform, record_id}` at ingestion)

**Freshness & decay (consumed by segmentation/synthesis, defined here for contract):**
- Records carry `created_at` so downstream can apply recency windows
- `profile_snapshot` records enable detecting when a user's self-description changes
- Suggested decay tiers (configurable by downstream): hot (< 7d), warm (7–90d), cold (> 90d), stale (> 365d)

**Deletion/edit propagation:**
- Crawler emits `tombstone` records when previously-observed content is no longer present
- Reddit: hard requirement — must comply with content deletion within same crawl cycle
- Other platforms: best-effort detection on next incremental crawl
- Downstream: tombstoned message bodies are redacted; metadata retained for structural integrity

**Snapshot versioning:**
- `profile_snapshot` records are append-only — new snapshot per crawl run if any field changed
- Enables synthesis to detect: "user changed bio from X to Y" (persona drift signal)
- `community` records also snapshot (member count, description changes over time)

---

### Technical Architecture

**Runtime:** Connector workers on ECS Fargate, one container image per source family. Common base class (`CommunityConnector`) handles:
- OAuth token refresh + storage (AWS Secrets Manager)
- Rate-limit budgeting with per-platform profiles
- Retry with exponential backoff
- Schema validation (Pydantic models for each record type) before write
- PII scrubbing via Comprehend before records land in the lake
- Cursor management (Postgres)

**Storage (canonical):**
- S3 bronze tier (Iceberg tables) partitioned by `platform/community_id/crawl_run_id/record_type`
- This is the source of truth. Ingestion reads from here.

**Web crawling (fallback):** Playwright for sources already classified as `owned`, `consented`, or `public-permitted`, where API access is insufficient or cost-prohibitive. API-first always. Never use fallback crawling, proxy rotation, or CAPTCHA-bypass tooling to bypass a `blocked` policy row.

**Secrets:** Doppler (`doppler run --project api_keys --config dev`) for local dev. AWS Secrets Manager for production.

**Interfaces:**
```
Connector.fetch(target: CrawlTarget, since: Cursor) → Iterator[Record]
```
Where `Record` is one of: `account`, `profile_snapshot`, `community`, `thread`, `message`, `interaction`, `tombstone`.

Output: Iceberg tables partitioned by `platform/community_id/date`. Each record includes `crawl_run_id` for replay.

---

### Obsidian Projection (View Layer — NOT Source of Truth)

An optional post-crawl step generates an Obsidian vault from the canonical records. This is for human exploration and debugging, not for the persona pipeline.

**Projection rules:**
- `community` → folder + `_community.md`
- `thread` → subfolder + `_topic.md`
- `message` → entries in `conversation.md` with `[[wikilinks]]` to user notes
- `account` + latest `profile_snapshot` → `_users/{platform}/{username}.md`
- `interaction` → implicit via wikilinks + optional Dataview queries
- Graph view: color by platform, group by community, user nodes connect across threads

**Vault structure mirrors the canonical hierarchy:**
```
vault/
├── _users/{platform}/{username}.md
├── {platform}/
│   ├── {community}/
│   │   ├── _community.md
│   │   ├── {thread_slug}/
│   │   │   ├── _topic.md
│   │   │   └── conversation.md
│   │   └── ...
│   └── ...
└── ...
```

This projection is regenerable from the lake at any time. Losing the vault loses nothing.

---

### Phasing

**Phase 1 — MVP (4 weeks)**
- Discord connector (bot-based: servers → channels → threads → messages → accounts)
- Approved web connector family (owned sites, consented communities, public-permitted forums/blogs/comments on an allowlist)
- Canonical record schema (Pydantic models, Iceberg tables)
- Incremental crawl with cursor management
- Collection-basis enforcement (`owned | consented | public-permitted | blocked`)
- Tombstone detection for Discord + approved web sources where feasible
- Profile snapshot versioning
- Obsidian projection generator (basic)
- **Gate: Reddit commercial data license + policy review should be initiated.**

**Phase 2 — Expand (4 weeks)**
- Reddit connector if commercial license + policy review are approved
- X connector (pay-per-use, budget-capped)
- Twitch connector (Helix + IRC chat)
- Interaction record extraction (reply graphs, mention graphs)
- Incremental crawl hardening (dedup, gap detection, run-to-run diff, drift-noise checks)
- Persona eval integration: held-out recognition, cross-run stability, persona drift precision
- **Gate: X budget ceiling approved. Discord bot verified if >75 servers.**

**Phase 3 — Full coverage (4 weeks)**
- Facebook Groups/Pages connector (requires app review — start in Phase 1)
- Instagram connector (same Meta app review)
- Scheduled crawl runs (cron or event-driven via orchestration)
- Cross-platform user heuristic aliasing (shared bio links, self-identification) — **within-platform only for Reddit per policy**
- Stale-persona detection: flag accounts with no new messages in N crawl cycles
- **Gate: Meta app review approved. Cross-link policy audit complete.**

---

### Out of Scope

- Transforms beyond schema normalization (owned by ingestion/orchestration dbt step)
- Vector embedding of crawled content (owned by synthesis)
- Sentiment analysis or NLP enrichment (downstream consumer)
- Cross-platform identity resolution via ML (phase 2 uses heuristics only, Reddit prohibited)
- Private/DM message crawling (ethical + ToS boundary)
- Persona generation or twin runtime (owned by synthesis + downstream)

---

### Connector Implementation Notes

Platform-specific guidance for the Day-1 and Phase-2 connectors. These are engineering notes, not product requirements — they capture current API realities and recommended libraries.

#### Reddit (Phase 2)

**Library:** PRAW (Python Reddit API Wrapper) — handles OAuth, rate limiting, and pagination natively.

```python
# Auth
reddit = praw.Reddit(
    client_id="...", client_secret="...", user_agent="agent-personas-crawler/0.1"
)
# Subreddit → threads (hot/new/top)
for submission in reddit.subreddit("wallstreetbets").hot(limit=100):
    # → thread record
    # Expand comment tree
    submission.comments.replace_more(limit=32)  # expand "load more" nodes
    for comment in submission.comments.list():   # flattened DFS
        # → message record (comment.author → account record)
        # → interaction record (comment.parent() for reply edges)
```

**Key constraints:**
- OAuth required — 100 req/min (OAuth), 10 req/min unauthenticated. PRAW handles `X-Ratelimit-*` headers automatically and sleeps when needed (configurable via `ratelimit_seconds`, default 5s).
- `replace_more(limit=N)` expands "load more comments" nodes. Set `limit=0` to strip them (faster, lossy). Set `limit=None` for full tree (slow, complete). For persona evidence, use `limit=32` as a balance — captures ~95% of visible comments without hammering the API.
- `submission.comments.list()` returns a flattened DFS traversal. Use `comment.parent_id` and `comment.depth` to reconstruct thread structure for the `message.reply_to_message_id` field.
- Author can be `None` (deleted accounts). Emit a tombstone when `comment.author is None` but `comment.body != "[deleted]"` — indicates suspended account.
- Subreddit descriptions: `reddit.subreddit("X").description` → `community.description` field.
- **Content deletion compliance:** On each incremental crawl, check previously-observed message IDs. If body is now `[deleted]` or `[removed]`, emit a `tombstone` record with `previous_body_hash`. Do NOT retain the original body.
- **Commercial use:** Free tier is non-commercial. Reddit's Responsible Builder Policy (updated March 2026) requires explicit permission/contract for commercial data use and prohibits model training on Reddit content without consent. Cross-linking Reddit data with off-platform identifiers is **explicitly prohibited**.

#### Discord (Phase 1 — Day 1)

**Library:** discord.py (v2.6+) or Pycord — async-native, handles gateway and REST API.

```python
import discord

intents = discord.Intents.default()
intents.message_content = True     # PRIVILEGED — requires verification at 100+ servers
intents.members = True             # PRIVILEGED — for user profile data

bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    for guild in bot.guilds:
        # → community record (guild.name, guild.member_count, etc.)
        for channel in guild.text_channels:
            # → community record (sub-community, parent_community_id = guild.id)
            async for message in channel.history(limit=None, after=since_cursor):
                # → message record
                # → account record (message.author)
                # → interaction record (message.reference for replies, message.mentions)
```

**Key constraints:**
- **Rate limits:** 50 req/sec global. `channel.history()` paginates automatically (100 messages per request). For large servers, this is the bottleneck — budget ~2 sec per 100 messages.
- **MESSAGE_CONTENT_INTENT:** Required to read message body text. Bots in <100 servers get it by default. Bots in 100+ servers need Discord verification + approval with a "compelling use case." Plan for this if crawling many servers.
- **SERVER_MEMBERS_INTENT:** Required for `guild.members` (full member list + profile data). Same verification gate at 100+ servers.
- Bot must be **invited** to each server with `READ_MESSAGE_HISTORY` + `VIEW_CHANNEL` permissions. This means the crawl surface is limited to servers where we have an invitation.
- **Thread handling:** Discord threads are separate from channel message history. Use `channel.threads` (active) and `channel.archived_threads()` to enumerate. Each thread is its own message history. Map to `thread` records with `parent_community_id = channel.id`.
- **Deletion events:** Discord gateway sends `MESSAGE_DELETE` events in real-time. If running a persistent bot, capture these as tombstone records. For batch crawling, compare current message IDs against previous run (slower, gap-prone).
- **Snowflake IDs:** Discord uses Snowflake IDs which encode creation timestamp. Extract with `discord.utils.snowflake_time(id)` — use as `created_at` when API doesn't return timestamp directly.

#### X / Twitter (Phase 2)

**Library:** tweepy v4+ or raw HTTP to X API v2.

**Key constraints:**
- **Pay-per-use:** $0.005/post read (current, but pricing lives in Developer Console and is unstable). 2M post reads/month cap on pay-per-use. Above that → Enterprise ($42K+/yr).
- **Search endpoint** (`/2/tweets/search/recent`) is the primary crawl vector for topic-based collection. Returns tweets from the last 7 days. Full-archive search requires Academic or Enterprise access.
- **Dedup:** Requesting the same tweet within a 24-hour UTC window counts as one charge. Cache tweet IDs per crawl run.
- **User lookup:** `GET /2/users/by/username/{username}` → account record. Expansions (`tweet.fields`, `user.fields`) reduce call count.
- **Budget control:** Implement a hard per-run spending cap. Track reads in real-time against the monthly ceiling. Fail-safe: stop crawling when at 80% of budget.

#### Twitch (Phase 2)

**Library:** Twitch Helix API (REST) for metadata + IRC/EventSub for live chat.

**Key constraints:**
- **Channel metadata:** Helix API (`GET /helix/channels`, `/helix/users`) for community and account records. Standard OAuth rate limits.
- **Live chat:** IRC connection (`irc.chat.twitch.tv:6667`) or EventSub for real-time chat messages. High volume — popular channels can hit 100+ msg/sec.
- **Sampling strategy (open question):** For persona evidence, we likely need representative samples, not full capture. Options: (a) time-window sampling (capture 5-min windows every 30 min), (b) user-triggered sampling (capture all messages from users who appear in >1 community), (c) entropy-based (skip repetitive spam/emotes, keep substantive messages).
- **VOD chat:** Not directly available via API for most channels. Third-party tools (e.g., chat replay from Twitch's own player) exist but are fragile.

---

### Open Source Tools & Build-vs-Reuse Assessment

Evaluated existing OSS projects for reuse or reference. None are drop-in replacements for our pipeline (none produce typed persona-evidence records), but several have battle-tested components worth forking or wrapping.

#### Discord

| Project | What it does | Reuse potential | Link |
|---|---|---|---|
| **LAION-AI/Discord-Scrapers** | Bot-based channel scraper → HuggingFace dataset push. Supports `fetch_all` (full history) and incremental (new-only). Pluggable `condition_fn` + `parse_fn` for filtering/parsing. | **High — best starting point for Discord connector.** Architecture matches our pattern: bot token → channel.history → filter → normalize → push. Fork and replace HF push with our Iceberg writer. | [GitHub](https://github.com/LAION-AI/Discord-Scrapers) |
| **pixelatedxp/discord-member-scraper** | Extracts every user ID in a server via message history, archived threads, and welcome logs. Handles rate limits, saves progress. | **Medium — useful for account record extraction.** Complements the message scraper with exhaustive user enumeration. | [GitHub](https://github.com/pixelatedxp/discord-member-scraper) |
| **santiment/discord_bot_scraper** | Bot-based scraper with configurable start date for historical retrieval. | **Low-Medium — simpler than LAION but reference for cursor-based history crawling.** | [GitHub](https://github.com/santiment/discord_bot_scraper) |
| **lorenz234/Discord-Data-Scraping** | Scrapes messages + member count from servers using Python. | **Low — basic reference only.** | [GitHub](https://github.com/lorenz234/Discord-Data-Scraping) |

**Recommendation:** Start from LAION-AI/Discord-Scrapers for the Discord connector. Replace the HuggingFace push with our record schema + Iceberg writer. Add the member enumeration logic from pixelatedxp for account coverage.

#### Reddit

| Project | What it does | Reuse potential | Link |
|---|---|---|---|
| **datavorous/yars** | Reddit scraper **without API keys** — uses `.json` endpoint trick. Search, subreddit posts (hot/new/top), user activity, media download. | **Medium — useful as fallback if API access is delayed.** No OAuth overhead, but fragile (undocumented endpoint) and may violate Reddit ToS for commercial use. | [GitHub](https://github.com/datavorous/yars) |
| **ksanjeev284/reddit-universal-scraper** | Full-featured: CLI, analytics (sentiment, keywords), Discord/Telegram alerts, REST API, Streamlit dashboard, DB export. | **Medium — over-engineered for our needs but good reference for subreddit traversal patterns and data export.** | [GitHub](https://github.com/ksanjeev284/reddit-universal-scraper) |
| **rodneykeilson/ScrapiReddit** | API-free scraper with all sort/filter options, resilient caching, full media capture, JSON/CSV output. | **Low-Medium — caching and resilience patterns worth studying.** | [GitHub](https://github.com/rodneykeilson/ScrapiReddit) |
| **PRAW (official)** | Python Reddit API Wrapper. Handles OAuth, rate limits, comment tree expansion (`replace_more`), pagination. | **High — this is the library, not a tool to fork. Already specified in our implementation notes.** | [Docs](https://praw.readthedocs.io/) |

**Recommendation:** Use PRAW as the library for the Reddit connector (API-compliant, rate-limit-aware). Reference yars only if we need an API-key-free fallback for prototyping. Do NOT ship yars-style scraping in production — it bypasses Reddit's API contract.

#### Cross-Platform / Graph

| Project | What it does | Reuse potential | Link |
|---|---|---|---|
| **Data4Democracy/media-crawler** | Scrapy-based crawler that builds a **graph of media connections** across articles, Twitter, Reddit. Crawls N nodes deep in reference trees. Considered Neo4j + ElasticSearch for storage. | **Low-Medium — architecture reference for graph-oriented crawling.** 35 stars, not actively maintained, but the reference-tree crawling pattern and graph storage thinking are relevant to our interaction record model. | [GitHub](https://github.com/Data4Democracy/media-crawler) |
| **ScriptSmith/socialreaper** | Multi-platform scraping library: Facebook, Twitter, Reddit, YouTube, Pinterest, Tumblr. Unified API. | **Low — broad but shallow. Useful as API reference for platforms we haven't built yet (YouTube, Pinterest).** Not maintained recently. | [GitHub](https://github.com/ScriptSmith/socialreaper) |

**Overall build-vs-reuse decision:** We are building custom connectors (the typed record schema + evidence_pointer chain is unique to our pipeline). But we should fork/wrap existing tools where they save time on the API interaction layer, particularly LAION-AI for Discord and PRAW for Reddit. The normalization, schema enforcement, Iceberg write, and tombstone handling are all custom.

---

### Open Questions

1. **Reddit commercial license** — what's the timeline and cost? This is a hard blocker for using Reddit data in persona generation. Free tier is non-commercial only.
2. **X API budget ceiling** — pay-per-use pricing is unstable (lives in Developer Console, not API docs). What monthly spend cap? Drives topic coverage.
3. **Discord server access model** — own bot invited by server admins, or partnerships? Determines crawl surface. MESSAGE_CONTENT_INTENT verification required at scale.
4. **Twitch chat sampling** — popular channels = 100+ msg/sec. Full capture vs sampling? Sampling strategy impacts persona signal density.
5. **Approved web allowlist** — which owned / consented / public-permitted sources do we trust enough for MVP, and who signs off on source terms review?
6. **Facebook/IG app review** — notoriously slow. Start in Phase 1 even though connectors are Phase 3?
7. **Cross-platform aliasing scope** — Reddit explicitly prohibits matching with off-platform identifiers. Do we alias only across approved non-Reddit sources, or do we skip cross-platform entirely for v1?
8. **Profile snapshot cadence** — every crawl run, or only when delta detected? Storage vs freshness tradeoff.
9. **Persona eval ownership** — crawler provides evidence, synthesis generates personas, who owns the held-out recognition eval? Needs joint ownership.
10. **Tombstone vs hard-delete** — Reddit requires removing deleted content. Tombstone (keep metadata, hash body, redact text) satisfies the spirit. Legal sign-off needed.
