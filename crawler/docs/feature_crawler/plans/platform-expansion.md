# Platform Expansion Plan

## Objective

Repeat the Discord pattern for other platforms without rebuilding the crawler from scratch each time.

Target outcome:

- one repeatable connector playbook
- one shared bronze contract
- one per-platform rollout checklist
- one place to decide when a platform deserves special runtime treatment

## Short answer on architecture

Keep `feature_crawler` as a single feature.

Do not create a separate top-level `feature_discord_crawler` right now.

Use this shape instead:

- shared runtime in `feature_crawler/crawler/core/`
- one package per platform in `feature_crawler/crawler/platforms/<platform>/`
- thin compatibility imports can remain in `feature_crawler/crawler/connectors/` while older callers are cleaned up
- platform-specific docs under `feature_crawler/docs/platforms/<platform>.md`
- platform-specific runner scripts only when the platform needs special auth or crawl orchestration

Split Discord into its own top-level feature only if at least one becomes true:

- Discord needs its own queue workers or infra lifecycle
- Discord needs auth/session handling too different from every other source
- Discord-specific code becomes larger than the rest of crawler combined
- platform-specific testing or release cadence starts conflicting with shared crawler work

Right now none of those are strong enough. The shared bronze contract is the valuable part. Keep that central.

## What to copy from the Discord rollout

For every new platform, ship the same sequence:

1. policy row
2. auth note
3. connector
4. CLI entrypoint
5. local runner script if needed
6. output partition rule
7. first-run handoff note
8. regression tests
9. slow-crawl defaults before scale

If any platform skips one of these, it is not actually production-shaped.

## Platform template

Each platform should answer these before code:

- community unit: what is the top crawl scope
- thread unit: what is the conversation container
- message unit: what is the atomic utterance
- identity anchor: what user ID is stable enough
- auth path: bot token, OAuth, cookie/session, public fetch, archive
- crawl mode: API, browser, archive, webhook, export
- incremental cursor: timestamp, message id, page cursor, event stream
- delete/edit detection: native, best-effort, none
- folder layout: what should a human see on disk
- legal/policy status: owned, consented, public-permitted, blocked

## Recommended order after Discord

### 1. Twitch

Why first:

- good behavioral signal density
- stable public APIs and chat surfaces
- cleaner policy path than Reddit or Meta

Ship:

- channel metadata
- stream-session thread model
- chat message capture
- stable Twitch user IDs

Folder proposal:

```text
feature_crawler/data/twitch/<channel-name>/<stream-date>/<crawl_run_id>.jsonl
```

### 2. Reddit

Why second:

- strong persona signal
- mature thread structure
- but policy/commercial constraints need heavier review

Shipped now:

- official OAuth API connector
- subreddit metadata
- submission threads
- flattened comment replies
- shared cursor-store integration

Still to harden:

- deletion/tombstone diffing
- commercial approval path
- stricter rate-budget observability

Folder proposal:

```text
feature_crawler/data/reddit/<subreddit>/<topic-date>/<crawl_run_id>.jsonl
```

### 3. Generic forum / Discourse / approved web boards

Why:

- closest to what already works
- low architectural risk

Focus:

- better thread extraction
- stable author keys
- per-site allowlist docs

### 4. YouTube or community comment surfaces

Why later:

- signal can be useful
- but identity quality and thread quality are weaker than Discord/Reddit/Twitch

## Shared connector checklist

Before merge:

- connector emits canonical records only
- output path is human-browsable
- no secrets in docs
- no raw crawl data committed
- one smoke command
- one real-world first-run example in handoff
- tests cover connector shape and sink pathing

## Code shape to keep

Keep:

- `crawler/core/models.py`
- `crawler/core/base.py`
- `crawler/core/sink.py`
- `crawler/core/cursor_store.py`
- `crawler/core/policy.py`
- `crawler/cli.py`

Add per platform:

- `crawler/platforms/<platform>/connector_api.py` when the source is API-first
- `crawler/platforms/<platform>/connector_browser.py` only if needed
- `crawler/platforms/<platform>/runner.py`
- `tests/test_<platform>_connector.py`

Avoid:

- platform-specific record formats
- platform-specific top-level storage roots outside `feature_crawler/data`
- one-off scripts that bypass `cli.py` and `sink.py`

## Decision rule: single vs separate

Use one feature if:

- bronze schema is shared
- sink and policy enforcement are shared
- most complexity is connector-specific, not product-specific

Create a separate feature only if:

- the platform needs its own ingestion contract
- the platform needs its own deployable service
- the platform needs different auth, storage, and operational ownership

## Immediate next docs to write

1. `docs/platforms/twitch.md`
2. `docs/platforms/approved-forums.md`
3. per-platform first-run checklist

## Immediate next code plan

1. harden shared cursor persistence beyond Reddit
2. formalize per-platform output path helpers in sink
3. add Twitch as the next full connector using the same package shape
