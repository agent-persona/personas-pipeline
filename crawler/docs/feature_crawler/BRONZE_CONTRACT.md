# Bronze Contract

## Purpose

Define the first runnable contract between `feature_crawler` and downstream ingestion.

Canonical output lands as JSONL partition files under a local bronze root:

```text
<bronze-root>/
  web/
    <community_id>/
      <YYYY-MM-DD>/
        <crawl_run_id>.jsonl
  discord/
    <server-name>/
      <channel-name>_<channel-id>/
        <month>/
          <day>/
            <year>/
              <crawl_run_id>.jsonl
  reddit/
    <subreddit>/
      <YYYY-MM-DD>/
        <crawl_run_id>.jsonl
```

Each line is one canonical record.

Notes:

- Discord pathing is channel-aware so one crawl folder maps to one server + channel slice.
- Mixed multi-channel Discord runs fall back to `multi-channel/` under the server folder.
- Reddit currently uses the shared `<platform>/<community>/<date>/` layout.

## Record Types

- `account`
- `profile_snapshot`
- `community`
- `thread`
- `message`
- `interaction`
- `tombstone`

Each emitted record must include:

- `record_type`
- `platform`
- `crawl_run_id`
- type-specific IDs and timestamps
- `evidence_pointer`

## Policy Gate

The runtime refuses blocked sources before any records are written.

Collection basis:

- `owned`
- `consented`
- `public-permitted`
- `blocked`

`blocked` sources cannot run and cannot use fallback crawling.

## Current Runtime

First implementation ships:

- policy registry
- shared runtime package layout
- local JSONL sink
- shared cursor store
- runnable CLI
- approved web connector
- Discord connector
- Reddit OAuth connector

Not shipped yet:

- remote S3 / Iceberg sink
- `tombstone` emission
- Reddit deletion diffing

## Smoke

```bash
python3 -m feature_crawler.crawler.cli crawl-web \
  --url feature_crawler/tests/fixtures/approved_web_sample.html \
  --output-dir /tmp/crawler-out \
  --collection-basis owned \
  --allow-persona-inference
```

```bash
python3 -m feature_crawler.crawler.cli crawl-reddit \
  --subreddit python \
  --output-dir /tmp/crawler-out \
  --cursor-store /tmp/crawler-cursors
```
