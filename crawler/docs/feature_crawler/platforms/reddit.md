# Reddit Platform

## Scope

First Reddit pass supports two transports that map into the same canonical records:

- `oauth`: official Reddit API
- `public-json`: public Reddit page `.json` endpoints

Current crawl unit:

- community: subreddit
- thread: submission
- message: submission body or comment
- interaction: reply edge from one comment to another comment or OP

## Runtime shape

Code lives under:

- `feature_crawler/crawler/platforms/reddit/connector_api.py`
- `feature_crawler/crawler/platforms/reddit/runner.py`

Shared runtime stays in:

- `feature_crawler/crawler/core/`

## Env

Required:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`

Optional:

- `REDDIT_USER_AGENT`

## Smoke

```bash
python3 -m feature_crawler.crawler.cli crawl-reddit \
  --subreddit python \
  --output-dir feature_crawler/data \
  --auth-mode oauth \
  --sort new \
  --limit 25 \
  --comment-limit 128 \
  --cursor-store feature_crawler/data/cursors
```

No-key stopgap:

```bash
python3 -m feature_crawler.crawler.cli crawl-reddit \
  --subreddit webscraping \
  --output-dir feature_crawler/data \
  --auth-mode public-json \
  --sort new \
  --limit 25 \
  --comment-limit 128 \
  --cursor-store feature_crawler/data/cursors
```

## Policy notes

- crawl path is API-first and OAuth-authenticated
- public `.json` mode exists only as a stopgap while API approval is pending
- persona inference stays off for Reddit
- cross-linking stays off for Reddit
- commercial use still needs explicit Reddit approval/contract

## Gaps

- no deletion tombstones yet
- no rate-budget telemetry yet
- no media download path yet
- public `.json` mode can 429 or disappear; do not treat it as stable infra
