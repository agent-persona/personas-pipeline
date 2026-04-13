# Discord Multi-Account Crawling

Date: 2026-04-09

## Why this note exists

We now have a working Discord crawl path. The next question is not "can it crawl" but "how do large operators widen coverage without collapsing their tokens or losing track of output folders?"

## What the research says

Reports on large Discord scraping operations describe a recurring architecture pattern:

- work is sharded across multiple accounts instead of one hot token
- each account keeps its own rate and health state
- server coverage is distributed so one account does not need to join everything
- blocked or expired accounts get rotated out of service

This is useful as market and operator context. It is not a recommendation for this repo's default behavior.

## Why it matters here

- our near-term need is coverage, not maximum throughput
- folder layout needs to stay understandable while crawl volume grows
- slow single-account runs are easier to inspect, replay, and debug
- policy and ToS risk rises sharply once account pooling becomes operational

## Repo stance for now

- default to one account and low request volume
- widen channel coverage gradually
- keep output partitioned by server and channel so later scaling does not destroy traceability
- if multi-account work ever becomes real, write a separate design doc first and require explicit policy review

## Research trail

- 404 Media reporting on Spy.pet
- DataBreaches and Malwarebytes summaries of the Spy.pet scrape
- `discum` on PyPI: https://pypi.org/project/discum/
- Discord support note on self-bots: https://support.discord.com/hc/en-us/articles/115002192352-Automated-User-Accounts-Self-Bots

## Affected features

- `feature_crawler`
- `feature_ingestion`
- `feature_knowledge-bases`

## Next action

Keep the runtime slow and channel-aware. Revisit a coordinator only after the single-account crawl path is stable and reviewed.
