# Decisions

## 2026-04-09

### Decision

Build the first crawler slice as a stdlib-only Python package inside `feature_crawler/crawler`.

### Rationale

- repo already uses Python for local scripts and tests
- no root package manager exists yet
- fastest path to a working connector without cross-folder setup churn

### Tradeoff

- not production packaging yet
- no async/runtime infra yet

### Follow-up tasks

- add Discord connector
- add cursor persistence
- add `interaction` and `tombstone` emitters where source supports them

## 2026-04-09

### Decision

Ship approved web crawling first, with policy gate enforced before fetch.

### Rationale

- matches the current PRD shift toward approved sources
- gives one working connector now
- keeps blocked platforms blocked in code, not just docs

### Tradeoff

- does not yet prove Discord API integration
- approved web connector is page/paragraph oriented, not threaded discussion aware

### Follow-up tasks

- implement Discord connector
- expand web parsing for threaded comments when source structure is explicit

## 2026-04-09

### Decision

Partition Discord output by server name and channel folder, not guild ID alone.

### Rationale

- first crawl output needs to be human-browsable
- channel-level folders make QA and replay easier
- future wider Discord coverage should not collapse unrelated channel slices into one folder

### Tradeoff

- Discord path shape now differs from the simpler web partition layout
- mixed multi-channel runs need a fallback folder name

### Follow-up tasks

- add cursor persistence
- decide whether multi-channel CLI runs should split into one sink write per channel

## Record format

When a decision is made, add:

- date
- decision
- rationale
- tradeoff
- follow-up tasks
