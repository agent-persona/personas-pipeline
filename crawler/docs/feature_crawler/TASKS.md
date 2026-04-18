# Tasks

| ID | Owner | Status | Updated | Task | Exit criteria |
| --- | --- | --- | --- | --- | --- |
| CRW-001 | Max | done | 2026-04-09 | Decide connector framework boundaries vs build-vs-buy spike | Written decision in `DECISIONS.md` |
| CRW-002 | Max | done | 2026-04-09 | Define bronze record contract and PII scrub boundary | Contract documented and linked from README |
| CRW-003 | Max | done | 2026-04-09 | Confirm day-1 connector order | Prioritized connector list added to plan |
| CRW-004 | Max | done | 2026-04-09 | Ship first working connector under policy gate | `approved_web` connector + CLI + tests live in `feature_crawler/crawler` |
| CRW-005 | Max | done | 2026-04-09 | Add Discord connector on same canonical models | Discord bot/user/browser/archive paths write the same record families |
| CRW-006 | Max | done | 2026-04-09 | Add cursor persistence beyond CLI run context | Cursor store survives process restarts |
| CRW-007 | Max | done | 2026-04-09 | Make Discord crawl output channel-aware and document the first run | Discord writes under server/channel/date folders and docs reflect the first crawl |
| CRW-008 | Max | done | 2026-04-09 | Refactor crawler into shared core plus per-platform packages | `crawler/core` and `crawler/platforms/<platform>` are the main runtime entrypoints |
| CRW-009 | Max | done | 2026-04-09 | Add first Reddit crawler on the shared contract | `crawl-reddit` lands canonical records from the official OAuth API |
| CRW-010 | Max | done | 2026-04-11 | Add first LinkedIn profile crawler on the shared contract | `crawl-linkedin` emits canonical profile records from public HTML or session-backed profile pages |
| CRW-011 | Max | done | 2026-04-11 | Expand LinkedIn into auth and vendor-backed modes | OIDC auth helpers, code exchange, vendor transports, cursor-aware activity/network crawl paths are documented and tested |
