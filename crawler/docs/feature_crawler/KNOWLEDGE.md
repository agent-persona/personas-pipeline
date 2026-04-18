# Knowledge

## What we know

- Day-1 connectors in the PRD: GA4, GSC, HubSpot, Salesforce, Shopify, generic web crawl, SEMrush SERP
- PII scrubbing at the edge is a non-negotiable product trust requirement
- Downstream systems need traceable source metadata, not just cleaned payloads
- Discord crawl storage is easier to reason about when folders resolve to server + channel, not raw guild ID only

## What to research next

- Airbyte or Fivetran as long-tail connector fallback
- residential proxy strategy and crawl limits
- raw HTML retention vs parsed-only storage

## Durable notes

Add durable findings here or in `plans/presearch.md`, then update tasks or decisions when they change scope.

- 2026-04-09: added `knowledge-bases/discord-multi-account-crawling.md` as a high-level note on external large-scale Discord scrape patterns and why this repo should stay slow-by-default for now.
