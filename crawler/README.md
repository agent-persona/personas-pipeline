# crawler

**Stage 1.** Pull behavioral records from a tenant's data sources and normalize
them into a single `Record` shape that segmentation can read.

## What lives here

```
crawler/
├── crawler/
│   ├── base.py           # Connector ABC — extend to add a source
│   ├── adapter.py        # Vendored bronze records -> flat segmentation records
│   ├── feature_crawler/  # Vendored real crawler runtime and CLI
│   ├── models.py         # Record — the normalized output contract
│   ├── pipeline.py       # fetch_all()/fetch_from_run()
│   └── connectors/
│       ├── ga4.py        # Mock GA4 events (engineer + designer clusters)
│       ├── intercom.py   # Mock Intercom messages (lift into sample_quotes)
│       └── hubspot.py    # Mock HubSpot contacts (firmographic context)
├── config/
│   └── crawl_jobs.json   # Vendored runtime scheduling config
├── docs/
│   └── feature_crawler/  # Vendored runtime docs
├── scripts/
│   └── feature_crawler/  # Vendored helper scripts
└── tests/
    ├── feature_crawler/  # Vendored runtime tests
    └── test_adapter.py   # Segmentation seam tests
├── .env.example
└── pyproject.toml
```

## I/O contract

Input: `tenant_id: str` (and optional `since` ISO timestamp).

Output: `list[Record]` where `Record` is:

```python
class Record(BaseModel):
    record_id: str          # globally unique
    tenant_id: str
    source: str             # "ga4" | "intercom" | "hubspot" | ...
    timestamp: str | None
    user_id: str | None     # ties records for the same person together
    behaviors: list[str]    # normalized behavior tags — this is what clustering keys on
    pages: list[str]
    payload: dict           # source-specific data, preserved verbatim
    schema_version: str = "1.0"
```

This shape is byte-compatible with `segmentation.models.RawRecord`. The
top-level script round-trips via `model_dump()` to avoid a cross-package
import.

## How to run standalone

```python
from crawler import fetch_all
records = fetch_all(tenant_id="tenant_acme_corp")
print(len(records), records[0])
```

## Real crawler compatibility

The monorepo now vendors the standalone runtime under `crawler.feature_crawler`.
It still emits bronze-style `account`, `thread`, `message`, `community`,
`interaction`, `profile_snapshot`, and `tombstone` records.

To keep segmentation unchanged, use the adapter seam:

```python
from crawler import fetch_from_run

records = fetch_from_run(
    "output/runs/linkedin-demo",
    tenant_id="tenant_acme_corp",
)
```

`fetch_from_run()` accepts either:

- one mixed `.jsonl` file written by `crawler.feature_crawler.core.sink.JsonlSink`
- a bronze run directory containing one or more `*.jsonl` files

The adapter preserves the existing flat `Record` contract by deriving:

- `user_id` from bronze actor ids
- `behaviors` from thread/message/profile heuristics
- `pages` from synthetic conversation/profile paths plus evidence URLs
- `payload` as the original source-specific bronze payload

## Knobs you can turn

- **Add a connector.** Subclass `Connector`, implement `fetch()`, return a
  list of `Record`s, register it in `crawler/pipeline.py::DEFAULT_CONNECTORS`.
- **Use the vendored runtime.** Run `python -m crawler.feature_crawler.cli ...`
  to write real JSONL output, then adapt it back with `fetch_from_run()`.
- **Swap fixtures.** The three connectors ship with in-file mock data.
  Replace the constants (`_GA4_EVENTS`, `_INTERCOM_MESSAGES`, `_HUBSPOT_CONTACTS`)
  with larger fixtures to stress segmentation, or point them at disk files.
- **Per-connector filtering.** `fetch()` takes a `since` timestamp for
  incremental pulls (unused by mocks, required for real connectors).

## Scientific backing

Crawler is mostly **infrastructure** for the iteration program — rarely the
variable under test. Two problem spaces have exercised it directly:

- **Space 3 — Groundedness (sparse-data ablation, adversarial injection).**
  The connector fixtures get downsampled to simulate a low-data tenant, or
  seeded with false records to measure whether synthesis absorbs them. The
  default fixture is the control.
- **Space 6 — Population coverage (cross-tenant leakage).** Alternate
  fixture sets per tenant_id drive the same pipeline over multiple
  "tenants" to check whether generic LLM patterns dominate.

New fixtures live alongside the existing ones (e.g. a new
`_GA4_EVENTS_FRESH` constant, or a new connector file) rather than
replacing them — prior controls still reference the originals.

## Tests

The original repo ships `crawler/tests/` with fixture fakes and pytest tests;
they weren't copied into this lab bundle to keep the surface small. If you
need them, copy them from the parent repo at `crawler/crawler/tests/`.
