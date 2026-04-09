"""Golden tenant loader.

`PRD_LAB_RESEARCH.md` names the golden set as a prerequisite for every
experiment:

    "Frozen golden set: 20 tenants spanning B2B SaaS, e-commerce, agency,
     research, consumer app — each with sealed source data, hand-curated
     'ideal personas,' and a labeled regression set."

Until the real golden set lands, this module exposes a single stub tenant
(the mock crawler fixtures for `tenant_acme_corp`) so experiments have
something to run against on day one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class GoldenTenant:
    """One frozen tenant in the golden set.

    - `tenant_id`: stable identifier, used as the primary join key for
      experiment results.
    - `industry` / `product_description`: passed through to synthesis via
      cluster summaries.
    - `source_fixture_name`: key into the crawler fixture registry, so each
      tenant can hand off its own fixture to `crawler.fetch_all`.
    - `ideal_personas`: hand-curated "ground truth" persona dicts a judge
      or human eval compares against. Empty until researcher #5 fills it.
    - `regression_checks`: labeled input→expected-output pairs for the
      regression suite (e.g. "the loyalist cluster must appear at
      threshold=0.15").
    """

    tenant_id: str
    industry: str
    product_description: str
    source_fixture_name: str = "default"
    ideal_personas: list[dict] = field(default_factory=list)
    regression_checks: list[dict] = field(default_factory=list)


# TODO(space-5): expand this to 20 tenants with real sealed source data and
# hand-curated ideal personas. Today it's a single stub so experiment code
# has something to import.
_GOLDEN_STUB: list[GoldenTenant] = [
    GoldenTenant(
        tenant_id="tenant_acme_corp",
        industry="B2B SaaS",
        product_description="Project management tool for engineering teams",
        source_fixture_name="default",
    ),
]


def load_golden_set() -> Sequence[GoldenTenant]:
    """Return the frozen golden tenants.

    Every experiment in every problem space should iterate this list rather
    than constructing tenants ad-hoc. That's how we guarantee the control
    run and the variant run see the same source data.
    """
    return tuple(_GOLDEN_STUB)
