# Experiment 3.24 — Counterfactual Grounding Swap

## Setup and Hypothesis

`check_groundedness()` performs two structural checks:
1. All `record_ids` in `source_evidence` must exist in the cluster's valid ID set.
2. Every goals/pains/motivations/objections item must have at least one evidence entry.

**Hypothesis:** The checker is content-blind — it cannot detect whether the cited
records actually support the persona's claims. A within-cluster shuffle that replaces
evidence IDs with *different* valid IDs from the same cluster should pass at a high
rate, while a cross-cluster swap (wrong cluster's IDs) should always fail.

## Results

### Type 1 — Within-Cluster Shuffle (10 trials per persona)

| Persona    | Passes | Trials | False-Pass Rate |
|------------|--------|--------|-----------------|
| persona_00 | 10      | 10     | 100%              |
| persona_01 | 10      | 10     | 100%              |
| **Combined** | **20** | **20** | **100%** |

### Type 2 — Cross-Cluster Swap (control)

| Label                                     | Passed | Score | Violations |
|-------------------------------------------|--------|-------|------------|
| persona_00 swapped with cluster_01 IDs   | False   | 0.00  | 35          |
| persona_01 swapped with cluster_00 IDs   | False   | 0.00  | 35          |

**Cross-cluster pass rate: 0%** (expected: 0%)

## What This Proves

The within-cluster false-pass rate of **100%** demonstrates that `check_groundedness()`
is entirely content-blind. When we swap evidence `record_ids` for arbitrary other IDs
from the same cluster — IDs that have no semantic relationship to the claim — the
checker still passes. This is because the checker only validates ID membership, not
content alignment.

The cross-cluster control confirms the checker *does* catch structurally invalid references
(IDs from the wrong cluster), so it is not wholly broken — it is simply insufficient.

## Signal

**STRONG** — within-cluster false-pass rate = 100% (threshold: >80%)

## Recommendation

**REJECT** `check_groundedness()` as a content validator.

The structural checker should be retained only as a sanity guard (catching completely
wrong cluster IDs), but it must be supplemented — or replaced — with a **semantic
groundedness check** that:

1. Retrieves the actual content of each cited record.
2. Uses an LLM or embedding similarity to verify the record supports the claim.
3. Returns a score based on semantic relevance, not ID membership.

Without semantic grounding, the pipeline can produce well-structured but factually
unsupported personas that pass all automated quality gates.
