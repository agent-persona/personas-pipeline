# Experiment 3.22: Domain-specific grounding rules

## Metadata
- **Branch**: exp-3.22-domain-specific-grounding-rules
- **Date**: 2026-04-10
- **Problem Space**: 3

## Hypothesis
Strict evidence enforcement reduces false-positive grounding in sensitive claim categories

## Changes Made
- synthesis/synthesis/engine/domain_rules.py: Keyword-based sensitive claim classifier with N+1 source enforcement (>=2 unique record_ids required for flagged claims)

## Results

### Target Metric: High-risk flagging rate
| Cluster | Claims checked | Sensitive flagged | Flagging rate | N+1 satisfied |
|---|---|---|---|---|
| cluster_00 (Alex DevOps) | 38 | 3 | 0.079 | 2/3 |
| cluster_01 (Maya Designer) | 37 | 5 | 0.135 | 0/5 |
| **Average** | | | **0.107** | **2/8** |

**Note**: 'flagging rate' = sensitive claims flagged / total claims checked. This is not a traditional false-positive rate (no true negatives are tracked).

### Category breakdown
| Category | Count across both personas |
|---|---|
| financial | 5 |
| security | 0 |
| medical | 0 |
| legal | 3 |

### Qualitative analysis

**cluster_00 (Alex, DevOps/API persona):** All 3 flagged claims are legitimately in the "legal" category and are appropriate for a fintech B2B context — audit requirements, compliance-critical webhooks, and SLA expectations are real concerns for this persona. The "legal" keyword triggers on "compliance" and "audit" which are accurate matches. 2/3 satisfied N+1 (have 2+ backing records), the third (decision_triggers.3 — SLA mention) has 0 sources because decision_triggers fields are not covered by source_evidence in the current schema.

**cluster_01 (Maya, Freelance Designer):** All 5 flagged claims are in the "financial" category, triggered by "pay", "cost", "money" variants. These are billing/pricing concerns about the SaaS subscription and hourly rate comparisons — real and appropriate for a cost-sensitive solo freelancer. 0/5 satisfy N+1 because sample_quotes and decision_triggers have no field_path entries in source_evidence.

**Overall pattern:** The keyword classifier produces no false positives in the traditional sense — every flagged claim genuinely touches a sensitive concept. The 0.107 flagging rate represents real sensitivity density in B2B SaaS personas. The low N+1 satisfaction rate (2/8) is a structural artifact: decision_triggers and sample_quotes are not routinely backed by field_path-specific source_evidence entries.

## Signal Strength: **MODERATE**
## Recommendation: **defer**

The keyword classifier is sound and correctly identifies sensitive claim categories with no spurious keyword matches. Two issues block adoption:
1. **Coverage gap**: decision_triggers and sample_quotes are not covered by source_evidence field paths, making N+1 enforcement structurally impossible for those fields without schema changes.
2. **Context blindness**: "financial" triggers on billing/pricing language appropriate for a SaaS subscription context. A domain-aware allowlist would reduce unnecessary N+1 pressure on pricing-adjacent language that is well-evidenced at the persona level.

## Cost
- All runs: $0.00
