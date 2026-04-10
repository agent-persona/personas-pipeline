# Experiment 6.14: Persona graph relationships

## Metadata
- **Branch**: exp-6.14-persona-graph-relationships
- **Date**: 2026-04-10
- **Problem Space**: 6

## Hypothesis
Graph topology quantifies population-level coherence and reveals persona clustering patterns

## Changes Made
- evals/persona_graph.py: Shared-trait graph builder with density and normalized edge weight metrics

## Results

### Graph Structure
- **Nodes**: 2 personas (Alex the API-First DevOps Engineer, Maya the Freelance Brand Designer)
- **Edges**: 1 edge (shared traits)
- **Shared traits**: build, delivery, experience, long, manual, many, minutes, project, rate, requires, sessions, slack, steps, workflow
- **Total unique traits**: 322

### Target Metric: Graph density
| Metric | Value | Interpretation |
|---|---|---|
| shared_trait_count | 14 | terms shared between personas |
| total_unique_traits | 322 | total vocabulary across both |
| **graph_density** | **0.0435** | **fraction of traits shared** |
| normalized_edge_weight | 0.0836 | geometric-mean normalized |

### Cross-validation with exp-6.09
- exp-6.09 cosine distance: 0.762 (STRONG spread)
- exp-6.14 graph_density: 0.0435 (very low — personas are highly distinct)
- Consistent? **Yes** — both measures independently confirm strong separation. A cosine distance of 0.762 means vectors share little directional overlap; a graph density of 0.044 means only 4.4% of total trait vocabulary is shared. The two methods use different mathematical lenses (angle vs. set overlap) and arrive at the same conclusion.

### Trait breakdown
The 14 shared terms are largely generic workflow vocabulary: `build`, `delivery`, `experience`, `long`, `manual`, `many`, `minutes`, `project`, `rate`, `requires`, `sessions`, `slack`, `steps`, `workflow`. None are domain-specific — Alex owns `webhook`, `graphql`, `terraform`, `idempotent`, `iac`, `gitops`; Maya owns `brand`, `white-label`, `typeface`, `deliverable`, `moodboard`, `revision`. The overlap is noise-level generic language, not genuine conceptual overlap.

## Signal Strength: **STRONG** (density < 0.1)
## Recommendation: **adopt**
Graph density provides a fast, interpretable distinctiveness score that requires no embedding model. At 0.044 it strongly confirms the TF-IDF cosine result (0.762 distance) from exp-6.09, adding a complementary set-theoretic view. The metric scales naturally to N>2 personas as a population-level coverage score. Shared-trait inspection also surfaces exactly what vocabulary personas have in common, which is actionable for refining trait extraction prompts.

## Cost
- All runs: $0.00
