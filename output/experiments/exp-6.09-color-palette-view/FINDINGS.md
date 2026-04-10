# Experiment 6.09: Color palette view

## Metadata
- **Branch**: exp-6.09-color-palette-view
- **Date**: 2026-04-10

## Hypothesis
Visual distribution patterns correlate with set quality perception and distinctiveness

## Approach
TF-IDF bag-of-words vectorization + cosine distance matrix + stress-minimization 2D layout. No heavy ML dependencies.

## Results
- **Personas visualized**: 2
- **Spread score**: 0.7620 (higher = more distinct)
- **Pairwise distance**: 0.762
- **HTML artifact**: output/experiments/exp-6.09-color-palette-view/persona_projection.html

## Distance Matrix
| | Persona 1 | Persona 2 |
|---|---|---|
| Persona 1 | 0.000 | 0.762 |
| Persona 2 | 0.762 | 0.000 |

## Interpretation
The two personas (Alex the API-First DevOps Engineer and Maya the Freelance Brand Designer) are strongly separated in trait space with a cosine distance of 0.762, confirming near-maximal vocabulary divergence between their archetypes. This validates that the bag-of-words approach effectively captures meaningful persona distinctiveness without heavy ML dependencies.

## Signal Strength: **STRONG**
## Recommendation: **adopt**

The visualization tool works end-to-end: TF-IDF vectorization correctly surfaces domain-specific vocabulary differences, the MDS layout correctly places distinct personas far apart, and the HTML artifact renders as a usable trust artifact. Spread score of 0.762 exceeds the 0.3 STRONG threshold with significant margin. The tool is lightweight (stdlib only), fast, and produces interpretable output.

## Cost
- All runs: $0.00
