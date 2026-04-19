# Tail Insight Detection (D-NEW) — Design Document

> **Status:** Design only — no implementation yet.

## Problem Statement

**"Tail insights"** are rare, non-obvious findings that emerge from qualitative research — the unexpected patterns, edge cases, and minority experiences that standard analysis misses. They're the reason organizations do qualitative research instead of relying solely on surveys: a single interview participant revealing a workflow workaround can reshape a product strategy.

**The gap:** In every practitioner evaluation (Speero, NN/g), synthetic personas achieved 0% detection of tail insights. LLM-generated personas reproduce majority patterns well but systematically miss the surprising, low-frequency observations that make qualitative research valuable. There is also zero academic measurement of this phenomenon — no paper has proposed a metric or benchmark for tail insight fidelity.

**Why it matters:** If synthetic personas can't surface tail insights, their utility is limited to confirming what we already suspect. This is the #1 gap identified in our research roadmap — the difference between "synthetic personas are a useful supplement" and "synthetic personas can replace some qualitative research."

## Definitions

- **Tail insight:** A finding that is (a) true/valid, (b) non-obvious given the persona's demographic profile, and (c) not derivable from the majority distribution of responses.
- **Non-obvious:** Cannot be predicted from demographic attributes alone (age, income, location, etc.) — requires the lived-experience detail that qualitative methods capture.
- **Ground truth:** Real qualitative research findings annotated by domain experts as "tail" vs "expected."

## Three Experimental Approaches

### Approach 1: Paired Real/Synthetic Comparison

**Method:** Run identical research questions against (a) real interview transcripts and (b) synthetic persona responses. Have blinded domain experts extract insights from each. Compare the sets.

**Mechanics:**
1. Source 10–20 real qualitative studies with documented "key findings"
2. Generate synthetic personas matching the study's participant demographics
3. Run the same interview protocol on synthetic personas
4. Expert panel blind-codes insights from both sources
5. Compute: What % of real tail insights appear in synthetic output?

**Strengths:** Highest ecological validity — measures exactly what practitioners care about.
**Weaknesses:** Expensive (expert annotation), small N, depends on study selection.

### Approach 2: Retrospective Insight Extraction

**Method:** Given a corpus of known tail insights (from published case studies), test whether synthetic personas can reproduce them when prompted with the right context.

**Mechanics:**
1. Curate a set of 50–100 documented tail insights from published UX/market research
2. For each insight, create a "setup context" (the research question + participant profile)
3. Generate synthetic persona responses to the setup context
4. Score whether the tail insight appears in the synthetic response (semantic similarity threshold)

**Strengths:** Scalable, repeatable, no expert annotation per run.
**Weaknesses:** Hindsight bias — knowing the insight exists changes what "non-obvious" means.

### Approach 3: Benchmark Suite with Planted Insights

**Method:** Create synthetic research scenarios with planted tail insights at known positions. Test whether persona-based analysis surfaces them.

**Mechanics:**
1. Design 20–30 research scenarios with 3–5 planted insights each (mix of obvious and tail)
2. Each scenario has: research question, participant profiles, expected majority responses, and planted tail insights
3. Generate synthetic persona responses for each scenario
4. Score: How many planted tail insights surface? How many false positives?

**Strengths:** Full control over ground truth, reproducible, supports precision/recall measurement.
**Weaknesses:** Artificiality — planted insights may not capture the full difficulty of real-world tail detection.

## Data Requirements

### Ground Truth Needed

| Source | Description | Target Size |
|--------|-------------|-------------|
| Published case studies | Real qualitative findings annotated as tail/expected | 50–100 insights |
| Expert panel annotations | Blinded coding of insight novelty (1–5 scale) | 3+ annotators per insight |
| Demographic baselines | Expected response distributions for common research questions | 20–30 question domains |
| Planted benchmark suite | Synthetic scenarios with known tail insight positions | 20–30 scenarios |

### Annotation Schema

Each ground-truth insight needs:
```json
{
  "insight_id": "string",
  "source_study": "string",
  "insight_text": "string",
  "research_question": "string",
  "participant_demographics": {},
  "novelty_score": 1-5,        // 1=obvious from demographics, 5=completely unexpected
  "prevalence": 0.0-1.0,       // fraction of participants who exhibited this
  "domain": "string",          // e.g., "healthcare UX", "financial behavior"
  "annotator_agreement": 0.0-1.0
}
```

## Scorer Interface Design

The tail insight scorer fits the `BaseScorer` pattern as a **set-level scorer** (insights emerge from patterns across multiple personas, not individual responses).

```python
class TailInsightDetectionScorer(BaseScorer):
    """D-NEW: Measures synthetic personas' ability to surface tail insights."""

    dimension_id = "D-NEW"  # TBD final assignment
    dimension_name = "Tail Insight Detection"
    tier = 6  # highest tier — requires curated benchmark data
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D-NEW is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        """
        Expects source_contexts to carry:
        - extra_data["tail_insights_benchmark"]: list of planted/known insights
        - extra_data["persona_responses"]: synthetic responses to score against

        Returns: recall, precision, novelty-weighted F1 for tail insight detection.
        """
        ...
```

### Expected `extra_data` Format

```python
# On source contexts:
{
    "tail_insights_benchmark": [
        {
            "insight_id": "ti_001",
            "insight_text": "Users with chronic conditions prefer voice interfaces not for convenience but because screen fatigue from medical portals causes them to skip refills",
            "novelty_score": 5,
            "prevalence": 0.08,
            "semantic_threshold": 0.75,  # min similarity to count as "detected"
        },
        ...
    ],
    "persona_responses": [
        {
            "persona_id": "p1",
            "response_text": "...",
        },
        ...
    ]
}
```

### Output Details

```python
{
    "tail_insight_recall": 0.15,      # fraction of tail insights detected
    "tail_insight_precision": 0.60,    # fraction of claimed insights that were real tails
    "novelty_weighted_recall": 0.08,   # recall weighted by novelty score
    "detected_insights": ["ti_003"],   # which benchmark insights were found
    "missed_insights": ["ti_001", "ti_002", ...],
    "false_discoveries": [...],        # insights claimed but not in benchmark
    "detection_by_novelty": {          # recall broken down by novelty tier
        "low (1-2)": 0.40,
        "medium (3)": 0.15,
        "high (4-5)": 0.00,
    },
}
```

## Measurement Methodology

### Scoring Insight Detection

An insight is "detected" when any synthetic persona response has **semantic similarity ≥ threshold** to the benchmark insight text. Steps:

1. Embed all benchmark insight texts using the project Embedder
2. Embed all persona response texts
3. For each benchmark insight, find the max similarity across all persona responses
4. If max_sim ≥ semantic_threshold (default 0.75), mark as detected

### Novelty-Weighted Scoring

Not all tail insights are equally hard to detect. Weight recall by novelty score:

```
novelty_weighted_recall = Σ(detected_i × novelty_i) / Σ(novelty_i)
```

This ensures the scorer rewards detecting truly surprising insights more than semi-obvious ones.

### False Discovery Rate

Persona responses that don't match any benchmark insight but contain "novel" claims (high self-reported confidence + low similarity to majority responses) are flagged as potential false discoveries — fabricated insights that sound non-obvious but aren't grounded in real data.

## Benchmark Dataset Specification

### Format

```
benchmark/
  tail_insights/
    scenarios/
      scenario_001.json    # research question + participant profiles
      scenario_002.json
      ...
    insights/
      insights_001.json    # annotated tail insights for scenario_001
      insights_002.json
      ...
    baselines/
      majority_responses_001.json  # expected majority response distribution
      ...
```

### Size Targets

- **Minimum viable:** 10 scenarios × 5 insights each = 50 annotated insights
- **Target:** 30 scenarios × 5 insights each = 150 annotated insights
- **Stretch:** 50 scenarios across 10 domains = 250+ insights

### Curation Criteria

1. **Diversity of domains:** Healthcare, finance, education, consumer tech, B2B SaaS (min 5 domains)
2. **Novelty distribution:** ~30% low novelty (1-2), ~40% medium (3), ~30% high (4-5)
3. **Prevalence range:** Insights should span 1%–15% prevalence (above 15% isn't really "tail")
4. **Annotator agreement:** Only include insights with inter-annotator agreement ≥ 0.6 (Krippendorff's alpha)
5. **Source authenticity:** All insights must trace to published research, not synthesized examples

## Known Limitations and Open Questions

### Limitations

1. **No established ground truth exists.** We must create the benchmark dataset ourselves, introducing potential bias in what we consider a "tail insight."
2. **Semantic similarity is a proxy.** An insight can be "detected" by embedding similarity but miss the causal mechanism that makes it actionable.
3. **Domain dependence.** A scorer trained/benchmarked on UX insights may not transfer to market research or policy analysis.
4. **Threshold sensitivity.** The semantic similarity threshold (0.75) is an educated guess — too low gives false positives, too high misses paraphrased detections.

### Open Questions

1. **How to handle partial detection?** A persona might surface half of an insight's mechanism (e.g., "users prefer voice" without "because of screen fatigue from medical portals"). Binary detected/not-detected loses this nuance.
2. **Should we measure insight generation or insight recognition?** Generation (open-ended prompting) is harder but more realistic. Recognition (given the scenario, does the persona mention X?) is easier to score.
3. **What's the right baseline?** 0% detection (current state) makes any improvement look good. Should we compare against GPT-4 without persona framing?
4. **Is there a minimum persona set size for tail insights to emerge?** Real qualitative research typically needs 15–30 participants. Does synthetic research need more?
5. **How to distinguish "creative hallucination" from "genuine tail insight"?** An LLM might generate novel-sounding claims that have no basis in reality.

## Implementation Roadmap

### Phase 1: Benchmark Construction (prerequisite)
- Curate 10 scenarios from published UX/market research case studies
- Expert-annotate 50 insights (5 per scenario) with novelty scores
- Build baseline majority response distributions for each scenario
- Estimated effort: 2–3 weeks (annotation bottleneck)

### Phase 2: Minimal Scorer (Approach 2 — Retrospective)
- Implement `TailInsightDetectionScorer` with semantic similarity matching
- Score against the 50-insight benchmark
- Establish baseline: what's the current detection rate?
- Estimated effort: 1 week

### Phase 3: Planted Benchmark (Approach 3)
- Design 20 synthetic scenarios with planted insights
- Implement precision/recall and novelty-weighted metrics
- Add false discovery detection
- Estimated effort: 2 weeks

### Phase 4: Paired Comparison (Approach 1)
- Partner with a qualitative research team for real/synthetic comparison
- Requires IRB or data sharing agreement for real participant data
- Full blinded evaluation protocol
- Estimated effort: 4–6 weeks (external dependency)

### Phase 5: Refinement
- Calibrate semantic similarity thresholds using Phase 2/3 data
- Add partial detection scoring
- Explore generation vs. recognition modes
- Integrate with the full persona eval pipeline

---

**Recommendation:** Start with Phase 1 + 2 (benchmark + retrospective scorer). This gives us a measurable baseline with minimal infrastructure. Phase 3 adds rigor. Phase 4 is the gold standard but requires external collaboration.
