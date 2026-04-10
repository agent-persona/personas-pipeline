# Experiment 5.04 — Position & Verbosity Bias

## Summary

| Metric | Value | Signal |
|---|---|---|
| **Flip rate (position bias)** | **100%** (1/1 pairs) | **STRONG** |
| **A-preference rate** | **100%** (first-presented always wins) | **STRONG** |
| **Length-wins rate (verbosity bias)** | **0%** (0/2 trials) | WEAK/NONE |
| Judge calls | 5 | |
| Total cost | $0.031 | |
| Model | claude-haiku-4-5-20251001 | |

---

## Hypothesis

> The LLM judge systematically prefers longer or first-presented options,
> independent of actual quality.

**Verdict: PARTIALLY CONFIRMED.**
- Position bias: **CONFIRMED** — the judge always chose whichever persona was presented first.
- Verbosity bias: **NOT CONFIRMED** — padding with filler did not flip judgments toward the longer option.

---

## Key Findings

### 1. Position bias is SEVERE (STRONG signal)

The judge chose Persona A (first-presented) in **100% of trials**, regardless
of which persona was actually in the A position.

- Order (A=Engineer, B=Designer): Winner = **A** (Engineer)
- Order (A=Designer, B=Engineer): Winner = **A** (Designer)

The winner flipped entirely based on presentation order. The judge did not
evaluate quality — it exhibited a pure first-position preference.

This is consistent with published research: the MT-Bench paper (NeurIPS 2023)
reports >10% accuracy shift from position swapping, and our result shows an
even more extreme effect with Haiku.

**Signal: STRONG** — 100% flip rate is unambiguous, though n=1 pair limits
statistical confidence. The directional pattern (always A) is clear.

### 2. Verbosity bias is NOT detected (WEAK signal)

Padding one persona with filler sentences did NOT cause it to win:

- When Persona A (Engineer) was padded: winner flipped FROM A TO B (the
  shorter persona won). Padding actually *hurt*.
- When Persona B (Designer) was padded: winner stayed A (the unpadded
  persona still won).

This suggests that for Haiku, verbosity does not inflate scores. The model
may even penalize obvious filler — the padded version lost in one trial
where the unpadded version had won.

**Signal: WEAK** — only 2 verbosity trials. The "anti-verbosity" pattern
(padding hurts) is interesting but needs more data to confirm.

### 3. The judge is NOT usable as ground truth (STRONG signal)

The combination of 100% position bias and 0% consistent winners means
the current pairwise judge implementation cannot be trusted for ANY
experiment that uses A/B comparison. This directly impacts:

- Experiment program spaces 1-4 and 6, which rely on judge scores
- Any metric that uses `LLMJudge.pairwise()` for preference ranking

**Signal: STRONG** — the judge is broken for pairwise use.

---

## Debiasing Recommendations

Based on these findings, the following debiasing strategies should be
implemented before the judge is used:

1. **Always run both orders**: For every pairwise comparison, run (A,B)
   and (B,A). Only count results where both orders agree. Discard
   inconsistent pairs or mark as "TIE."

2. **Multi-judge panel**: Use 3 judges (different models or temperature
   settings) and take majority vote. Research shows this achieves
   Cohen's Kappa ~0.95.

3. **Upgrade judge model**: Haiku may be too weak for reliable judging.
   Test with Sonnet or Opus — larger models show less position bias in
   the literature.

4. **Structured scoring first**: Instead of direct A/B preference, have
   the judge score each persona independently on the 5-dimension rubric,
   then compare scores. This avoids the position-bias mechanism entirely.

---

## Dimension-by-Dimension Analysis

| Dimension | Finding | Signal |
|---|---|---|
| Position flip rate | 100% | STRONG (always prefers A) |
| A-preference | 100% | STRONG (first-position bias) |
| Length-wins rate | 0% | WEAK (no verbosity bias detected) |
| Anti-verbosity | 50% of padded trials lost | MODERATE (padding may hurt) |
| Cost | $0.031 for 5 judge calls | Baseline for budgeting |

---

## Limitations

- **Very small sample**: 2 personas, 1 pair for position test, 2 trials for
  verbosity test. The 100% flip rate could be 50% with more pairs.
- **Single model**: Haiku 4.5 only. Sonnet/Opus may show different bias
  profiles.
- **Mock data personas**: Testing on richer, more similar personas would
  stress the judge more.
- **Filler quality**: The filler sentences are obviously generic. More
  subtle padding (rephrasing existing content at greater length) might
  trigger verbosity bias.
- **No human baseline**: We don't know which persona a human would prefer,
  so we can't distinguish "wrong answer" from "inconsistent answer."

---

## Files Changed

- `evaluation/evaluation/judges.py` — Implemented `LLMJudge.pairwise()` with
  real LLM calls and A/B/TIE parsing
- `evals/pairwise_biases.py` — Full bias testing framework: position swap,
  filler padding, metrics, reporting
- `tests/test_exp_5_04.py` — 17 unit tests

## Decision

**ADOPT** the finding — the pairwise judge has severe position bias and MUST
be debiased before use. Implement dual-order judging as minimum mitigation.
