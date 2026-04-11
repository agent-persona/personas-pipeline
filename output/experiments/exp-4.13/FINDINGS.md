# exp-4.13 — Length Matching

**Branch:** `exp-4.13-length-matching`
**Guide:** Guide 2 — Synthesis Pipeline Architecture
**Date:** 2026-04-11
**Status:** WEAK-POSITIVE with mechanism caveat — realism rises monotonically across modes (`fixed` → `mirror` → `mirror_with_floor`) but the *causal mechanism* is confounded. At n=40 judgments per mode, the deltas (0.15–0.18 on a 5-point scale) are directionally consistent with Guide 2's hypothesis but well within judge-noise.

## Hypothesis

From Guide 2: *"matching [length] feels natural in chat but breaks when the user is deliberately terse"* — so a pure mirror collapses to one-word replies and reads as disengaged, but `mirror_with_floor` (mirror length, never below one full sentence) should beat both `fixed` and pure `mirror` on realism.

## Method

### Twin runtime change

Added `LengthMode = Literal["fixed", "mirror", "mirror_with_floor"]` and a `length_mode` parameter to `TwinChat.__init__` and `build_persona_system_prompt`. The single instruction line in the system prompt's `## Rules` section is substituted:

- **fixed** — `"Keep responses under 4 sentences unless asked to elaborate."` (current prod behavior)
- **mirror** — `"Match the user's message length. If they send a single word, answer in a single word. If they send a paragraph, match it with a paragraph. Length is a social signal — don't over-explain when they're terse, and don't under-explain when they're taking time to write."`
- **mirror_with_floor** — `"Match the user's message length, but never drop below one complete sentence. If they send a single word, still respond with at least one full sentence — short but complete. If they send a paragraph, match the paragraph. Length is a social signal; use it, but don't become a one-word bot."`

### Conversation design

A **20-turn scripted conversation** alternating verbose (V) and terse (T) user turns: V, T, V, T, ... (10 of each). Content is domain-agnostic so both personas engage. History is preserved turn-to-turn, so the model sees the unfolding dialogue — length-matching behavior should appear *within* a single conversation.

6 conversations total: 3 modes × 2 personas × 20 turns = 120 twin replies + 120 LLM-judge realism ratings (1–5 scale).

## Results

### Aggregate metrics

| Mode | Length r | Twin len (terse) | Twin len (verbose) | Realism (terse) | Realism (verbose) | Realism (overall) |
|---|---|---|---|---|---|---|
| `fixed` | 0.903 | 531 chars | 1305 chars | 3.95 | 3.85 | **3.900** |
| `mirror` | 0.905 | 448 chars | 1528 chars | 4.00 | 4.10 | **4.050** |
| `mirror_with_floor` | 0.757 | 885 chars | 1554 chars | 4.05 | 4.10 | **4.075** |

### Ranking by realism (overall)

1. **`mirror_with_floor`** — 4.075 (+0.175 over `fixed`)
2. **`mirror`** — 4.050 (+0.150 over `fixed`)
3. **`fixed`** — 3.900 (baseline)

Monotonic improvement across the three modes, directionally consistent with the Guide 2 hypothesis.

## Mechanism caveat (important)

The quantitative ranking supports the hypothesis, but **the mechanism is not what Guide 2 predicted**.

Guide 2's implicit model: `mirror_with_floor` should produce *shorter* replies to terse user turns (closer to user length than `fixed`, while avoiding the one-word collapse of pure `mirror`). That's the theory of why it would win.

Actual behavior:

| Mode | Mean twin reply length on TERSE user turns |
|---|---|
| `fixed` | 531 chars |
| `mirror` | **448 chars** (shorter — actually mirroring) |
| `mirror_with_floor` | **885 chars** (LONGER than `fixed` — model elaborated freely) |

The model interpreted `mirror_with_floor`'s instruction as *"you're allowed to elaborate"* rather than *"match length but don't collapse."* It produced the longest terse replies of any mode. So `mirror_with_floor` did not win by matching length; it won by giving the judge more character surface area to assess.

**Qualitative example** — user turn: *"Which tool do you open first?"* (29 chars)

| Mode | Twin reply (first 80 chars) | Full len |
|---|---|---|
| `fixed` | "Adobe Creative Cloud, honestly—that's where the actual work happens. But befo…" | 684 |
| `mirror` | "Figma. Always Figma.\n\nThat's where the actual work lives, so I check it first…" | 554 |
| `mirror_with_floor` | "Honestly? Slack or Gmail, depending on the client. Most of my clients are on…" | 1056 |

Notice: `mirror` led with a true one-word-ish opener ("Figma. Always Figma.") and then elaborated. This is the most "mirror-like" behavior of any mode. `mirror_with_floor` wrote the longest reply of all three. The instruction is being under-applied (the model treats the floor as a target, not as a minimum).

**Positive qualitative finding:** The explicitly-one-word probe ("One word: busy or calm?") produced the same 5-char reply ("Busy.") under all three modes. When the user names the length explicitly, all three modes obey. The difference only matters when the user *signals* terseness through actual length rather than by asking for it.

## Interpretation

Three possible readings of the ranking:

1. **Length-matching matters (Guide 2's claim).** The mirror and mirror_with_floor modes score higher because length-matching is a real realism signal, even though the effect is small.
2. **Longer replies look more realistic (confound).** The judge may reward replies that include more persona-specific detail, regardless of whether that detail was "asked for." Under this reading, `mirror_with_floor` wins because it wrote the longest text, not because it mirrored well.
3. **Noise.** A 0.15–0.18 delta on a 5-point scale at n=40 is barely distinguishable from random variance. The monotonic ordering is suggestive but not statistically robust.

Reading #2 is the most concerning because it would mean `fixed` (the current production behavior) could be beaten by a one-line instruction that just says "elaborate more" — no length-matching logic needed. The experiment as designed can't distinguish #1 from #2.

### A follow-up test that could distinguish

Run a fourth mode, `elaborate_always` — *"When the user sends a short message, always elaborate with at least 3 sentences. When they send a long message, also elaborate with at least 3 sentences."* — i.e., force long replies regardless of user length. If `elaborate_always` ties or beats `mirror_with_floor` on realism, the signal is confounded. If `mirror_with_floor` wins, the matching logic is load-bearing.

This would add ~$0.10 and isn't run here — flagged as follow-up.

## Recommendation

**SOFT ADOPT** `mirror` as the default in `twin/twin/chat.py`. Rationale:

- `mirror` beats `fixed` by +0.15 on realism while also genuinely producing shorter terse replies (real length-matching, not just elaboration).
- `mirror_with_floor`'s win is a mechanism-confounded +0.175 and requires the follow-up test above before adoption.
- `fixed` costs the most on realism and is the most bot-like in qualitative review.

**DEFER** final decision pending:

1. Run the `elaborate_always` confound test (see above).
2. Expand sample to ≥5 personas × ≥40 turns per mode so the monotonic ordering clears judge-noise.
3. Re-score with calibrated judge from exp-5.06 — all four realism means (3.90, 4.05, 4.08) are suspiciously close to `4`, which is a common judge-anchor value and suggests the judge's scale compression is eating our signal.

**Note on realism absolute values:** The ~3.90 baseline realism here is notably lower than exp-1.15's 4.80 in-character score. That gap is an artifact of judge-question framing ("realism" vs. "in-character adherence"), not a regression. The within-experiment ranking is the only comparable number.

## Cost

- Synthesis (2 personas): ~$0.057
- Twin conversations (120 turns): ~$0.105
- LLM-judge (120 judgments): ~$0.067
- **Total: ~$0.229**
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `conversations.json` — full 20-turn transcripts for all 6 (mode × persona) combinations
- `realism_scores.json` — per-turn realism ratings
- `summary.json` — aggregate metrics
- `scripts/run_exp_4_13.py` (in branch)
- `twin/twin/chat.py` (in branch) — `LengthMode`, `_LENGTH_RULES`, `length_mode` param
