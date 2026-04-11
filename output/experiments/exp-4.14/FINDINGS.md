# exp-4.14 — Latency vs Realism Tradeoff (PARTIAL)

**Branch:** `exp-4.14-latency-vs-realism`
**Guide:** Guide 2 — Synthesis Pipeline Architecture
**Date:** 2026-04-11
**Status:** **PARTIAL — MECHANISM SHIPPED, REALISM SIGNAL BLOCKED ON exp-5.06.** The `artificial_delay_ms` parameter is implemented and verified to inject delay accurately. The realism question this experiment was designed to answer cannot be resolved without human raters.

## Hypothesis

From Guide 2: response speed is a perceived realism variable. Real humans take 1–6s to type a short reply; Claude returns in ~500ms. A 0ms response reads as bot-like. **Predicted curve:** realism peaks around 1500–3000ms, drops at 0ms (too fast) and at 6000ms (frustration).

## Method

### Twin runtime change

Added `artificial_delay_ms: int | Callable[[str, int], int] | None` to
`TwinChat.__init__`. Implementation:

```python
if delay_ms > 0:
    await asyncio.sleep(delay_ms / 1000.0)
```

Sleep happens *after* the model returns and *before* the `TwinReply` is
yielded to the caller, so the caller experiences `model_latency + delay`
as a single perceived response time. The new `TwinReply` dataclass
exposes `model_latency_ms`, `artificial_delay_ms`, and `total_latency_ms`
so callers can verify the actual wall-clock budget.

The callable variant `(text, output_tokens) -> int` is supported so future
experiments can scale delay with reply length to simulate "thinking + typing"
rather than constant wait. This run uses a constant int per bucket.

### Run design

5 delay buckets × 2 personas × 5 turns each = **50 twin calls + 50 proxy
judgments**. Buckets: `[0, 500, 1500, 3000, 6000]` ms.

### Realism judging — text-only proxy with explicit caveat

A text-only Claude-as-judge cannot perceive latency because the judge
never waited. This run uses a proxy judge anyway, with three purposes:

1. **Smoke-test** the judging pipeline end-to-end so it's ready when
   human raters land.
2. **Confirm the null result** — verify that proxy realism is *flat*
   across delay buckets, which would prove the proxy is blind to the
   independent variable and validate the need for exp-5.06 humans.
3. **Generate transcripts** that humans can later re-rate against the
   same items.

## Results

### Mechanism — works perfectly

| Configured delay | Mean injected sleep | Sleep accuracy | Mean total wall-clock |
|---|---|---|---|
| 0 ms | 0 ms | n/a | 2876 ms |
| 500 ms | 500 ms | **1.000** | 2965 ms |
| 1500 ms | 1500 ms | **1.000** | 4175 ms |
| 3000 ms | 3000 ms | **1.000** | 5724 ms |
| 6000 ms | 6000 ms | **1.000** | 8744 ms |

The `asyncio.sleep` injection is exact to the millisecond. The model itself
takes ~2.5–3s on Haiku before any artificial delay — meaning Guide 2's
"500ms baseline" is actually wrong for this stack. Even the *0ms bucket*
in this experiment lands at ~3s wall-clock from the user's perspective
because the model itself is the dominant latency. **Implication for prod:**
the relevant delay range to test is *additional* delay on top of model
latency, not absolute wall-clock.

### Proxy realism scores — flat as predicted (negative control passed)

| Bucket | Proxy realism mean (n=10) |
|---|---|
| 0 ms | 5.00 |
| 500 ms | 4.90 |
| 1500 ms | 4.80 |
| 3000 ms | 4.90 |
| 6000 ms | 4.90 |

Variance: 0.20 across all buckets, with no monotonic trend. This confirms
**the proxy judge is blind to the independent variable** — exactly the
expected null result. A text-only judge cannot rate "feels real because
of timing" because it has no timing context to evaluate.

This is not a failure of the experiment. It's a successful **negative
control**: it proves that the question Guide 2 asks about latency
genuinely cannot be answered without human raters. Any future claim
that "latency improves persona realism" must come with human-rater data,
not LLM-judge data.

## Interpretation

Two findings, one positive and one structural:

**Positive (mechanism):** `artificial_delay_ms` works. Future experiments
that need controllable response timing have a tested API, with both
constant and length-callable variants. Total wall-clock latency is
exposed on `TwinReply` so callers can measure perceived speed.

**Structural (the question this experiment was meant to answer remains
open):** Without humans, we cannot tell whether realism actually peaks at
1500–3000ms total latency, or whether it's monotonically better-with-faster,
or whether it doesn't matter. The proxy judge's flat scores rule out
"text alone reveals timing-based realism" — so any future re-test must
go through exp-5.06's human pipeline.

### Important corollary about model latency

Haiku 4.5 takes **~2876 ms** on average per `messages.create` call for these
short adversarial probes. The Guide 2 framing assumed a ~500ms model
baseline; on this stack, that assumption is wrong by ~6×. If the goal is
to put twin reply timing in the "feels human" range (Guide 2's 1500–3000ms
window), we are *already there* without any artificial delay. Adding 1500ms
on top puts us at ~4400ms which may already be on the slow side.

This reframes the experiment: the relevant question isn't *"how much delay
should we add"* but *"is the model already too slow / about right / too fast
for natural-feeling chat?"* — and that question still requires humans.

## Recommendation

**KEEP** the `artificial_delay_ms` API in `twin/twin/chat.py`. It is a
zero-cost feature that future experiments will need.

**DO NOT** make any production claim about the realism effect of latency
until exp-5.06 humans have rated the transcripts in `conversations.json`
under known wall-clock conditions. The text-only proxy is structurally
incapable of answering the question.

**CHANGE** the wall-clock target range. Guide 2's 1500–3000ms window
implicitly assumed the model itself was negligible. On Haiku 4.5, model
latency alone is already ~2900ms per turn for short replies. The relevant
test buckets going forward should be 0ms (raw model speed) and *small*
adjustments around it: −1000, 0, +500, +1500, +3000 *additional* delay.

**Pair with exp-5.06** when launching the Prolific pilot: include the
exp-4.14 transcripts in Protocol B so raters can pick which feels more
realistic without seeing the timing — the lab equivalent of a blinded
A/B. Because the underlying text is identical and only the wall-clock
differs, this would isolate the timing variable from the content variable.

## Cost

- Synthesis (2 personas): ~$0.071
- Twin (50 calls + sleeps): ~$0.080 LLM, ~275 seconds wall-clock just on injected sleeps
- Proxy judge (50 judgments): ~$0.040
- **Total: ~$0.191**
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `conversations.json` — full 5-bucket × 2-persona × 5-turn transcripts
  with per-turn `model_latency_ms`, `artificial_delay_ms`, `total_latency_ms`
- `proxy_scores.json` — text-only realism scores (negative control)
- `summary.json` — aggregate timing + score metrics
- `scripts/run_exp_4_14.py` (in branch) — reproducible runner
- `twin/twin/chat.py` (in branch) — `artificial_delay_ms` parameter, new
  `TwinReply` timing fields
