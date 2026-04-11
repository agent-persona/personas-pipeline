# Experiment 4.10: Personality strength dial

## Metadata
- **Branch**: exp-4.10-personality-strength-dial
- **Date**: 2026-04-11
- **Problem Space**: 4 (Twin behavior & expression)
- **Signal**: **DEFER** — core endpoint (human-rated realism) not
  measured; structural manipulation works at the high end, does not
  work at the low end. Recommend deferring ship/kill until the rating
  pass completes and the n=2 sample is expanded.
- **Run artifacts**: `persona_00.json`, `persona_01.json` in this dir
- **Raw comparison**: `structural_comparison.log` in this dir

## Hypothesis
The twin currently builds a static system prompt from a persona dict,
with every trait (goals, pains, vocabulary, quotes) carrying equal
weight. There is no way to modulate *how strongly* the twin expresses
its persona — every reply is "medium" by default.

Adding an `intensity` dial that modulates how *salient* persona traits
are in the output — from subtle (traits only when directly relevant)
to vivid (traits consistently audible, but still plausibly human) —
should produce a curve in human-rated realism that is **non-monotonic**,
with a peak strictly inside (0.0, 1.0). At the low end the twin should
read as generic; at the high end, over-performed; somewhere in the
middle, most real.

The plan was deliberately *not* worded to make this easy to confirm.
Every non-balanced band, including `vivid`, instructs the model to
"stay a coherent, plausible human", so a flat or monotonically
increasing curve is a legitimate possible outcome and not a sign the
experiment failed.

**What the dial is not**: a dial for mood, hostility, combativeness,
frustration, or self-caricature. An earlier draft of the plan
conflated "stronger personality" with "more abrasive", which would
have biased the high end to feel less real for reasons unrelated to
expressive strength. The final wording varies salience and stylistic
consistency, not mood.

## Changes Made
- `twin/twin/chat.py`:
  - Added `INTENSITY_BANDS` (5 named bands at 0.0 / 0.25 / 0.5 / 0.75 / 1.0)
    and `intensity_label(float) -> str` that snaps a float in [0,1] to
    the nearest band. Out-of-range input is clamped.
  - Added `_INTENSITY_FRAGMENTS` and `_INTENSITY_VOCAB_RULES` dicts
    holding the band-specific prompt text.
  - Extended `build_persona_system_prompt(persona, intensity=0.5)`.
    For `label != "balanced"`, inserts a `## Expressive intensity`
    section between "How you talk" and "## Rules", and swaps the
    vocabulary rule line for a band-specific variant. For
    `label == "balanced"`, **early-skips both modifications** and
    returns the byte-identical pre-experiment prompt. This is
    load-bearing: the experimental control must equal current
    production behavior, not a reworded "balanced" prompt.
  - `TwinChat.__init__` gained an `intensity: float = 0.5` parameter
    and a `_prompt_cache: dict[str, str]` keyed on band label.
  - `TwinChat.system_prompt` is now a `@property` backed by the cache,
    preserving the pre-experiment public attribute for any external
    reader.
  - `TwinChat.reply(..., intensity: float | None = None)` gained a
    per-call override so a single `TwinChat` instance can sweep
    across bands without re-instantiation. The cache is keyed on
    label, not float, so 0.55 and 0.50 both hit the `balanced`
    entry (by design — saves rebuilds during fine sweeps).
- `twin/twin/__init__.py`: re-exports `intensity_label`. `INTENSITY_BANDS`
  stays module-private for now.
- `scripts/run_full_pipeline.py`:
  - Added `INTENSITY_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]` and
    `PROBE_QUESTIONS` (5 probes with IDs: frustration, neutral,
    preference, disagreement, reflective).
  - Replaced `stage_twin_chat`'s single-question demo with a
    5 bands × 5 probes = 25 calls per persona sweep. Each result is
    stored as a dict in `entry["twin_intensity_sweep"]` with keys
    `probe_id`, `question`, `intensity`, `label`, `text`, `cost_usd`.
    `twin_demo_cost` now aggregates across the sweep.
  - Generation settings (model, max_tokens, default temperature) are
    held constant across bands. Intensity is the only manipulated
    variable.

Unchanged (by design):
- Persona content is **identical across all 5 bands**. We do not
  prune goals/pains/vocabulary at low intensity. Pruning would
  conflate "less expression" with "less content" and destroy the
  signal.
- `balanced` band is byte-equal to the pre-experiment prompt. This
  is asserted as a verification check.
- `evaluation/evaluation/judges.py` is still a stub. No auto-scoring.
- No changes to synthesis, segmentation, or ingest stages.

## Design decisions worth flagging
- **Float knob, categorical treatment.** The API takes a float, but at
  prompt-build time it snaps to one of 5 fixed text fragments. The
  manipulated variable is effectively categorical (5 prompt conditions).
  Writeups should describe this as "5 prompt conditions", not "a
  continuous intensity axis". The float is a convenience for the sweep
  loop, not a true continuous treatment.
- **Top band = "vivid", not "exaggerated".** "Exaggerated" is an
  instruction to sound fake; "vivid" is an instruction to be
  consistently distinctive. Every non-balanced band's fragment
  explicitly instructs the model to stay a "coherent, plausible
  human".
- **"Without repetition" / "without catchphrase stuffing" guardrails.**
  The `strong` and `vivid` rule overrides contain explicit
  anti-repetition clauses. Without these, the high bands would
  trivially collapse into "which band stuffs keywords hardest",
  producing a realism drop at high intensity that is an artifact
  of repetition rather than evidence about expressive strength.
- **Probe set, not a single question.** A single frustration-shaped
  probe would bias the whole curve toward whichever band expresses
  pain most vividly. Using 5 probes across different conversational
  shapes (frustration, neutral factual, preference, mild
  disagreement, reflective) forces the realism signal to generalize
  across shapes.

## Setup
- **Tenant**: `tenant_acme_corp` (B2B SaaS, project management tool)
- **Clusters**: `clust_4c973b9e03fa`, `clust_b14cacd128db`
- **Synthesis model**: `claude-haiku-4-5-20251001`
- **Twin model**: `claude-haiku-4-5-20251001`
- **Judge**: none (`LLMJudge.score_transcript` still a stub)
- **Personas**:
  - `Aiden Chen, DevOps Architect` — synthesis 2 attempts, groundedness 1.00, $0.0341
  - `Maya – The Time-Conscious Freelance Designer` — 2 attempts, 1.00, $0.0334
- **Sweep**: 2 personas × 5 probes × 5 bands = 50 twin calls
- **Twin sweep cost**: $0.0808 total ($0.0423 + $0.0385)
- A separate default-config baseline run (per the README "Shared
  harness" control requirement) was produced from pre-experiment
  commit `12dfba3` in a git worktree and lives at
  `baseline/persona_*.json`. See `### Baseline comparison` below.
- **Note on flakiness**: both the experimental run and the separate
  baseline run flaked on first-attempt synthesis. Experimental run:
  3/3 attempts failed on the first cluster (scores 0.70 / validation /
  0.88), retry cleared it. Baseline run: first attempts failed on
  both clusters (scores 0.89, 0.55, 0.63), succeeded on attempt 2
  for cluster 0 and attempt 3 for cluster 1. **The flakiness
  reproduces in the pre-experiment code** — it is pre-existing and
  unrelated to the intensity dial. Flagged for anyone chasing
  pipeline stability.

## Shared metrics (per README "Shared harness")

The top-level `README.md` specifies the convention every experiment
must satisfy:

> Per `PRD_LAB_RESEARCH.md`, no experiment lands without:
> 1. **A hypothesis** — written before the run.
> 2. **A control** — a run on the same golden input with the default config.
> 3. **A metric** — one of the shared metrics in `evaluation/metrics.py`
>    (schema validity, groundedness, distinctiveness, judge score, drift, cost).
> 4. **A result + decision** — adopt / reject / defer, written up in your space's
>    results file.

The shared metrics module at `evaluation/evaluation/metrics.py`
currently implements three functions; the rest are stable signatures
with TODO bodies owned by problem space 5. Computed on both the
experimental run (`persona_00.json`, `persona_01.json`) and the
separate default-config control run (`baseline/persona_00.json`,
`baseline/persona_01.json`):

| Shared metric | Experimental | Baseline (control) | Source |
|---|---|---|---|
| `schema_validity(persona_dicts, PersonaV1)` | **1.00** (2/2) | **1.00** (2/2) | `evaluation/metrics.py` |
| `groundedness_rate(reports)` | **1.00** | **1.00** | `evaluation/metrics.py` |
| `cost_per_persona(total_cost, n=2)` | **$0.0742** | **$0.0442** | `evaluation/metrics.py` |

Metrics values verified by running
`.venv/bin/python -m evaluation.metrics` against both artifact sets.
The experimental run's higher per-persona cost is the twin-stage
sweep (25 calls/persona vs. 1 call/persona in baseline) — expected
and recoverable to baseline any time the dial is set to `balanced`
and called once per persona.

Notes on interpretation:

- `cost_per_persona` at ~$0.074 is well under the experimental cap;
  the recommended n≈10 followup would still leave plenty of headroom.
- `schema_validity` and `groundedness_rate` are **proxy-only signals
  for this experiment**. They tell us the synthesis stage is healthy
  (both personas validated, both scored 1.00 on groundedness), not
  whether the intensity dial affects twin realism. This experiment
  is a twin-stage (space 4) manipulation; the synthesis-stage metrics
  are reporting on input quality, not on the treatment itself.
- The shared metrics that would actually close the primary endpoint
  (human-rated realism and its automated analogues) are all TODO in
  `metrics.py` and blocked on problem space 5:
  - `turing_pass_rate` — the direct human-rated realism endpoint.
  - `judge_rubric_score` — Opus-as-judge per-dimension realism rubric.
  - `human_correlation` — Spearman between judge and human labels,
    needed before trusting any automated realism proxy.
  - `distinctiveness` — needed if we want to show the bands are
    actually producing different outputs in embedding space rather
    than only in surface vocabulary.
  - `drift` — turn-N-vs-turn-1 stylometric drift, which would be the
    natural multi-turn generalization of this experiment.
- The ad-hoc structural metrics reported in `## Results` below
  (word count, vocabulary count, vocab density per 100 words) are
  **supplemental diagnostics, not substitutes for a shared metric**.
  They are only informative in this writeup because the
  realism-shaped shared metrics above are TODO; once those land,
  the structural numbers should be demoted to a smoke-test appendix.

## Results

### Baseline comparison (separate default-config run)

Per the README "Shared harness" control requirement, a second
pipeline run was produced from commit `12dfba3` (the parent of
this branch, pre-experiment) in a git worktree with all other
config held constant. Artifacts live at `baseline/persona_*.json`.
The baseline run uses the old `stage_twin_chat` — one frustration
question per persona, shape `{twin_demo_reply, twin_demo_cost}`
with no `twin_intensity_sweep` key.

**Matched slots, unmatched personas.** Both runs processed the
same mock tenant (`tenant_acme_corp`), and both produced 2
clusters that clearly correspond to the same behavioral
archetypes (infra/devops + time-conscious designer). But the
cluster IDs differ (`clust_c952277bda07` vs `clust_4c973b9e03fa`,
`clust_5b86aa018d69` vs `clust_b14cacd128db`), and the synthesized
personas are not the same:

| slot | baseline persona | experimental persona |
|---|---|---|
| 0 | `Alex, the Infrastructure Automation Lead` | `Aiden Chen, DevOps Architect` |
| 1 | `Maya the Time-Conscious Freelance Designer` | `Maya – The Time-Conscious Freelance Designer` |

Slot 1 is essentially the same character in both runs (same
tagline, same focus on billable-hour friction). Slot 0 is the
same archetype with a different name and a different top
frustration — baseline Alex leads with "manual team provisioning",
experimental Aiden leads with "context switching". This is the
same persona-drift confound exp-1.11 flagged: *"Baseline and
experimental personas are two distinct Haiku runs on the same
clusters. Names and content differ."*

**Segmentation is not deterministic on cluster IDs.** An earlier
version of the plan asserted segmentation was deterministic. The
baseline run falsifies that: same input records, same parameters,
different cluster IDs on both slots. The *content* of the clusters
appears stable (same archetype in each slot), but the ID hashes
differ. Either the clusterer has a stochastic component or the ID
hash includes a seed/timestamp. This is a minor correction, not
an experiment blocker — but worth knowing for anyone doing cluster
ID-keyed comparisons across runs.

**Frustration-probe content, baseline vs experimental-balanced:**

- Baseline Alex (slot 0, pre-experiment code):
  *"Manual team provisioning. Every time we hire, I'm still
  clicking through invitation UIs instead of scripting it..."*
- Experimental Aiden (slot 0, `balanced` band of the sweep):
  *"Context switching. I'm constantly bouncing between my project
  management UI, GitHub, Slack, and my terminal..."*

These are different top frustrations. But because Alex and Aiden
**have different persona dicts** (different `pains` lists from
two distinct synthesis runs), this cannot be attributed to the
twin code. The twin faithfully renders each persona's top
frustration; the personas themselves just have different ones.
Slot 1 (Maya) produces closer content in both runs, matching the
near-identical persona dicts.

**What the baseline run *does* demonstrate:**

1. **Synthesis-stage flakiness is pre-existing**, not introduced
   by the intensity changes. Both runs required retries, and the
   baseline run (on `12dfba3` without any of our changes) needed
   *more* retries (3 total vs. 2 in the experimental run). This
   fully clears the intensity dial of any responsibility for the
   flake.
2. **Shared-metric parity holds**: `schema_validity` and
   `groundedness_rate` are both 1.00 in both runs. The experiment
   does not degrade synthesis quality.
3. **Convention compliance**: the branch now has the `baseline/`
   + main artifact layout prior experiments (exp-2.16, exp-2.17,
   exp-5.08) established.

**What the baseline run does NOT resolve:**

- **Finding 5 (content drift on preference probe) is still open.**
  A clean A/B on Finding 5 would require calling the pre-experiment
  twin code against the *experimental persona dict* (Aiden) with
  the preference question, to check whether pre-experiment code
  on a fixed persona picks the same top tool as the `balanced`
  slice of the sweep on the same persona. Because the baseline
  run produced a different persona entirely (Alex), its
  preference-probe answer cannot be compared to Aiden's. See the
  "Recommendations" section for the targeted follow-up.
- **Sampling-noise triangulation on the `balanced` slice.** Same
  reason: the baseline run's default-config twin call was made
  against a different persona, so it is not a second sample of
  the same underlying distribution. Disentangling "dial effect"
  from "temperature sampling noise" still requires a fixed-persona
  repeat-sampling experiment.

In short: the baseline run clears the intensity dial of
synthesis-quality and flakiness concerns, satisfies the README
control requirement, documents the persona-drift confound, and
sharpens what the follow-up needs to look like — but it does not
close the content-drift concern on its own.

### Verification (unit-level)
- `intensity_label(0.0) == 'subtle'`, `0.25 == 'understated'`,
  `0.5 == 'balanced'`, `0.75 == 'strong'`, `1.0 == 'vivid'`.
  Out-of-range input (−5, 2) clamps correctly. ✓
- **Baseline parity**:
  `build_persona_system_prompt(persona, 0.5) == build_persona_system_prompt(persona)`
  (byte-equal). This is the experimental-control guarantee. ✓
- All 4 non-balanced bands produce distinct prompts, each containing
  the correct `## Expressive intensity` fragment and the correct
  vocab-rule override. `balanced` contains neither section. ✓
- `TwinChat(persona, client)` with no intensity arg still works;
  `twin.system_prompt` equals the old prompt. ✓
- Prompt cache is label-keyed: `_system_for(0.55)` and `_system_for(0.5)`
  share one cache entry. ✓

### End-to-end run
Both personas produced a `twin_intensity_sweep` of length 25 with all
5 labels × 5 counts and all 5 probes × 5 counts. All 5 frustration-probe
replies are textually distinct within each persona. Cost was ~$0.04
per persona (50 Haiku calls total).

### Structural comparison: experimental bands vs balanced baseline

Aggregates across 2 personas × 5 probes = 10 replies per band:

```
        band    n   words  sents  vocab  v/100w
      subtle   10   102.0    5.4   1.90    2.18
 understated   10   106.9    5.8   1.80    1.76
    balanced   10   120.1    6.7   1.60    1.51   <- baseline
      strong   10   108.0    4.8   2.50    2.56
       vivid   10   115.4    5.3   2.90    2.46
```

Deltas vs `balanced`:

```
        band    d_words   d_vocab   d_v/100w
      subtle     -18.1     +0.30     +0.67
 understated     -13.2     +0.20     +0.25
      strong     -12.1     +0.90     +1.06
       vivid      -4.7     +1.30     +0.95
```

**Finding 1 — High-end vocabulary manipulation works.**
`strong` and `vivid` use more vocabulary than `balanced` on both
absolute count (+0.9 / +1.3) and density (+1.06 / +0.95 per 100
words). The `balanced → strong → vivid` raw-count gradient is
monotone (1.60 → 2.50 → 2.90). The high-end bands are doing
measurably different things in the direction the prompt asks for.

**Finding 2 — Low-end vocabulary manipulation does not work.**
`subtle` uses *more* vocabulary than `balanced` in both absolute
count (1.90 vs 1.60) and density (2.18 vs 1.51 per 100 words).
The "Use your characteristic vocabulary sparingly" rule override
is not landing on these probes. Possible causes:
  1. **Length confound**: `subtle` replies are 18 words shorter on
     average, which inflates density even at constant absolute usage.
     But absolute usage is also higher (1.90 vs 1.60), so length
     alone does not explain it.
  2. **Probe/fragment mismatch**: the `subtle` fragment says
     persona traits surface "when the conversation directly touches
     them". These probes are *already* directly tool/work-themed,
     so "when relevant" collapses to "always" for this probe set.
  3. **Model compliance asymmetry**: Haiku may be more responsive
     to "use more" than to "use less". Cannot distinguish this
     from (2) without a larger probe set that includes genuinely
     off-topic questions.

**Finding 3 — Catchphrase-stuffing guardrail holds.**
Across all 50 replies, exactly **1** (in `strong`) contained any
single vocabulary word repeated ≥3 times in one reply. `vivid`:
**0/10**. The "without catchphrase stuffing" clause in the rule
override is landing. A null realism result at high intensity, if
it shows up in rating, would not be attributable to trivial
keyword repetition.

**Finding 4 — `balanced` is the longest band on average.**
Word counts: `subtle` 102 → `understated` 107 → **`balanced` 120**
→ `strong` 108 → `vivid` 115. The baseline is longest. This is a
**potential confound for the rating pass**: if a human rater's
unconscious heuristic is "longer = more considered = more real",
the baseline gets an unearned advantage. Rater instructions should
explicitly say "do not weight length". Breaking down by probe
shows the effect is not uniform — `balanced` is longest on
frustration, neutral, and disagreement, but not preference or
reflective.

**Finding 5 — Substantive content drift on one probe.**
The `preference` probe asks "if you could only keep one tool,
which?". For Aiden Chen (persona 0):
  - `subtle`: "GitHub, no question."
  - `understated`: "GitHub, no question."
  - `balanced`: "Terraform, no question."
  - `strong`: (phrased as "a single API-driven source of truth")
  - `vivid`: "GitHub, without hesitation."

The persona's stated top tool **flips between `balanced` and the
other bands**. This is exactly the failure mode verification step
6 warns about: the dial is supposed to modulate *expression*, not
*substantive claims*. Two caveats:
  1. It is one probe on one persona out of 10 (persona, probe) pairs.
     Not enough data to conclude the dial is systematically
     corrupting content.
  2. The persona plausibly has multiple near-tie top tools
     (Terraform, GitHub, Kubernetes). Haiku may be picking
     differently between them for reasons unrelated to intensity,
     with the framing acting as a red herring.

On the `disagreement` probe, all 5 bands take the same stance
("push back on automation-away claim") and vary only in texture —
that is the intended behavior.

To disentangle "dial corrupts content" from "model nondeterminism"
we would need multiple samples per band (same persona, same probe,
different seeds / non-zero temperature) or a lower temperature on
the current runs. The separate baseline run (see
`### Baseline comparison` above) **cannot resolve this** because
it produced a different persona entirely (Alex, not Aiden) — see
the persona-drift confound there. A targeted follow-up that
calls the pre-experiment twin code against the **experimental
persona dict** is described in Recommendations.

### Semantic-drift audit (eyeball check, persona 0, preference + disagreement probes)
Cross-band substantive claims are **consistent on disagreement** (all
bands push back and cite architecture/constraint-modeling as the
non-automatable work) and **inconsistent on preference** (GitHub vs
Terraform as per Finding 5). Cannot make a general statement from
N=2 probes.

### Mood / caricature audit (eyeball check)
The guardrail against mood drift is holding in the observable data.
The `vivid` replies in the frustration probe read as more elaborated
and more tool-name-dense than `balanced`, but not more hostile,
more frustrated, or more caricatured. None of the `vivid` replies
introduce catchphrase repetition, all-caps, or the kind of
performative framing the earlier "exaggerated" wording would have
encouraged. This is weak evidence (eyeball only) that the rename
from `exaggerated` → `vivid` and the "stay a coherent, plausible
human" clause are doing their job.

## Signal summary

| Dimension | Signal |
|---|---|
| Unit/integration tests | PASS |
| Baseline parity (byte-equal) | PASS |
| Shared metrics (`schema_validity`, `groundedness_rate`) | PASS — 1.00 in both experimental and baseline runs |
| Separate default-config control run | PRESENT — clears dial of synthesis-quality and flakiness concerns |
| High-end salience (vocab usage) | POSITIVE — monotone `balanced → strong → vivid` |
| Low-end salience (vocab usage) | **NEGATIVE** — `subtle` ≥ `balanced` |
| Catchphrase stuffing | NONE — guardrail holds |
| Mood / caricature drift | NONE observed (eyeball, weak evidence) |
| Length confound | PRESENT — `balanced` longest band |
| Content drift | ONE observed case (preference probe, persona 0), cannot generalize from N=2; baseline run cannot resolve (persona-drift confound) |
| Human realism curve (core endpoint) | **NOT MEASURED** |

## Recommendations

1. **Do not ship yet.** The core endpoint is unmeasured, and the
   structural evidence is mixed. Treat this branch as
   "implementation and partial run complete; findings deferred
   pending rating".

2. **Do the rating pass before iterating on the prompt.** The
   temptation from Finding 2 is to reword the `subtle` fragment
   to force lower vocabulary usage. Resist until rating is done.
   If `subtle` and `balanced` score similarly on realism, that is
   *the result* — evidence that the low-end manipulation doesn't
   produce a realism difference on these probes, not a bug to fix.

3. **Instrument rating protocol against the length confound**
   (Finding 4). Tell the rater explicitly: "longer replies are
   not automatically more real; do not use length as a tiebreaker."

4. **Sample size is the main followup, not prompt wording.** N=2
   personas × 5 probes is enough for structural smoke but not
   enough for a realism curve. Rerunning the pipeline 3–5 more
   times (different synthesis seeds) to reach n≈10 personas
   would let us:
     - Check whether the `strong`/`vivid` vocab gradient holds up.
     - Check whether the preference-probe content flip is systematic
       or persona-0-specific noise.
     - Get enough rating samples for any realism curve to be
       visible above rater noise.
   Cost is ~$0.15 per run, so 5 more runs is ~$0.75.

5. **Targeted content-drift test on a fixed persona.** The
   separate baseline run cannot resolve Finding 5 because it
   produced a different persona (Alex, not Aiden). The clean
   follow-up is to load the experimental Aiden persona dict,
   call the **pre-experiment** `TwinChat.reply()` (from `12dfba3`)
   with the preference probe directly, and compare its output to
   the `balanced` slice of the existing sweep. Same persona, same
   prompt, same config — the only difference is that one was
   generated in isolation and the other inside a 25-call sweep
   session. If the answers match, the embedded-control argument
   holds. If they diverge, something about batching is affecting
   the output and the `balanced` slice cannot stand in for a true
   control. A temperature-0 variant of this same test would also
   distinguish "dial effect" from "sampling noise" on the
   GitHub-vs-Terraform flip.

6. **Defer the `subtle` rewording to a followup experiment.**
   Specifically: test a probe set that includes genuinely
   off-topic questions (e.g. "what's a book you've read recently
   that isn't about work?"), where `subtle` *should* clearly
   suppress work-vocabulary and `balanced`/`vivid` should not.
   If `subtle` still overuses vocabulary on off-topic probes,
   the prompt wording is the problem. If it suppresses correctly,
   the current probe set is the problem.

## Caveats
- **N=2 personas, N=5 probes.** Too small to reject or confirm the
  non-monotonic hypothesis even with a clean rating pass.
- **Single-turn only.** Realism often depends on multi-turn
  consistency; this experiment cannot measure it.
- **No human rating yet.** The structural findings above say
  "the manipulation is measurable" for the high end, not "it
  produces a realism difference". Those are very different claims.
- **No inter-rater reliability.** When rating happens, the
  protocol assumes a single rater unless explicitly arranged
  otherwise.
- **Rater gradient inference.** Raters will likely start to
  recognize the subtle → vivid gradient after a few personas,
  even with shuffled presentation. Counter-mitigate by having
  the rater commit scores per-probe before moving to the next.
- **Default temperature.** Bands were compared at default
  temperature, which means some inter-band differences may be
  sampling noise rather than prompt effects. Finding 5 is the
  clearest case.
- **Eyeball-grade mood audit.** The "no mood drift" claim is
  from reading replies, not from any rubric. A judge might find
  consistent small differences I missed. Weak evidence only
  usable to say "no obvious mood drift", not "no mood drift".
- **One failed synthesis run** (retry succeeded) is a pre-existing
  flakiness not introduced by this experiment. Flagged for
  separate investigation.

## Cost

### Experimental run
- Synthesis (2 personas, 2 successful attempts each): $0.0675
  - `Aiden Chen, DevOps Architect`: $0.0341
  - `Maya – The Time-Conscious Freelance Designer`: $0.0334
- Twin intensity sweep (50 calls, 2 personas × 25): $0.0808
  - Aiden sweep: $0.0423
  - Maya sweep: $0.0385
- **Experimental run total**: $0.1483
- `cost_per_persona` (shared metric): **$0.0742**

### Baseline run (separate default-config control, from `12dfba3`)
- Synthesis (2 personas, 2/3 attempts): $0.0856
  - `Alex, the Infrastructure Automation Lead`: $0.0336 (2 attempts)
  - `Maya the Time-Conscious Freelance Designer`: $0.0520 (3 attempts)
- Twin demo replies (2 calls, 1 per persona): $0.0027
  - Alex: $0.0012
  - Maya: $0.0015
- **Baseline run total**: $0.0883
- `cost_per_persona` (shared metric): **$0.0442**

### Branch totals
- Prior failed synthesis run (3/3 attempts, first cluster, retry
  resolved): not itemized, estimated ~$0.03.
- **Total spent on branch**: ~$0.27

Well under the experimental cap, but the recommended followup
(5 more pipeline runs to reach n≈10 personas) would bring total
cost to ~$0.9.

## Open questions (deferred)
- Is the human-rated realism curve non-monotonic, monotone, or flat?
  **This is the experiment's core question and is unanswered.**
- Is the preference-probe content flip (Finding 5) a dial effect
  or sampling noise? Needs temperature-0 rerun or multiple samples
  per band.
- Does the `subtle` band ever suppress vocabulary, or is the
  wording fundamentally asymmetric to the `vivid` wording in a way
  Haiku can't follow? Needs an off-topic probe set.
- Does the length confound (Finding 4) reproduce across more
  personas, and is it the `balanced` prompt's fault or a Haiku
  idiosyncrasy? If it reproduces, rater instructions need a
  stronger anti-length directive, or the rating UI should hide
  length (e.g. normalize to short excerpts).
- If the realism curve turns out flat (no effect in either
  direction), is the right next move to kill the dial, or to test
  a more aggressive prompt variant that *does* leak mood
  (knowing it's confounded) to establish whether any prompt-level
  intervention can move the realism needle at all?
- Should the dial eventually become a per-turn override driven
  by the conversation (e.g. "raise intensity when the user
  sounds hostile")? Out of scope here; depends on rating results.
