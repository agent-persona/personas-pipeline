# exp-5.06 — Human Protocol Design

## Goal

Produce the **canonical human ground truth** pipeline that every downstream
experiment's LLM-judge score can be calibrated against.

Batches 1–4 relied entirely on Claude-as-judge for qualitative metrics. We
don't know whether Claude-as-judge tracks human raters on persona-related
tasks. **Without a calibration number, every WEAK/STRONG verdict in prior
batches has an unknown error bar.** This experiment builds the pipeline to
close that gap.

## Status

**BUILT but NOT LAUNCHED.** The three candidate protocols are specified,
stimuli generation runs offline, Prolific config is drafted, ingest code
computes Krippendorff's α and Cohen's κ — but the actual Prolific study
has not been uploaded. Budget approval and account access required.

Why: this experiment is in the "Execute the plan now" batch-5 run. Everything
that doesn't require an external human-labeler account was built and tested
end-to-end. The remaining work is operational (upload, fund, collect).

## The three candidate protocols

### Protocol A — Blind matching (N=3)

- **Stimulus:** a 6-turn chat transcript where one of the participants is a
  twin driven by *one of three* candidate personas (only the rater doesn't
  know which).
- **Task:** "Which of these three personas produced this transcript?"
- **Response:** forced-choice, `persona_a | persona_b | persona_c`.
- **Agreement metric:** Cohen's κ on the 3-way nominal label across raters.
- **Pros:** direct test of persona-distinctiveness. High external validity.
- **Cons:** needs at least 3 distinct personas per item; sensitive to how
  similar the distractors are.

### Protocol B — Pairwise preference

- **Stimulus:** the same user prompt shown to two different twins (same or
  different personas, same underlying cluster). Rater sees both responses
  side-by-side.
- **Task:** "Which response feels more like a real person from the described
  persona?"
- **Response:** `a_better | b_better | tie` (3-way ordinal with a tie option,
  or forced 2-way if we drop ties).
- **Agreement metric:** Krippendorff's α (ordinal) or Cohen's κ (nominal with
  ties).
- **Pros:** pairwise is usually the easiest-to-rate format and gives the
  cleanest agreement numbers in the RLHF literature.
- **Cons:** preference is relative — absolute quality isn't measured. Useless
  if both replies are bad.

### Protocol C — Forced-choice persona ID

- **Stimulus:** a single twin response to a single prompt, plus the full
  text of N candidate personas (name, summary, vocabulary, sample quotes).
- **Task:** "Which of these N personas wrote this reply?"
- **Response:** `persona_1 | persona_2 | ... | persona_N`.
- **Agreement metric:** Cohen's κ on nominal labels.
- **Pros:** simplest cognitive load per item — rater just pattern-matches
  style.
- **Cons:** relies heavily on the persona descriptions being distinct at the
  surface level. If two personas sound alike in writing, this protocol will
  over-penalize the synthesis.

## Protocol selection criteria

The protocol we canonize is the one with the **highest inter-rater agreement**
(`α` for ordinal, `κ` for nominal) on a shared 60-item pilot. Agreement
thresholds we'll use:

- `α ≥ 0.80` or `κ ≥ 0.70`: adopt as canonical
- `0.60 ≤ α < 0.80`: usable with caveats, run larger pilot to confirm
- `α < 0.60`: protocol is too noisy to ground anything; try the next one

## Files

- `protocols.py` — canonical definitions of the three protocols
- `build_stimuli.py` — CSV generator that turns existing batch-4 transcripts
  into Prolific-ready stimuli files for each protocol
- `prolific_config.yaml` — study-config template with audience/pay/qual
  screening (fill-in-the-blanks for the operator)
- `ingest.py` — reads Prolific label CSVs and computes α, κ, and
  Claude-as-judge-vs-human agreement
- `agreement.py` — from-scratch Krippendorff's α (ordinal) and Cohen's κ
  implementations (no SciPy dependency)
- `test_agreement.py` — unit tests on known inputs where the answers are
  published in the literature

## Manual steps to launch (blocked on operator)

1. Run `python build_stimuli.py` in this worktree to generate
   `output/experiments/exp-5.06/stimuli_protocol_{a,b,c}.csv`.
2. Review each CSV for sensitive content (we're using mock B2B personas so
   should be clean, but always check before uploading a public study).
3. Create Prolific studies using the fields in `prolific_config.yaml`.
4. Upload each CSV as study stimuli. Set rater count to 5 per item.
5. Let the studies complete (estimated 2–4 hours for 60 items × 5 raters
   × 3 protocols at a £9/hr rate, total budget ~£675).
6. Download the raw labels as CSV and place them in
   `output/experiments/exp-5.06/labels_protocol_{a,b,c}.csv`.
7. Run `python ingest.py` → writes
   `output/experiments/exp-5.06/FINDINGS.md` with the α/κ per protocol and
   the Claude↔human agreement number for the winning protocol.

## Downstream callback

When the winning protocol is known, update `evaluation/evaluation/judges.py`
with a `calibrated_judge_confidence(protocol: str) -> float` function returning
the α/κ, so every downstream FINDINGS.md can annotate its judge-based claims
with the calibration number.
