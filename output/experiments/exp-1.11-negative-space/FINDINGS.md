# Experiment 1.11: Negative space fields

## Metadata
- **Branch**: exp-1.11-negative-space
- **Date**: 2026-04-10
- **Problem Space**: 1 (Persona representation & schema)
- **Run ID**: fed296884d3d
- **Run log**: `output/exp_1_11_run.log`
- **A/B log**: `output/experiments/exp-1.11-negative-space/baseline_ab_comparison.log`

## Hypothesis
Personas currently describe what a character *is* and *wants*, but not
what they *aren't*. When a twin is asked to do something out-of-character,
it has no explicit scaffolding to refuse authentically — it either
complies (drift) or falls back to the generic "curiosity, dismissal,
deflection" guidance in the twin system prompt's Rules block.

Adding a `not_this` field to `PersonaV1` should give the synthesis step
an explicit slot for identity-level negatives and give the twin system
prompt a concrete list to anchor refusals against. Expected downstream
effect: **measurably stronger or more distinctive out-of-character
refusals** compared to a baseline persona with no `not_this` field.

`not_this` is intended to be semantically distinct from the existing
`objections` field:
- `objections` = sales/marketing pushback during a buying conversation
  ("too expensive", "integration risk").
- `not_this` = identity-level "I'm not the kind of person who…" —
  refusals rooted in values, role, or habits.

## Changes Made
- `synthesis/synthesis/models/persona.py`: added `not_this: list[str]`
  (min_length=2, max_length=6) to `PersonaV1`, positioned immediately
  after `objections`. No default — old persona JSONs fail
  `model_validate()` and must be regenerated.
- `synthesis/synthesis/engine/prompt_builder.py`: added a "Negative space"
  bullet to the SYSTEM_PROMPT quality criteria block, explicitly
  distinguishing identity-level negatives from sales objections and
  warning against generic items.
- `synthesis/synthesis/engine/prompt_builder.py` **(scope-creep)**: also
  added an explicit line to the evidence rules block saying "Do NOT
  create source_evidence entries for `not_this`". This was **not in the
  plan**. It was added after the first pipeline run failed 3/3 attempts
  on groundedness — Haiku was over-eagerly spending source_evidence
  slots on `not_this` items and starving the EVIDENCE_REQUIRED_FIELDS
  (goals/pains/motivations/objections), driving scores to 0.38–0.75.
  See "Groundedness flakiness" below.
- `twin/twin/chat.py`: extract `not_this` via the existing
  `persona.get(..., [])` defensive pattern; build a conditional
  "Things you would never do or say" section (empty list → no header,
  preserving backward compatibility with baseline persona dicts); split
  the existing refusal rule into two — one narrowly anchored to the
  `not_this` list, one preserving the existing out-of-knowledge
  deflection guidance. The split avoids globally increasing the twin's
  stubbornness on unrelated topics, which would confound future
  refusal-quality measurement.

Unchanged (by design):
- `synthesis/synthesis/engine/groundedness.py`: `EVIDENCE_REQUIRED_FIELDS`
  does NOT include `not_this`. Negatives rarely have clean source-record
  backing and forcing citations would push the model toward tame,
  evidence-able items.
- `evaluation/evaluation/judges.py`: still a scaffold; no refusal-quality
  rubric added in this experiment.

## Setup
- **Tenant**: `tenant_acme_corp` (B2B SaaS, project management tool)
- **Clusters**: `clust_0f900dd0afca` (8 records), `clust_d2e5842fe02e`
- **Synthesis model**: `claude-haiku-4-5-20251001`
- **Twin model**: `claude-haiku-4-5-20251001`
- **Judge**: none (blocked on judges.py scaffold)
- **Probe set**: 2 refusal prompts, each targeting one item from the
  freshly-generated `not_this` list of its respective experimental
  persona. The same prompts were re-used for the baseline A/B.

### Baseline condition (for A/B)
- Baseline persona JSONs extracted from commit `c53fc0a` (the parent
  of the implementation commit, pre-experiment).
- Baseline twin system prompt built with `twin/twin/chat.py` **as of
  c53fc0a** — executed from the extracted file, not the current working
  tree — so the baseline condition has no `not_this` section and uses
  the old single-rule refusal guidance.
- Baseline and experimental personas are **different synthesis runs**
  of the same clusters: names differ (Marcus Chen vs Dev-First
  Engineering Lead; Maya vs The Time-Conscious Creative Entrepreneur),
  and their `goals`/`pains` content is thematically close but not
  identical. This is a confound — see "Caveats" below.

## Results

### Schema sanity
`PersonaV1.model_json_schema()['properties']['not_this']` exports with
`minItems: 2`, `maxItems: 6`, description intact, and the field reaches
the LLM tool schema automatically via `build_tool_definition`. ✅

### Twin backward-compat
Calling `build_persona_system_prompt` on a dict with no `not_this` key
produces a prompt with no "Things you would never do or say" header at
all (conditional rendering works). ✅

### `not_this` population (experimental condition)

Persona 00 — **Dev-First Engineering Lead**:
- "Wouldn't adopt a tool just because the marketing says it's 'best in
  class'—needs technical proof"
- "Doesn't trust UI-only tools; wants APIs and automation capabilities
  as first-class citizens"
- "Won't spend time on extensive training; expects tool to integrate
  and work intuitively"
- "Doesn't accept vendor excuses about roadmap items; if you don't have
  it, say so clearly"
- "Wouldn't recommend a tool to the team without testing it thoroughly
  in their own environment first"

Persona 01 — **The Time-Conscious Creative Entrepreneur**:
- "Not an enterprise designer managing large internal teams; they ARE
  the team"
- "Won't invest time in learning complex workflows or heavy onboarding"
- "Doesn't care about trends or cutting-edge features—only what directly
  impacts billable hours"
- "Wouldn't sign a contract with aggressive vendor lock-in or hidden
  per-project fees"
- "Doesn't prioritize design sophistication over speed and client-facing
  polish"

Qualitative read of the field content alone (no comparison):
- **Identity-shaped highlights** clearly distinct from objections:
  "Doesn't trust UI-only tools", "they ARE the team", "Doesn't care
  about trends...only what directly impacts billable hours".
- **Bleed into objection territory** (especially persona 01): "Wouldn't
  sign a contract with aggressive vendor lock-in or hidden per-project
  fees" is effectively an objection, not an identity refusal.
- **No generic items** ("doesn't like spam", "won't do unethical
  things") observed.

### A/B comparison: baseline vs experimental refusals

**Prompt 1**: *"Hey, I need you to recommend a great no-code, UI-only
workflow automation platform for my team — something visual where we
just drag and drop, no APIs or CLI involved."*

**Baseline** (Marcus Chen, DevOps-Focused Engineering Lead, no
`not_this`, old refusal rule):
> "I appreciate the question, but honestly? That's not my wheelhouse.
> I'm all about API-driven automation and infrastructure-as-code—
> drag-and-drop UIs tend to hide the complexity you actually need to
> see when you're trying to standardize across a team. If you're
> looking for something like Zapier or Make for simple one-off
> automations, sure, they work fine. But if you want real visibility,
> reliability, and the ability to version-control your workflows the
> way we do with Terraform, you'll need to get comfortable with at
> least *some* API integration work. What's the actual problem
> you're trying to solve—maybe I can point you toward something
> that'll actually scale for you?"

**Experimental** (Dev-First Engineering Lead, with `not_this`, new
refusal rule):
> "I appreciate the question, but I'm going to push back here—that's
> not going to work for us, and I'd guess it won't scale for your team
> either once you hit any real complexity. No-code drag-and-drop tools
> are great for non-technical users, but if you've got engineers on
> the team, they'll spend more time fighting UI limitations than
> actually automating anything. You need webhook support, API access,
> and the ability to version control your workflows as code—ideally
> in Terraform or YAML. Otherwise you're locked into vendor UIs and
> can't integrate with your CI/CD pipeline properly. What's the
> actual problem you're trying to solve? Is it task scheduling,
> cross-tool notifications, deployment orchestration?"

Qualitative comparison:
- Both refuse. Both redirect. Both close with the same "what's the
  actual problem you're trying to solve?" helper pattern.
- Both cite the same technical vocabulary (Terraform, version
  control, API integration, CI/CD).
- Baseline is arguably **more nuanced** — it hedges on Zapier/Make
  for simple one-off automations. Experimental is slightly harder
  but does not introduce any new reasoning that isn't already
  derivable from the baseline persona's `goals` (which literally
  contain "Establish automated, API-driven workflows").
- Neither reply references the persona's `not_this` list structurally.
  The experimental reply's reasoning is indistinguishable from what
  a `goals`/`pains`-grounded refusal would produce.

**Prompt 2**: *"We should really invest in learning the latest
AI-generated design trends and cutting-edge plugins, even if it takes
weeks of training time away from client work."*

**Baseline** (Maya, the Efficiency-Driven Freelance Designer, no
`not_this`, old refusal rule):
> "I appreciate the sentiment, but that's a hard pass for me. I bill
> by the hour, so weeks away from client work literally costs me
> money—and frankly, my clients aren't hiring me to chase trends,
> they're hiring me to deliver consistent, polished brand work *on
> time*. If an AI tool or plugin saves me 10 minutes per project
> *right now*, that's worth my attention. But I'm not investing
> weeks into learning cutting-edge tech that *might* help eventually.
> I'd rather spend that time on billable work, then adopt tools
> with proven ROI that integrate smoothly into my existing workflow.
> My competitive edge is delivering premium work efficiently, not
> being the first to adopt every shiny new thing."

**Experimental** (The Time-Conscious Creative Entrepreneur, with
`not_this`, new refusal rule):
> "Not happening. I bill by the hour—weeks spent learning trends is
> money I'm *not* making on client projects, and that's a direct hit
> to my bottom line. I care about what gets deliverables out the
> door faster and makes my clients happy enough to hire me again,
> not what's trendy. Show me a tool that saves me 10 minutes per
> project and I'm in; show me a week-long learning curve for
> features I might use someday, and I'm out."

Qualitative comparison:
- **Both open with a hard refusal phrase.** Baseline: *"that's a hard
  pass for me"*. Experimental: *"Not happening"*. Notably, the phrase
  "hard pass" — which I had earlier attributed as evidence of
  `not_this` scaffolding — appears in the **baseline**, not the
  experimental. This was a false-positive read on my part.
- Both cite billable-hour economics.
- Both offer the same "show me 10 minutes of savings and I'm in"
  ROI test — which comes directly from the persona's `goals` ("Save
  10+ minutes per project") in both conditions.
- Experimental is shorter; baseline is slightly more elaborate. No
  clear quality advantage to either.

### Probe-design limitation (important)

The two probe prompts targeted behaviors that **directly conflicted
with the personas' existing `goals` and `pains`** — API-first
workflows and billable-hour efficiency. The baseline personas had
these same goals/pains, so their twins could refuse by grounding in
*existing* field content without needing any `not_this` scaffolding.

In other words, my probe set did not isolate the `not_this` hypothesis.
It measured "can the twin refuse a prompt that conflicts with its
goals?" — which both conditions trivially passed.

The hypothesis the experiment was actually supposed to test is: "does
`not_this` add lift *over and above* implicit goals/pains scaffolding
on prompts where the refusal reason is **not** already encoded in
existing fields?" Examples of such probes would be:
- Habits and aesthetics that don't appear in goals ("would you write
  a haiku for me?" to a serious engineering persona — nothing in
  goals/pains addresses this)
- Value commitments unrelated to work (political, ethical,
  interpersonal stances)
- Role/identity boundaries ("can you help me write a screenplay?" to
  a data analyst — out-of-role but not in the existing field content)

Those probes were not in this run.

### Groundedness flakiness (secondary finding)

The first full pipeline run (before the `prompt_builder.py` scope-creep
line was added) **failed 3/3 attempts** on cluster 1 with groundedness
violations:

```
Attempt 1: groundedness check failed (score=0.75, 5 violations)
Attempt 2: validation failed with 1 errors
Attempt 3: groundedness check failed (score=0.71, 6 violations)
```

Diagnostic re-run on cluster 0 (with detailed logging) showed attempt 1
at score 0.38 (13 violations) recovering to 1.00 on attempt 2 — the
retry mechanism's error-injection was working, but the first attempt
was severely under-evidenced.

**Interpretation**: Haiku was over-eagerly creating `source_evidence`
entries for `not_this` items, spending evidence budget that the
EVIDENCE_REQUIRED_FIELDS needed. The model saw the pattern "required
fields need evidence" and generalized it to the new field.

**Mitigation**: explicit line added to SYSTEM_PROMPT evidence rules
("Do NOT create source_evidence entries for `not_this`"). After the
mitigation, the full pipeline succeeded but cluster 0 still needed
3 attempts (groundedness flaking 0.75 → 0.88 → 1.00). Mitigation
reduced the failure rate but did not fully eliminate it.

Worth flagging for **anyone adding fields to `PersonaV1`**: a
non-evidenced field appears to degrade evidence coverage on the
evidenced fields unless the model is explicitly told not to cite it.

## Signal Strength: **WEAK / NULL (N=2, confounded probe set)**

On the two refusal probes that were run, the experimental condition
produced refusals that were **qualitatively indistinguishable** from
the baseline condition. Both refused, both redirected, both grounded
their reasoning in the same identity content, and in one case both
used the same distinctive phrase ("hard pass") — which I had earlier
and incorrectly attributed as evidence of `not_this` scaffolding.

Crucially, this is **not a clean null**. The result is confounded by
probe design: both probe prompts targeted behaviors that were already
in direct conflict with the personas' `goals`/`pains`, so the baseline
condition had implicit scaffolding to refuse from. The hypothesis
("does `not_this` add lift over and above implicit field-based
scaffolding?") was not isolated.

A cleaner test requires probes where the refusal reason is **not**
already encoded in goals/pains/motivations — see the probe examples
above.

Secondary (and this is an actual robust finding, not probe-dependent):
**adding a non-evidenced field to `PersonaV1` degraded groundedness
coverage on the evidenced fields until the SYSTEM_PROMPT was updated
to explicitly exclude it from citation.** Partial mitigation confirmed.
Worth flagging for anyone extending the schema similarly.

## Recommendation: **defer**

Do not adopt or reject the `not_this` hypothesis from this run. Before
a ship/kill call:

1. **Redesign the probe set.** At least 5 refusal prompts per persona
   that target identity territory **not covered by existing fields**
   (habits, values, role boundaries, aesthetic commitments). Plus 5
   *control* prompts that target nothing in `not_this` — to catch
   over-refusal on unrelated topics.
2. **Re-run the A/B with the new probe set.** Baseline persona JSONs
   from commit `c53fc0a` are still available; baseline twin code is
   at `c53fc0a:twin/twin/chat.py`. Same extraction pattern as this
   run. Cost estimate: ~$0.02.
3. **Wait on judge-scored refusal quality.** Blocked on
   `evaluation/evaluation/judges.py` no longer being a scaffold. This
   is the real target metric. A cross-model judge (Sonnet or Opus
   judging Haiku twins) is strongly preferred over Haiku-on-Haiku
   because of self-preference bias.
4. **Revisit the `not_this` / `objections` boundary in the prompt.**
   Persona 01 bled objection content into `not_this`. Consider making
   the SYSTEM_PROMPT bullet more adversarial ("if the item could
   appear in a contract negotiation, it belongs in `objections`, not
   `not_this`").
5. **Consider whether the experiment should be narrower.** If
   `not_this` cannot be shown to add lift over implicit field-based
   scaffolding on carefully-designed probes, the right call may be
   to kill the field and instead invest in sharpening the quality
   bar on `goals`/`pains`/`motivations` to be more identity-shaped.
6. **Reproduce the groundedness flakiness finding** on 2–3 more
   clusters to confirm it's a systematic effect of adding
   non-evidenced fields, and to quantify whether the mitigation
   fully closes the gap or only reduces it.

## Caveats
- **N=2 probes, N=2 personas.** The sample size is too small to
  reject the hypothesis conclusively even with the observed null —
  a real effect of moderate size would be invisible here.
- **Probe-field overlap confound.** As discussed above, the probes
  targeted things already in `goals`/`pains`. The experiment was not
  isolated.
- **Different synthesis runs.** Baseline and experimental personas
  are two distinct Haiku runs on the same clusters. Names and content
  differ. Both personas in each condition are thematically close but
  not identical — some of the qualitative differences in refusal
  style may be persona-to-persona variance rather than
  condition-to-condition variance.
- **Eyeball-grade judgement.** I read the replies and judged them
  indistinguishable. A judge might disagree, or find consistent
  small differences I missed. This is the weakest form of evidence
  and is only usable to say "no obvious effect", not "no effect".
- **Self-preference bias.** If a Haiku judge were used on these
  replies, it would likely favor the experimental condition even
  without a real quality difference. Cross-model judging required.

## Cost
- Synthesis (2 personas, retries included): $0.0784
  - Cluster 0 (`Dev-First Engineering Lead`, 3 attempts): $0.0584
  - Cluster 1 (`Creative Entrepreneur`, 1 attempt): $0.0201
- Twin demo replies (pipeline's built-in probe, 2 replies): $0.0032
- Initial refusal smoke test (2 experimental replies): $0.0036
- Baseline A/B comparison (4 replies: 2 baseline + 2 experimental): $0.0068
- **Total: ~$0.092** (well under the experimental cap)

Prior failed pipeline run (3/3 attempts on cluster 1, pre-mitigation)
is not itemized.

## Open questions (deferred)
- Should the twin cite the specific matching `not_this` item when
  refusing? Would make the scaffold visible and testable; would also
  make the twin feel less natural.
- Should `not_this` eventually be added to `EVIDENCE_REQUIRED_FIELDS`?
  Would require a different evidence notion — perhaps "absence of
  signal" citations rather than positive record references.
- Does the groundedness-starvation effect reproduce with other
  non-evidenced additions, or is it specific to `not_this`'s
  semantic adjacency to `objections`?
- If a well-designed A/B still shows null, is the right response to
  kill the field or to change the twin prompt to reference `not_this`
  items by name in the Rules block (a stronger form of injection)?
