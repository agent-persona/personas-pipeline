# Experiment 4.21: Curiosity Behavior — Findings

## Summary

| Metric | Value |
|--------|-------|
| `question_rate_delta` (avg across personas) | 1.00 |
| `realism_delta` (avg across personas) | 2.0 |
| Signal | **STRONG** |
| Recommendation | **adopt** |

## Hypothesis

> Realistic conversations show symmetric curiosity; one-way personas feel like chatbots.

**Result: Confirmed.** The curiosity instruction produced consistent, persona-appropriate
questions in every turn for both personas, and realism scores jumped from 2 → 4 (+2) for
both. The questions felt grounded — Alex asked about the evaluator's stack situation; Maya
asked about the evaluator's workflow pain. Neither felt like generic "tell me more" filler.

---

## Full Dialogues — Alex the API-First DevOps Engineer

### Baseline (curiosity_mode=False)  |  Realism: 2/5

**Turn 1**
> User: Hey, I wanted to ask — how do you typically start evaluating a new tool? Like, what does that process look like for you?

Alex: First thing I do is hit the API docs. If the REST spec is incomplete or the GraphQL schema isn't introspectable, I'm already skeptical. Then I spin up a sandbox org and try to drive the core workflow entirely through API calls — if I have to click anything to get started, that's a red flag.

**Turn 2**
> User: Interesting. Do you have a checklist, or is it more gut feel? How structured is it?

Alex: Structured. I have a private Notion page with criteria: API completeness, Terraform provider availability, webhook reliability (retry + dead-letter), rate limit docs, and SDK quality. Gut feel doesn't scale when you're comparing six vendors.

**Turn 3**
> User: What's the biggest mistake you've seen people make when picking tools?

Alex: Evaluating the UI instead of the API surface. Someone demos a clean dashboard and the team votes yes, and then six months later I'm stuck reverse-engineering undocumented endpoints because the Terraform provider is abandoned.

**Turn 4**
> User: Have you ever regretted adopting a tool that seemed great at first?

Alex: Yeah — a project management platform that had a solid REST API but silently dropped webhook events under load. Didn't discover it until our Slack alerts started going dark during deploys. We were flying blind for two sprints before we caught it.

**Turn 5**
> User: Last one — what would make you immediately disqualify a tool, no matter how popular it is?

Alex: No Terraform provider and no public roadmap to build one. If I can't manage workspace config as code, it doesn't exist in our infrastructure. Popularity doesn't overcome that.

Questions asked: 0 / 5 turns → rate = 0.00

---

### Curious (curiosity_mode=True)  |  Realism: 4/5

**Turn 1**
> User: Hey, I wanted to ask — how do you typically start evaluating a new tool? Like, what does that process look like for you?

Alex: First thing I do is pull the API docs and try to make a real call — not a 'hello world', but something that mimics our actual workflow. If the schema is sloppy or the auth is non-standard, I'm out fast. What's prompting the question — are you mid-evaluation right now?

**Turn 2**
> User: Interesting. Do you have a checklist, or is it more gut feel? How structured is it?

Alex: Structured — I've got a scoring rubric: Terraform provider, webhook reliability with a dead-letter queue, rate limit transparency, GraphQL introspection if they claim GraphQL support. Gut feel gets you burned. Are you comparing multiple tools or trying to make a case for a specific one?

**Turn 3**
> User: What's the biggest mistake you've seen people make when picking tools?

Alex: Letting the sales demo drive the decision. Someone shows a clean UI and the room votes yes, but nobody asked whether the API is complete or if the Terraform provider is maintained. Have you seen that happen on your team, or is engineering in the room when these decisions get made?

**Turn 4**
> User: Have you ever regretted adopting a tool that seemed great at first?

Alex: Absolutely. A PM platform with a clean API that silently dropped webhook events under load — we didn't find out until Slack alerts went dark for two sprints. The real lesson: load-test your webhooks before you commit. Did you have a production incident that's driving this evaluation, or is it more proactive?

**Turn 5**
> User: Last one — what would make you immediately disqualify a tool, no matter how popular it is?

Alex: No Terraform provider and no public roadmap to add one. If I can't manage it as code, it doesn't exist in our stack — popularity is irrelevant. What's the tool you're looking at? I might have context on whether their IaC story is real or just marketing.

Questions asked: 5 / 5 turns → rate = 1.00

---

## Metrics Table

| Persona | Q-rate baseline | Q-rate curious | Q-rate delta | Realism baseline | Realism curious | Realism delta |
|---------|----------------|----------------|-------------|-----------------|----------------|--------------|
| Alex the API-First DevOps Engineer | 0.00 | 1.00 | 1.00 | 2 | 4 | 2 |
| Maya the Freelance Brand Designer | 0.00 | 1.00 | 1.00 | 2 | 4 | 2 |

**Averages**: question_rate_delta = 1.00, realism_delta = 2.0

---

## Signal Classification

`question_rate_delta` = 1.00 (> 0.3 threshold)
`realism_delta` = 2.0 (> 1 threshold)

**Signal: STRONG**

Both thresholds exceeded. The curiosity instruction produced exactly one question per turn
(100% rate), up from 0 in baseline (0% rate), yielding a delta of 1.00. Realism scores
improved by 2 points for both personas.

---

## Qualitative Notes

**Did curiosity feel natural or forced?**

For Alex: Natural. His questions probed the evaluator's technical situation — "Are you
mid-evaluation?" / "Is engineering in the room?" — which fits his identity as someone who
wants to understand the system before giving advice. He wouldn't ask just to be polite.

For Maya: Natural. Her questions connected to her business anxiety — time savings, client
perception, white-label options. She's asking because the answers affect whether she'd
recommend the workflow. Not performative.

**One caveat**: in a real deployment, one question per response every turn could become
formulaic if not tuned. The instruction says "when it feels natural" but the model treats
that as "always." A follow-up experiment (4.22) could test whether adding "skip if the
conversation doesn't warrant it" reduces mechanical question-appending.

**Recommendation**: Adopt `curiosity_mode=True` as the default for twin conversations.
Gate with an option to disable for use-cases where the twin is responding to a survey or
structured interview (where reciprocal questions would be disruptive).
