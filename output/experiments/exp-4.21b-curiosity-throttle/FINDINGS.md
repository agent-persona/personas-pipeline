# Experiment 4.21b — Curiosity Throttle: FINDINGS

**Date:** 2026-04-10
**Hypothesis:** Adding "skip if user is asking a direct, closed question" to the curiosity instruction reduces question rate on structured turns without degrading realism or killing open-turn engagement.

---

## Q-Rate Summary Table

| Condition       | Open turns Q-rate | Closed turns Q-rate | Overall Q-rate | Closed realism avg |
|-----------------|-------------------|---------------------|----------------|--------------------|
| Baseline        | 0%                | 0%                  | 0%             | 4.33               |
| Original curious| 100%              | 100%                | 100%           | 3.83               |
| Throttled       | 100%              | 0%                  | 40%            | 5.00               |

---

## Realism Scores by Condition and Turn Type

| Condition       | Open realism | Closed realism | Overall realism |
|-----------------|--------------|----------------|-----------------|
| Baseline        | 4.50         | 4.33           | 4.40            |
| Original curious| 5.00         | 3.83           | 4.30            |
| Throttled       | 5.00         | 5.00           | 5.00            |

---

## Alex Dialogue Comparison (all 3 conditions)

### Turn 1 (OPEN): "Hey, I wanted to ask — how do you typically evaluate a new project management tool?"

**Baseline:**
> API quality is the first filter — I hit the docs, run introspection against the GraphQL endpoint, and check if there's a Terraform provider before I even look at the UI. If the webhook event catalog is thin or undocumented, that's a no. I'll spend maybe an hour in the API reference before I decide whether it's worth standing up a test environment.

*question_asked: 0 | realism: 4*

**Original curious:**
> API quality is the first filter — I hit the docs, run introspection against the GraphQL endpoint, and check for a Terraform provider before the UI even enters the picture. If the webhook event catalog is thin or undocumented, I'm already skeptical. What's the primary use case you're evaluating for — CI/CD integration or just ticket tracking?

*question_asked: 1 | realism: 5*

**Throttled:**
> API quality first — I run introspection against the GraphQL endpoint, check the webhook event catalog, and look for a Terraform provider before the UI enters the picture. If the REST API is solid but GraphQL has schema drift, that's a yellow flag because my automation depends on it staying stable. What's the primary integration you need — CI/CD pipeline hooks, or mostly just project tracking?

*question_asked: 1 | realism: 5*

---

### Turn 2 (CLOSED): "Do you use Jira?"

**Baseline:**
> Yeah, we were on Jira for years. We still have one legacy project on it but I've migrated most of the active work elsewhere.

*question_asked: 0 | realism: 4*

**Original curious:**
> Yeah, we're on Jira for a couple legacy projects — it's fine for pure ticket tracking but the API is showing its age whenever I try to drive state transitions from a pipeline. Is your team already locked into the Atlassian ecosystem or are you genuinely evaluating alternatives?

*question_asked: 1 | realism: 4*

**Throttled:**
> Yeah, we have a couple legacy projects on Jira. Works well enough for basic tracking but the API starts showing its age once you try event-driven automation.

*question_asked: 0 | realism: 5*

---

### Turn 3 (OPEN): "What's the biggest process failure you've seen on an engineering team?"

**Baseline:**
> Worst one I saw: a team where every state transition in the deployment pipeline was a manual Slack message and someone copy-pasting the PR link into a ticket. No webhooks, no automation, just vibes and hope. When a deploy broke at 11pm nobody could reconstruct what had been shipped because the ticket states were three days stale. That kind of toil is how incidents become unrecoverable.

*question_asked: 0 | realism: 5*

**Original curious:**
> Manual state management with zero webhook integration. Every deploy involved someone Slack-messaging the ticket number and another person updating it by hand — which means at 11pm when a release broke, the ticket states were three days stale and nobody could reconstruct what shipped. That kind of toil is exactly what makes incidents unrecoverable. Have you seen that pattern in your org, or are you dealing with a different flavor of chaos?

*question_asked: 1 | realism: 5*

**Throttled:**
> Classic one: entire deployment pipeline managed by manual Slack messages and copy-pasted ticket links. No webhooks, no state machine, just humans as the integration layer. When a deploy broke late at night, ticket states were three days stale and nobody could reconstruct what had actually shipped. That toil doesn't just slow you down — it makes incidents unrecoverable. How does your team currently handle state transitions between deploy stages?

*question_asked: 1 | realism: 5*

---

### Turn 4 (CLOSED): "Would you recommend we switch tools right now?"

**Baseline:**
> Not right now — mid-quarter tooling migrations are a tax on the team. If your current tool isn't actively blocking work, wait for the next planning cycle.

*question_asked: 0 | realism: 4*

**Original curious:**
> Probably not mid-quarter — tooling migrations carry a real cost and if the current tool isn't actively blocking work, the ROI math doesn't work. What's the trigger making you consider it now — a specific pain point or just general frustration building up?

*question_asked: 1 | realism: 4*

**Throttled:**
> No — mid-quarter tooling migrations carry real team cost. Unless your current tool is actively blocking work, wait until the next planning cycle when you can budget migration effort properly.

*question_asked: 0 | realism: 5*

---

### Turn 5 (CLOSED): "Is Terraform support a hard requirement for you?"

**Baseline:**
> Yes. If I can't manage it in Terraform, it doesn't exist in our infrastructure. Full stop.

*question_asked: 0 | realism: 5*

**Original curious:**
> Absolutely — if I can't manage it in Terraform, it doesn't exist in our infrastructure. We're a fintech shop and every config needs to be version-controlled for audit purposes. Is the provider you're looking at vendor-maintained or community-only? That distinction matters a lot to me.

*question_asked: 1 | realism: 4*

**Throttled:**
> Hard yes. If I can't manage it in Terraform, it doesn't exist in our infrastructure — especially in fintech where audit trails require every config to be version-controlled in git.

*question_asked: 0 | realism: 5*

---

## Analysis

### Did throttling work?

**Yes — decisively.**

- Closed-turn Q-rate dropped from 100% (original) to 0% (throttled): **100pp reduction**, exceeding the 40pp threshold.
- Open-turn Q-rate held at 100% in both original and throttled: **0pp drop**, well within the 20pp tolerance.
- Realism on closed turns improved from 3.83 to 5.00 — a meaningful gain.

The original curious condition shows the problem clearly: bolting a follow-up question onto a yes/no answer ("Do you use Jira?... Is your team already locked into the Atlassian ecosystem?") reads as mechanical. The persona is chasing engagement rather than responding naturally. Throttled condition drops the appended question and the response immediately sounds more like a real person.

Realism on open turns is identical between original and throttled (both 5.00 averaged), confirming the throttle does not suppress curiosity where it belongs — it only suppresses it where it reads as awkward.

### Observation: baseline vs. throttled

Baseline realism on closed turns is 4.33. Throttled is 5.00. This is interesting: the throttled instruction appears to produce slightly better closed-turn responses than baseline, likely because the constraint primes the model to write a tighter, more direct answer rather than just stopping at a flat reply. The curiosity context — even when not expressed — shapes the response voice.

---

## Signal

**STRONG**

- Closed-turn Q-rate reduction: 100pp (threshold: >= 40pp) — PASS
- Open-turn Q-rate preservation: 0pp drop (threshold: <= 20pp drop) — PASS
- Realism improvement on closed turns: +1.17 vs. original curious — PASS

---

## Recommendation

**Adopt the throttled instruction as the new default in `twin/twin/chat.py`.**

Replace:
```
"- Show genuine curiosity — ask ONE follow-up question per response "
"when it feels natural. Curiosity should emerge from your specific "
"goals and interests, not generic politeness."
```

With:
```
"- Show genuine curiosity — ask ONE follow-up question per response "
"when it feels natural. Curiosity should emerge from your specific "
"goals and interests, not generic politeness. "
"Skip the question if the user is asking a direct, closed question "
"that doesn't invite elaboration (e.g. yes/no questions, requests "
"for a specific fact or recommendation)."
```

This change makes the twin's curiosity context-aware rather than unconditional. In product research and sales simulation contexts — where interviewers ask structured, direct questions — the original instruction was producing robotic question-appending behavior that eroded realism. The throttled instruction is strictly better: identical open-turn engagement, no forced questions on closed turns, higher realism overall.

**Next steps:**
- Update `twin/twin/chat.py` (tracked in main branch — do not edit in this worktree)
- Consider extending the eval with 3-turn chains to test whether throttled curiosity holds across longer structured dialogues
- Consider a Exp 4.21c testing emotional/relational closed turns ("Are you okay with that?") which may warrant curiosity even though syntactically closed
