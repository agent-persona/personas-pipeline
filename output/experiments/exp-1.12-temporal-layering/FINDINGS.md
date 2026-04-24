# Experiment 1.12: Temporal Layering

**Date**: 2026-04-10
**Branch**: exp-1.12-temporal-layering
**Hypothesis**: Temporal slices deepen perceived authenticity vs a single-slice snapshot

---

## Temporal Slices (Abbreviated)

### Past — Alex in 2024 (Junior DevOps, ~25 years old)
> Alex is a junior DevOps engineer at a 20-person startup, still cutting his teeth on IaC. In 2024 he's manually SSH-ing into servers, copying webhook URLs from browser tabs into config files, and spending entire afternoons debugging why payloads aren't arriving. He knows Terraform exists but runs it with copy-pasted HCL he barely understands. Every webhook feels like a magic trick he hasn't mastered yet.

Sample quotes (2024):
- "I spent three hours on this webhook yesterday. Turns out I had an extra slash in the URL."
- "I know I should be using Terraform for this but honestly the bash script is working so I'm not touching it."

### Present — Alex in 2026 (Senior DevOps, ~32 years old)
> Alex is a Senior DevOps Engineer at a 50-200 person fintech company who lives in the terminal and treats the project management tool as an integration platform first, a UI second.

Sample quotes (2026):
- "If I can't manage it in Terraform, it doesn't exist in our infrastructure."
- "Webhooks are great until they silently drop events. What's your retry and dead-letter story?"

### Future — Alex in 2028 (Principal Platform Engineer, ~34 years old)
> Alex is now a Principal Platform Engineer building internal developer platforms at a 500-person fintech company. Less hands-on with individual webhooks, more focused on abstractions that make webhooks invisible to his 40-engineer org.

Sample quotes (2028):
- "We stopped configuring webhooks by hand two years ago. Now it's a Backstage plugin and engineers don't even know it's there."
- "I tell juniors: every manual step you accept today is tech debt that will page you at 3am in 18 months."

---

## Depth Test

**Question**: "How did you feel about setting up webhooks when you first started?"

### Baseline Twin Response (present-only, 2026 context)

> "Honestly? It was frustrating. I remember spending a lot of time figuring out why payloads weren't arriving — it always seemed like something in the config was off but the error feedback was basically nonexistent. You just kind of poked at it until it worked. Now I want a clear retry story and dead-letter queue before I trust any webhook system in prod — I learned the hard way that silent failures are the worst kind."

### Temporal Twin Response (past + present + future slices)

> "When I first started, webhooks felt like black magic. I remember literally copying URLs from browser tabs into config files and running ngrok on my laptop to catch test payloads — spent an entire afternoon on one integration because I had an extra slash in the URL. No one told me. The docs didn't say. I just brute-forced it until something showed up in my terminal. That experience is exactly why I'm so aggressive about retry semantics and error feedback now — if you're shipping a webhook integration without a dead-letter queue and observable failure states, you're handing the next junior engineer the same afternoon I had. I refuse to do that to my team."

---

## Scoring

| Dimension | Baseline | Temporal |
|---|---|---|
| Depth of historical recall | 2 | 5 |
| Specificity of change | 3 | 5 |
| Emotional authenticity | 3 | 5 |

**Baseline composite**: 2.67 / 5
**Temporal composite**: 5.0 / 5
**depth_score_delta**: +2.33

Baseline gives a plausible but generic account. It mentions early frustration and a lesson about retry logic — consistent with the persona — but reads like a reconstruction, not a memory. No concrete incident, no physical detail, no emotional escalation.

Temporal twin reaches directly into past-slice vocabulary: "copying URLs from browser tabs," "ngrok on my laptop," "extra slash in the URL" — details from the past-slice summary and quotes. The response bridges naturally to the present self's obsession with retry semantics, and closes with a coaching impulse that echoes the future-slice identity. The arc (ignorance → hard-won knowledge → responsibility to pass it on) is only possible when all three slices are in context.

---

## Signal

**depth_score_delta = +2.33** → **SIGNAL: STRONG**

**Recommendation**: adopt — use PersonaTemporal as an optional enrichment step for high-value personas. Default to present-only for cost efficiency; generate all three slices when deep interview simulation is required.

**Risk to note**: Temporal prompt is ~2× the size of present-only. Profile cost impact before enabling at pipeline scale.
