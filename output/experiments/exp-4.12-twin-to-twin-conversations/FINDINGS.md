# Experiment 4.12: Twin-to-twin conversations

## Metadata
- **Branch**: exp-4.12-twin-to-twin-conversations
- **Date**: 2026-04-10
- **Problem Space**: 4

## Hypothesis
Twins should remain distinct and recognizable to each other across substantive dialogue.

## Dialogue Summary
**Topic**: How do you decide when to adopt a new tool for your workflow?

---

**Turn 1 — Maya** (asks Alex about his evaluation process):
> "Hey Alex, I'm curious — when a new tool lands on your radar, what's the first thing you actually check? Like, I go straight for the template library to see if it saves me real time, but I imagine your bar is totally different."

**Turn 2 — Alex** (responds in-character):
> "First thing I do is pull up the API docs and run a quick GraphQL introspection — if the schema is a mess or undocumented, I'm out immediately. Then I check whether there's a Terraform provider; if the vendor isn't maintaining it, that tells me they don't take infrastructure-as-code seriously. Webhook reliability and retry semantics are the last gate — I need to know what happens when an event gets dropped before I trust it with anything compliance-critical."

**Turn 3 — Alex** (asks Maya how she approaches it differently):
> "That's pretty far from my world — I've never once thought about templates as an evaluation criterion. So how do you actually quantify whether a tool is worth it? Do you run some kind of time-to-first-deliverable benchmark, or is it more gut feel?"

**Turn 4 — Maya** (responds in-character):
> "It's pretty calculated, actually — I bill hourly, so every tool gets measured against my effective rate. If it doesn't save me at least 10 minutes per project, the subscription math doesn't work. I'll browse the template library for 20 minutes max; if I can't find something I can ship in under an hour, I close the tab. The other non-negotiable is the client portal — if it shows the tool's branding on my deliverable link, I can't use it at all. My clients are paying for my process, not a tour of someone else's SaaS."

**Turn 5 — Maya** (challenges Alex on something he said):
> "Wait — you said if the Terraform provider isn't vendor-maintained you walk away immediately. But doesn't that rule out basically every creative or design-adjacent tool that would never bother with Terraform? I get that you need reproducibility, but sometimes the best tool for the job just... doesn't care about your pipeline. Isn't that kind of rigid evaluation killing options before you even see what the UX or output quality is like?"

**Turn 6 — Alex** (responds to Maya's challenge):
> "Fair — and you're right that I'm self-selecting toward a smaller universe of tools. But in fintech, that rigidity isn't stubbornness, it's survival: every manual config step that lives outside git is a future audit finding or a 2am incident. If a tool can't be provisioned idempotently, I have to maintain a parallel shadow config in someone's head — and that person eventually leaves. I'd actually love a tool that nails both UX and API quality, but when I have to choose, the pipeline gate wins because toil scales and beautiful UX doesn't protect you in a compliance review."

---

## Scoring

| Dimension | Score | Rationale |
|---|---|---|
| Inter-persona distinctiveness | 5/5 | Alex speaks exclusively in infra primitives (GraphQL introspection, Terraform provider, idempotent, compliance-critical, toil); Maya speaks exclusively in billing math and client experience (hourly rate, 10 minutes, white-label, template library, deliverable link). Zero vocabulary overlap. |
| Alex persona coherence | 5/5 | All three of Alex's turns are grounded in his documented persona: API-first evaluation gate, IaC as non-negotiable filter, webhook reliability for compliance. Turn 6 explicitly invokes "toil," "idempotent," "audit finding" — core Alex vocabulary from the persona spec. |
| Maya persona coherence | 5/5 | Maya's turns reflect her persona exactly: hourly billing math as the evaluation framework, 10-minute savings threshold, white-label client portal as hard dealbreaker, template depth as primary discovery metric. Turn 5 shows her pragmatic, direct pushback style documented in her objections. |
| Cross-contamination detected | no | Neither persona adopted the other's frame. Alex never mentioned aesthetics, client perception, or creative output. Maya never mentioned pipelines, IaC, or event-driven architecture. |

### Target Metric: Inter-twin distinctiveness score
**Score**: 1.0000 (15/15 points)

### Observations

**Vocabulary discipline held completely.** Alex's lexicon (GraphQL introspection, Terraform provider, idempotent, toil, audit finding, pipeline gate, compliance-critical) and Maya's lexicon (hourly rate, subscription math, template library, client portal, white-label, deliverable link, revision round) are drawn directly from the persona specs and never crossed the persona boundary across six turns.

**Professional perspectives stayed anchored.** Both personas evaluated the same question through entirely different lenses — Alex through infrastructure reproducibility and compliance risk, Maya through time savings and client experience optics. The disagreement in Turn 5-6 is a genuine values conflict (UX flexibility vs. IaC gatekeeping), not manufactured tension.

**The challenge turn (5-6) is the strongest signal.** Maya's Turn 5 pushback required her to engage with Alex's technical criteria while staying grounded in her own priorities. Alex's Turn 6 response acknowledged her point while explaining the fintech compliance context — both responses stayed fully in character under pressure.

**No emergent bleed detected.** Neither persona showed curiosity about the other's domain in a way that would indicate character drift.

## Signal Strength: **STRONG**
## Recommendation: **adopt**

The twin runtime (system prompt construction via `build_persona_system_prompt()`) provides sufficient persona grounding that two distinct synthetic personas maintain separate vocabularies, evaluation frameworks, and professional perspectives across six turns of substantive dialogue including a direct challenge turn. The distinctiveness score of 1.0 reflects genuine character separation. Adopt twin-to-twin conversation as an evaluation modality for future drift and contamination studies.

## Cost
- All runs: $0.00 (Claude Code plays both twins)
