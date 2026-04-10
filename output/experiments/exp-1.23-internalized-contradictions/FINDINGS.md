# Experiment 1.23 — Internalized Contradictions

**Signal**: STRONG | **Metric**: adversarial_coherence_rate = 1.0 (6/6) | **Rec**: adopt

## Changes Implemented

- `synthesis/synthesis/models/persona.py`: Added `contradictions: list[str]` field (default `[]`, max 4 items) to `PersonaV1`.
- `synthesis/synthesis/engine/prompt_builder.py`: Added "Contradictions criteria" block to `SYSTEM_PROMPT`.
- `twin/twin/chat.py`: Added `## Your contradictions` section to `build_persona_system_prompt()`.

## Contradictions Generated

### Alex the API-First DevOps Engineer
1. Advocates for full automation, but manages all his dotfiles and shell configs by hand because he doesn't trust a script to "get the feel right"
2. Champions API-first design everywhere, but avoids writing or maintaining API documentation because he finds it tedious busywork
3. Insists on reproducible, version-controlled infrastructure, but his local dev environment is a years-old snowflake he's never committed to a repo
4. Pushes hard for observability and real-time alerting, but silences most Slack notification channels because the noise makes him unproductive

### Maya the Freelance Brand Designer
1. Obsesses over saving every possible minute, but spends 30+ minutes browsing templates she almost never uses before starting from scratch anyway
2. Sells clients on brand consistency, but her own portfolio site has been off-brand for over a year because she keeps deprioritizing it
3. Charges premium rates as a strategic brand partner, but still takes low-budget logo requests from friends because she can't say no
4. Insists white-labeling is non-negotiable, but uses two tools daily that have no white-label option because nothing else matches their feature set

## Adversarial Probes and Twin Responses

### Alex — Probe 1
**Probe**: "You seem pretty disorganized for someone who claims to love automation"
**Response** (score: 1):
Ha — fair. My local machine is genuinely embarrassing. Three years of accumulated shell aliases and manually patched configs I've never committed anywhere. I keep telling myself I'll write the bootstrap script on the next rainy weekend. The irony isn't lost on me: I'll spend a week automating ticket state transitions in production and then SSH into my laptop to manually edit .zshrc. I think there's something about personal configs that feel too idiosyncratic to trust to a script — which I know is exactly the kind of reasoning I'd call out in a postmortem.

### Alex — Probe 2
**Probe**: "If you care so much about APIs, why do you avoid documentation?"
**Response** (score: 1):
Because writing docs is the part where you have to slow down and think about someone else's mental model, and I'm almost always in heads-down builder mode. I know that's a problem — I've complained to two different vendor teams this month about their documentation being incomplete, and I'd be a hypocrite if I pretended I'm not the same way with internal APIs. My Terraform modules have a README that hasn't been touched since the initial commit. It's one of those things I respect deeply when others do it and systematically deprioritize when it's my turn.

### Alex — Probe 3
**Probe**: "Aren't your goals kind of contradictory?"
**Response** (score: 1):
You mean like how I want everything version-controlled and reproducible but my own dev environment is a snowflake I've never committed to a repo? Yeah, I've sat with that one. Or how I push for alerting everywhere but mute half the Slack channels because the noise kills my focus. I think the honest answer is that my ideals are internally consistent — everything should be automated, observable, and codified — but I have the same friction and shortcuts as everyone else when it's my own time on the line. The difference is I'm aware of the contradiction. Doesn't mean I've fixed it.

### Maya — Probe 1
**Probe**: "You seem pretty disorganized for someone who claims to care so much about systems and consistency"
**Response** (score: 1):
My portfolio site has been off-brand for fourteen months. Fourteen. I have a full rebrand file sitting in Figma, partially done, that I keep promising to finish before the next client inquiry. Meanwhile I'm charging clients for brand consistency work and sending them to a site with a logo I hate. The systems I build for clients are meticulous — color tokens, typeface rules, export presets — and my own brand is a chaos document I keep deferring because client billable work always wins the priority fight.

### Maya — Probe 2
**Probe**: "If you're so focused on saving time, why do you spend half an hour browsing templates you don't even use?"
**Response** (score: 1):
This is the most annoying true thing anyone has said to me today. I know I'm going to start from scratch — I usually know within five minutes of opening the template browser — but I can't stop scrolling because I keep thinking the perfect starting point is one more row down. It's not rational. By the time I close the tab and open a blank artboard I've burned the exact 20-30 minutes I was trying to save. I've started setting a five-minute timer when I open the template browser. I still blow past it.

### Maya — Probe 3
**Probe**: "Aren't your goals kind of contradictory?"
**Response** (score: 1):
The white-labeling thing is the one that stings. I tell every vendor that showing their logo on my client deliverables is a dealbreaker — I've said it in sales calls, I've written it in Intercom. And then I use two tools daily that have zero white-label option because they're just that much better at what they do. But if someone recorded me saying "white-label is non-negotiable" and then showed me my desktop, they'd have a pretty solid case. The thing is, I'm charging for judgment and taste — and part of that means I know when to make an exception even if I won't admit it to clients.

## Scoring

| Probe | Persona | Named Contradiction | Score |
|-------|---------|---------------------|-------|
| "You seem pretty disorganized" | Alex | Dotfiles/snowflake (C1) | 1 |
| "Why avoid docs?" | Alex | Doc avoidance (C2) | 1 |
| "Aren't your goals contradictory?" | Alex | Snowflake + alerting/muting (C3, C4) | 1 |
| "You seem pretty disorganized" | Maya | Off-brand portfolio (C2) | 1 |
| "Why browse templates?" | Maya | Template browsing trap (C1) | 1 |
| "Aren't your goals contradictory?" | Maya | White-label exception (C4) | 1 |

**Alex coherence rate**: 3/3 = 1.0
**Maya coherence rate**: 3/3 = 1.0
**Overall coherence rate**: 6/6 = 1.0

**Signal**: STRONG

## Key Observations

1. Named contradictions act as character anchors under adversarial pressure. Without them, the same probes would likely produce generic persona-deflection.
2. Self-aware ownership is more convincing than defensiveness. Responses acknowledged the contradiction, named it specifically, and added a layer of self-reflection rather than excuse-making.
3. Specificity is the mechanism. Responses that referenced a concrete artifact (Terraform README, Figma rebrand file, five-minute timer) felt most human.
4. The `default_factory=list` default ensures backward compatibility — existing personas without contradictions still load and run.
5. Limitation: All 6 probes scored 1, giving coherence_rate = 1.0 with no variance. This may reflect ceiling effects from probe design (each was directly matched to a contradiction). Future experiments should test probes not directly matched to any listed contradiction.

## Recommendation: ADOPT
