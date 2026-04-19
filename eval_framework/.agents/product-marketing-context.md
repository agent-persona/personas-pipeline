# Product Marketing Context

*Last updated: 2026-04-17 — V2 with owner inputs on model, target, and goals.*

## Product Overview
**One-liner:** A structured evaluation framework for AI-generated personas — runs 52 scorers across 4 tiers to detect alignment failures, demographic bias, and LLM-default inflation patterns.
**What it does:** Persona Eval grades synthetic personas (the kind you'd feed into LLM agents, market-research simulations, or AI roleplay) on schema validity, factual coherence, narrative depth, and bias. It surfaces *where* a persona is shallow, biased, or LLM-defaulty — not just a single score.
**Product category:** LLM evaluation / synthetic persona QA. Adjacent shelf: LLM-evals tooling (LangSmith, Braintrust, Promptfoo, Langfuse, Inspect AI).
**Product type:** Open-source Python framework / dev tool.
**Business model:** Fully open-source. Growth goal is adoption (stars + users), not revenue. Monetization TBD — community-first.

## Target Audience
**Target companies:**
- AI/LLM product companies building agents, copilots, or simulation
- Any team generating or consuming user profiles/personas (product, UX, research)
- Applied-research labs and solo founders shipping LLM features

**Decision-makers:**
- AI eval engineers (primary — technical user + champion)
- Product/UX people who want deeper insight into their user profiles
- Solo founders building AI products (all roles in one)

**Primary use case:** Catching when LLM-generated personas are biased, shallow, or hallucinate facts — *before* those personas drive downstream agent behavior or research conclusions.

**Jobs to be done:**
- "Help me trust the synthetic personas I'm using for testing/research"
- "Show me which dimensions of my persona generation pipeline are broken"
- "Give me a rigorous, citable methodology I can defend to my team / customers / reviewers"

**Use cases:**
- Eval gate in a synthetic-persona generation pipeline (CI for personas)
- Bias audit before publishing synthetic-persona-driven research
- Comparing prompt strategies for persona generation across LLM vendors
- Benchmarking commercial persona-gen tools

## Personas
| Persona | Cares about | Challenge | Value we promise |
|---------|-------------|-----------|------------------|
| AI Eval Engineer (champion+user) | Reproducible, granular metrics, easy to extend | Their team is shipping LLM features blind on persona quality | 52 scorers out of the box, plug-in BaseScorer for custom dims |
| Head of Applied Research (decision maker) | Defensibility, bias coverage, audit trail | Reviewers / customers / regulators ask "how do you know your synthetic data isn't biased?" | Tier-4 bias scorers + audit dashboard you can show |
| Solo founder building AI persona tool (all-in-one) | Move fast, look credible | Can't justify building eval infra from scratch | Drop-in framework, real reports, instant credibility |


## Problems & Pain Points
**Core problem:** Synthetic personas from LLMs look plausible but are often biased (LLM defaults), shallow (no real behavioral grounding), or factually incoherent — and nobody has a rigorous way to measure that. Teams ship features or publish research on top of broken synthetic data without knowing.

**Why alternatives fall short:**
- **General LLM eval tools** (LangSmith, Braintrust, Promptfoo) — built for prompt/output eval, not persona-specific dimensions like demographic distributional realism or hedge/inflation bias
- **Roll-your-own scorers** — every team rebuilds the same thing badly; no shared methodology
- **Manual review** — doesn't scale, can't catch distributional bias across a batch

**What it costs them:** Wasted compute, wrong product decisions from biased synthetic users, reputational risk if research gets called out, weeks rebuilding eval infra.

**Emotional tension:** "I don't actually know if my personas are good. I'm hoping nobody asks."

## Competitive Landscape
**Direct:** [GUESS] No direct head-to-head competitor — closest is internal eval scripts at companies like Synthesia, Tavus, Persona AI. Confirm if you've seen others.
**Secondary:** LangSmith, Braintrust, Promptfoo, Langfuse, Inspect AI — falls short because they're generic LLM-output eval, not persona-aware (no demographic distribution checks, no JTBD coherence, no bias-by-design scorers).
**Indirect:** Manual QA, "looks good to me" review — falls short because it doesn't scale and misses systematic LLM defaults that humans don't notice.

## Differentiation
**Key differentiators:**
- 52 scorers spanning 4 tiers (schema → factual → depth → bias) — broader coverage than any single eval tool
- Tier-4 bias scorers are *designed to fail* — they specifically detect LLM register inflation, hedging, and balanced-opinion patterns
- Tiered gating — invalid personas don't pollute downstream metrics
- Set-level scorers — catches diversity/distribution problems across a batch, not just per-persona
- Audit dashboard — every metric in `details` is inspectable; you can defend any score

**How we do it differently:** Persona-first, not prompt-first. Treats the persona itself as the artifact under test, with research-backed dimensions per tier.
**Why that's better:** You get specific, actionable failure modes ("D45 register inflation: 0.3") instead of a generic "score: 0.7."
**Why customers choose us:** Out-of-the-box rigor + extensibility. They get a defensible methodology in an afternoon instead of a quarter.

## Objections
| Objection | Response |
|-----------|----------|
| "We can build this ourselves." | Sure — but we already shipped 52 scorers + 303 tests + a research-backed bias methodology. Reproduce that and it's a quarter of work. |
| "How do we know your scorers are right?" | Every scorer's metrics live in `details` and the audit dashboard. Pruning verdicts (KEEP / REMOVE) are documented. Nothing is a black box. |
| "We use LangSmith / Braintrust already." | Great — those eval the prompts and outputs. We eval the *personas themselves*. Complementary, not competing. |
| [NEED INPUT — top objection you've actually heard] | |

**Anti-persona:** [GUESS] Teams generating personas with no downstream consequence (e.g., one-off marketing brainstorm). They don't need this rigor.

## Switching Dynamics
**Push:** "I built a quick eval script and it's already 500 lines. This is going to consume my quarter."
**Pull:** "There's already a framework with 52 scorers, bias detection, and a dashboard? Why am I building this?"
**Habit:** Already wrote some scoring code; sunk cost feels real.
**Anxiety:** "Will I have to rewrite my generation pipeline to fit theirs?" (Answer: no — pass JSON in, get JSON out.)

## Customer Language
[NEED INPUT — paste verbatim quotes from any conversations you've had: Slack DMs, calls, GitHub issues, X replies. This is the highest-leverage section to fill in.]

**Words to use:** [GUESS] eval, scorer, dimension, bias, distribution, audit, defensible, rigor, drop-in
**Words to avoid:** [GUESS] "AI-powered," "revolutionary," "next-gen" — too generic
**Glossary:**
| Term | Meaning |
|------|---------|
| Scorer | A single evaluator producing pass/fail + 0–1 score + metric details |
| Tier | One of 4 evaluation layers (schema, factual, depth, bias) with gating |
| Set-level scorer | Operates on a batch of personas (e.g., diversity checks) |
| EvalResult | The scorer's output: passed, score, details dict |
| Inflation bias | LLM tendency to over-formalize, over-hedge, or over-balance opinions |

## Brand Voice
[GUESS — adjust]
**Tone:** Direct, technically rigorous, no hype.
**Style:** Show the work. Code snippets, real metrics, no marketing fluff.
**Personality:** Researcher who ships. Skeptical of LLM defaults. Respects the reader's time.

## Proof Points
**Metrics:**
- 52 scorers, 4 tiers
- 303+ tests
- 12-persona golden dataset
- Bias dimensions (D45/D46/D47) with pruning verdicts documented

**Customers:** [NEED INPUT] None yet? Early users? Beta installs?
**Testimonials:** [NEED INPUT]
**Value themes:**
| Theme | Proof |
|-------|-------|
| Rigorous out of the box | 52 scorers + 4-tier gating + audit dashboard |
| Bias-aware | Designed-to-fail Tier 4 bias scorers (D45/D46/D47) |
| Extensible | BaseScorer subclass = new dimension in <50 lines |

## Goals
**Business goal:** 100 GitHub stars + grow active users / installs.
**Conversion action:** Star the repo → pip install → run on their own personas → share results.
**Current metrics:** [NEED INPUT — current star count, PyPI installs if published, any traffic data]
