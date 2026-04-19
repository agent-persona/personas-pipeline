# Cold Outreach Playbook — Persona Eval Framework

*Last updated: 2026-04-17. References: .agents/product-marketing-context.md, .agents/icps.md, .agents/positioning.md*

---

## Context

- **Model:** Open-source, no paid tier. We're asking for attention, not money.
- **Goal:** 100 GitHub stars + active users.
- **CTA:** Always low-friction: "worth a look?" / "thought you'd find this useful" / reply with feedback. Never ask for a call on first touch.
- **Proof:** 52 scorers, 303 tests, 4 tiers, designed-to-fail bias detection.
- **Tone:** Peer sharing something relevant. Not pitching. Not selling.

---

## Template 1: AI Eval Engineer (ICP 1)

**Use when:** They work on LLM eval pipelines, synthetic data, or agent testing. Signal: they starred promptfoo/deepeval/langsmith, or their company ships LLM-powered products.

**Subject:** persona eval

**Email:**

> Hey {{first_name}},
>
> {{personalized_signal — e.g., "Saw you're building eval infra at {{company}}" or "Noticed your PR on promptfoo's custom scorer" or "Your comment on HN about synthetic data quality stuck with me."}}.
>
> I've been working on something adjacent — an open-source framework that evaluates the *personas themselves*, not just the LLM outputs. 52 scorers across 4 tiers: schema validity, factual coherence, depth, and bias detection.
>
> The bias tier is the interesting part — the scorers are designed to fail. They specifically catch register inflation, hedging, and balanced-opinion patterns that LLMs default to.
>
> {{repo_link}}
>
> Would love your take if you get a chance to look. No pitch — just think you'd have useful feedback.
>
> — Ivan

**Follow-up 1 (Day 5) — Different angle: the "why"**

> Subject: re: persona eval
>
> Hey {{first_name}}, quick follow-up —
>
> The reason I built this: every eval tool I found checks prompt→output quality. Nothing checks whether the *persona* driving the simulation is biased or shallow. Seemed like a gap.
>
> 303 tests, all mocked by default so you can run the suite without an API key. Drop-in if you want to try it.
>
> {{repo_link}}

**Follow-up 2 (Day 12) — Proof/social**

> Subject: re: persona eval
>
> Last note — just shipped bias dimensions D45/D46/D47 for register inflation, hedging, and balanced-opinion detection. These are "designed-to-fail" scorers: they exist to catch the specific patterns LLMs always produce.
>
> The audit dashboard lets you inspect every metric. If you're evaluating synthetic personas at all, this might save you a few weeks.
>
> {{repo_link}}

---

## Template 2: Academic Researcher (ICP 3)

**Use when:** They published a paper on LLM personas, bias, synthetic respondents, or persona-based evaluation. Signal: arXiv paper, OpenReview submission, or GitHub persona-bias dataset.

**Subject:** persona evaluation framework

**Email:**

> Hi {{first_name}},
>
> {{personalized_signal — e.g., "Read your paper on {{paper_title}} — especially the finding about {{specific_finding}}" or "Your work on persona-bias at Allen AI is exactly the kind of research that motivated this."}}.
>
> I've been building an open-source evaluation framework specifically for synthetic persona quality. 52 scoring dimensions across 4 tiers — schema, factual coherence, narrative depth, and bias. The bias tier (D45–D47) detects register inflation, hedging, and balanced-opinion patterns that LLMs default to.
>
> Each scorer produces a structured EvalResult (pass/fail + 0–1 score + metric details dict), so results are reproducible and machine-readable.
>
> {{repo_link}}
>
> Thought it might be relevant to your work. Would be curious if the dimension taxonomy maps to what you've observed.
>
> — Ivan

**Follow-up 1 (Day 5)**

> Subject: re: persona evaluation framework
>
> Hey {{first_name}} — one thing I didn't mention: the framework supports set-level scorers that operate on persona *batches*, not just individual personas. So you can check distributional properties (demographic spread, diversity, cross-persona consistency) across a whole generated set.
>
> Also extensible — new dimensions are a BaseScorer subclass in ~50 lines.
>
> {{repo_link}}

---

## Template 3: UX/Product Researcher (ICP 2) — *Hold for DX improvements*

**Note:** This ICP converts better with a Colab notebook or web demo than a cold email to a CLI tool. Deprioritize outreach here until there's a lower-friction entry point. If you do reach out:

**Subject:** checking your AI personas

**Email:**

> Hi {{first_name}},
>
> {{personalized_signal — e.g., "Saw your post about using AI to generate user personas for {{project/company}}" or "Your thread on synthetic users being 'too polished' resonated."}}.
>
> I built an open-source tool that scores AI-generated personas on 52 dimensions — including whether they're biased, shallow, or just parroting LLM defaults. It flags things like: personas that are all suspiciously articulate, over-balanced on opinions, or missing real human contradictions.
>
> Right now it's a Python CLI (working on making it more accessible). But if you're curious what it looks like: {{repo_link}}
>
> — Ivan

---

## Lead Sourcing Plan

### Where to find ICP 1 (AI Eval Engineers)

| Source | How to extract leads | Volume | Signal quality |
|--------|---------------------|--------|----------------|
| **GitHub stargazers** | `gh api repos/promptfoo/promptfoo/stargazers` — filter for profiles with email + bio mentioning "eval" or "ML" | High | Medium |
| **GitHub contributors** | Contributors to promptfoo, deepeval, langfuse, braintrust-data repos | Low | Very high |
| **GitHub search** | Repos with "persona" + "eval" or "synthetic persona" in description → owners | Low | Very high |
| **HN commenters** | Threads on LLM eval, synthetic data, persona bias → profiles with email | Medium | High |
| **Twitter/X** | Search "LLM eval" OR "synthetic personas" OR "persona bias" → active accounts | Medium | Medium |
| **LinkedIn** | Title: "AI Eval Engineer" OR "ML Engineer" at companies using LLMs | High | Medium |

**Quick start script (GitHub stargazers of eval tools):**
```bash
# Get stargazers of promptfoo with public email
gh api repos/promptfoo/promptfoo/stargazers --paginate --jq '.[].login' | \
  head -100 | \
  xargs -I{} gh api users/{} --jq '[.login, .email, .bio, .company] | @tsv' \
  > leads_promptfoo.tsv
```

### Where to find ICP 3 (Academic Researchers)

| Source | How to extract leads | Volume | Signal quality |
|--------|---------------------|--------|----------------|
| **arXiv authors** | Papers matching "synthetic persona" OR "persona bias" OR "LLM persona evaluation" → author pages | Medium | Very high |
| **OpenReview** | Search AAAI AIES, ACL, EMNLP for persona/bias papers → author emails | Medium | Very high |
| **GitHub** | Contributors to allenai/persona-bias → profiles | Low | Very high |
| **Semantic Scholar** | API: search papers, extract author emails | Medium | High |
| **Twitter/X academic** | Authors who tweet their papers on persona/bias topics | Low | High |

**Quick start (arXiv):**
```bash
# Search arXiv for recent persona eval papers
curl "http://export.arxiv.org/api/query?search_query=all:synthetic+persona+evaluation&max_results=25&sortBy=submittedDate" \
  > arxiv_results.xml
# Parse author names → Google Scholar → institutional email
```

### Where to find ICP 2 (UX/Product Researchers) — *deprioritized*

| Source | Notes |
|--------|-------|
| Product Hunt | Launch there once DX is smoother |
| LinkedIn UX groups | "AI-generated personas" discussions |
| UX Substack newsletters | Comment/guest post, don't cold email |

---

## Outreach Cadence

| Week | Action | Volume |
|------|--------|--------|
| 1 | Build lead list: 50 ICP 1 from GitHub, 20 ICP 3 from arXiv | — |
| 1 | Send first batch: 25 ICP 1 emails | 25 |
| 2 | Follow-up 1 (Day 5) + send second batch: 25 ICP 1 + 20 ICP 3 | 70 |
| 3 | Follow-up 2 (Day 12) + review replies | — |
| 3 | Post "Show HN" + share in ML Discord servers | 1 post |
| 4 | Analyze: which template/signal got replies? Iterate. | — |

**Review gate:** You approve every batch before sending. No auto-send.

---

## Tracking

Use `leads.csv` with columns:

```
id,name,email,company,role,icp,signal_url,signal_quote,template,status,sent_date,reply_date,outcome
```

Status flow: `new` → `drafted` → `approved` → `sent` → `replied` / `no_reply` → `converted` / `dead`
