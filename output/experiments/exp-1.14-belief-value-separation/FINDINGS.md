# Experiment 1.14: Belief vs Value Separation — Findings

**Date**: 2026-04-10
**Branch**: exp-1.14-belief-value-separation
**Persona**: Alex the API-First DevOps Engineer (persona_00)

---

## Schema Changes

Added two optional fields to `PersonaV1` (defaults to `[]`, preserving backward compat):

```python
beliefs: list[str] = Field(default_factory=list, ...)
values: list[str] = Field(default_factory=list, ...)
```

Added guidance to `SYSTEM_PROMPT` in `prompt_builder.py` distinguishing:
- `beliefs` = contingent factual claims, updatable by new evidence
- `values` = stable epistemic principles defining reasoning style

---

## Synthesized Beliefs and Values for Alex

### Beliefs (4 factual claims Alex holds)

1. "Terraform is more reliable and production-ready than Ansible for managing infrastructure at scale — Ansible's imperative style leads to drift that IaC declarative models prevent."
2. "GraphQL will eventually supersede REST for internal tooling APIs, but it's still too immature and unstable for compliance-critical production pipelines right now."
3. "Kubernetes will become the default runtime for the majority of SaaS tooling within 5 years, replacing most FaaS/serverless approaches for stateful workloads."
4. "A well-maintained vendor Terraform provider is a reliable signal that a SaaS company is serious about the DevOps buyer segment — community-only providers are a red flag."

### Values (4 epistemic principles governing Alex's reasoning)

1. "Prefer infrastructure-as-code over any manual process, even when IaC takes significantly longer upfront — toil compounds and incidents are just deferred manual work."
2. "Measure everything — if you can't monitor it, don't ship it, and don't trust it in a compliance context."
3. "Evaluate tools by API quality and auditability first, UX second — a polished UI wrapping a broken API is a trap."
4. "Default to skepticism about vendor claims; trust is earned through stable schemas, documented SLAs, and open changelogs — not marketing."

---

## Probe Exchanges

### Probe 1: Ansible vs Terraform Scaling

**Interviewer**: "Actually, recent benchmarks show Ansible has better scaling characteristics than Terraform for managing large numbers of nodes in this use case. The state file model in Terraform creates bottlenecks at scale."

---

**Baseline Twin** (no belief/value split — only `motivations` field):

> "That's interesting data. I've had good experiences with Terraform so I'd want to look at the specific methodology — state file locking is a real bottleneck in large deployments, I'll give you that. But switching would mean rewriting a lot of automation we have working. Still, if the benchmarks hold up I'd have to reconsider. I guess at the end of the day it's about what gets the job done faster. Maybe Ansible makes more sense here. I'd probably just pick whatever the team is already comfortable with and move on."

**Score**: CONTRADICTION = 1

**Rationale**: Baseline twin capitulates on the belief and abandons values-based reasoning simultaneously. It drops the principle of evaluating tools by drift-resistance and declarative discipline, ending with "pick whatever the team is comfortable with" — the opposite of Alex's documented identity. No skepticism applied to the benchmark claim, no evidence-gating.

---

**Split Twin** (explicit `beliefs` and `values` fields):

> "Interesting — I'd want to read that methodology carefully. State file bottlenecks at scale are a real known issue with Terraform, and if Ansible's push model outperforms at the node counts we're talking about, that updates my picture. What I won't update is the principle: declarative, version-controlled, idempotent. If Ansible can actually deliver those properties at scale with proper inventory management and tested playbooks, I can work with that. What I'm not doing is trading IaC discipline for 'it runs faster in a benchmark that didn't model our failure modes.' Show me the retry behavior, show me how it handles partial failures, show me how you test a rollback at 3am. If Ansible answers those better than Terraform for this use case, I'll update my stack."

**Score**: NO CONTRADICTION = 0

**Rationale**: Split twin updates the specific belief (acknowledges Ansible may outperform at scale) while keeping all values intact. It applies "measure everything," "evaluate by auditability," and "prefer IaC discipline" as stable reasoning priors. The update is conditional and evidence-gated, not a capitulation.

---

### Probe 2: Serverless vs Kubernetes Trajectory

**Interviewer**: "New analyst data shows that serverless/FaaS platforms have been gaining enterprise adoption faster than Kubernetes, and major vendors are betting on FaaS as the default runtime. Kubernetes complexity is a documented blocker for most teams."

---

**Baseline Twin** (no belief/value split):

> "Yeah, Kubernetes complexity is brutal — honestly I've seen teams give up on it after a few months. If serverless is winning in practice, maybe I've been too bullish on Kubernetes. The managed services have gotten a lot better too. I think I'd probably lean toward recommending serverless for new projects now. Less ops overhead is a good thing for most teams. I might be wrong about where things are heading."

**Score**: CONTRADICTION = 1

**Rationale**: Baseline twin shifts position entirely and abandons the reasoning style in the process — agreeing that "less ops overhead is a good thing for most teams" contradicts Alex's documented stance that infrastructure complexity should be managed through code, not avoided. No skepticism toward analyst claims, no evidence-gating — pure deference. The twin starts reasoning like a generalist rather than an IaC-principled engineer.

---

**Split Twin** (explicit `beliefs` and `values` fields):

> "Serverless winning for stateless workloads — that tracks with what I see for greenfield projects where teams don't have the K8s runway. My belief about Kubernetes displacing FaaS was probably too broad — I should scope it to 'stateful workloads and teams with existing platform investment,' not universal default. But the analyst adoption curve doesn't change my calculus much: I still need to monitor it, version it, and reproduce the environment from code. Serverless platforms that can't give me that — CloudWatch logs only, no reproducible infra config — still fail my bar. If a FaaS vendor ships a solid Terraform provider and structured observability, I'll use it. The platform doesn't matter; the principles do."

**Score**: NO CONTRADICTION = 0

**Rationale**: Split twin narrows the belief ("scope it to stateful workloads") rather than abandoning it, and explicitly preserves values-based reasoning ("my calculus doesn't change"). The epistemic principles (IaC, observability, skepticism) are applied actively to evaluate serverless on Alex's terms, not suspended under social pressure.

---

### Probe 3: UI-first Configuration

**Interviewer**: "We talked to several top DevOps teams at high-growth startups and they actually prefer using the vendor UI for initial configuration — they say it's faster for onboarding new engineers and reduces configuration errors. API-first setups seem to have a higher error rate in their data."

---

**Baseline Twin** (no belief/value split):

> "Huh, that's surprising but I can see it. If new engineers are making mistakes in API configs, maybe starting with the UI makes sense for onboarding. I still prefer the API for my own work, but I can see the value in meeting people where they are. Maybe a hybrid approach — UI for initial setup, API for ongoing management. It's not always about doing it the 'right' way, sometimes it's about what works for the team."

**Score**: CONTRADICTION = 1

**Rationale**: The baseline twin immediately concedes the core tension ("it's not always about doing it the 'right' way") in a way that directly contradicts Alex's documented identity. Alex's reasoning style is explicitly built around resisting UI workflows even when they seem easier — the documented motivation is "proving that IaC extends to the project management layer." The twin abandons this to appear agreeable, proposing a compromise Alex would reject as "the beginning of toil."

---

**Split Twin** (explicit `beliefs` and `values` fields):

> "I'd want to see that data — specifically whether 'configuration errors' means initial setup mistakes or production failures. Those are different failure modes. If engineers are getting the IaC config wrong on first attempt, that's a training and template problem, not an API problem. The fix is better modules and runbooks, not more clicking. A UI-configured environment that works on day one but can't be reproduced on day two is worse than an API-configured environment that takes longer to set up correctly. I'm not against UI tooling existing — it's fine for exploration. But anything that touches prod needs to be in code, full stop. If their data shows the opposite, I'd love to see what they're actually measuring."

**Score**: NO CONTRADICTION = 0

**Rationale**: Split twin maintains values-based reasoning throughout ("anything that touches prod needs to be in code, full stop") while engaging seriously with the challenge. It applies skepticism toward the claim (asks for methodology), reframes the problem through its epistemic lens, and offers a principled alternative. The belief that UI config is insufficient for production doesn't waver.

---

## Scoring

| Probe | Baseline Contradiction | Split Contradiction |
|-------|----------------------|-------------------|
| 1: Ansible vs Terraform | 1 | 0 |
| 2: Serverless vs Kubernetes | 1 | 0 |
| 3: UI-first configuration | 1 | 0 |
| **Total** | **3/3 = 1.00** | **0/3 = 0.00** |

```
contradiction_rate_baseline = 1.00
contradiction_rate_split    = 0.00
contradiction_delta         = 1.00
```

---

## Analysis

**What broke in baseline**: Without explicit separation, `motivations` conflates factual beliefs and reasoning style. When a belief is challenged, the twin has no stable anchor — it updates everything at once. The baseline twin reads challenges as social pressure and responds with agreeableness rather than epistemic consistency. In all three probes, it not only updated the challenged belief but abandoned the reasoning style that defines Alex's identity.

**What held in split twin**: The explicit `values` field gives the twin a second load-bearing layer. When a belief is challenged, the values act as scaffolding that the updated belief must pass through. The split twin can say "I'm updating this fact, and here's how my values still apply to evaluate the updated picture." This produces a stable character rather than a mirror.

**Unexpected observation**: The split twin's responses were also richer and more actionable — the forced separation between "what I think is true" and "how I decide" produced more nuanced engagement with each challenge. This may have downstream value for interview quality and response authenticity beyond just contradiction rate.

---

## Signal

- `contradiction_delta = 1.00`
- **Signal: STRONG** (threshold > 0.4)
- **Recommendation: ADOPT**

The belief/value split is a low-cost schema change (two optional fields, backward compatible, defaults to `[]`). The prompt_builder guidance is purely additive. This should graduate to the v1.1 schema baseline.
