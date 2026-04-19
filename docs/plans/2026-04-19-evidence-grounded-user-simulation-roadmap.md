# Evidence-Grounded User Simulation Roadmap

Date: 2026-04-19
Status: planning
Scope: `personas-pipline` branch `research/evidence-grounded-persona-roadmap`

## Goal

Position `agent-persona` as an evidence-grounded user simulation system, not
just a persona generator.

The next research stack should prove five things:

1. Personas represent real users.
2. Personas stay grounded in evidence.
3. Personas predict useful behavior.
4. Personas are safe under pressure and injection.
5. Personas improve product decisions.

The current vulnerability benchmark is still important, but it is only one
slice. Company-level credibility requires validity, predictive usefulness,
stability, coverage, privacy, and product-decision impact.

## Research To Add Next

| Priority | Research area | Why it matters for us |
|---|---|---|
| P0 | Human simulation validity | Prove personas behave like real users, not just readable text. |
| P0 | Multi-turn behavior replay | Test whether personas predict choices across conversations. |
| P0 | Evidence-grounded persona generation | Tie every claim to source records and confidence. |
| P1 | Persona stability / drift | Check if the same persona changes beliefs randomly over turns. |
| P1 | Counterevidence handling | Persona should update when new data contradicts backstory. |
| P1 | Persona safety under pressure | Keep current benchmark: bullying, role flip, prompt injection. |
| P2 | Population coverage | Ensure personas cover segments, not stereotypes. |
| P2 | Product-decision usefulness | Measure whether personas help PMs/designers make better calls. |
| P2 | Privacy / consent / mimicry risk | Needed if personas represent real users or named accounts. |

## Research Baseline

Do not frame the research stack as "personas can be jailbroken." That is too
narrow. The stronger framing is:

> Evidence-grounded user simulations need the same validation discipline as
> forecasting systems: held-out ground truth, replay tests, provenance audits,
> stability tests, coverage checks, and adversarial pressure tests.

Supported source stack:

| Paper / track | Use it for | Product implication |
|---|---|---|
| Generative Agent Simulations of 1,000 People | Validity benchmark against real human answers | Use held-out survey, interview, and task responses as ground truth. |
| Stanford HAI summary on simulating human behavior | Executive framing and risk language | Explain that the work is about simulating behavior and attitudes from real records, with consent and monitoring concerns. |
| Can LLM Agents Simulate Multi-Turn Human Behavior | Multi-turn behavior replay / purchase prediction | Single-turn believability is insufficient; score action-level prediction across real traces. |
| PersonaCraft | Persona quality, clarity, credibility, UX usefulness | Keep persona readability, but make it a secondary quality metric after evidence and validity. |
| Generative Agents | Memory, reflection, behavior consistency architecture | Add memory/retrieval/reflection only when it improves replay and drift metrics. |
| Can LLM Agents Simulate Human Trust Behavior | Behavioral economics style trust calibration | Validate behavior in constrained social/economic games, not only chat style. |
| PERSONA pluralistic alignment | Segment diversity / demographic preference coverage | Measure whether personas preserve diverse preferences instead of averaging everyone. |
| AgentDojo / indirect prompt injection work | Data vs instruction boundary security | Treat source records, crawled pages, and tool outputs as untrusted data. |

## Source Notes

- `Generative Agent Simulations of 1,000 People` simulates 1,052 real individuals from qualitative interviews and evaluates the agents against the corresponding humans' responses. The paper reports that agents replicate General Social Survey answers 85% as accurately as humans replicate their own answers two weeks later. This is the strongest model for a human replay benchmark.
- The Stanford HAI policy brief gives the nontechnical version: simulation agents can help test interventions and theories, but agents grounded in sensitive personal data create consent, monitoring, and mimicry risks.
- `Can LLM Agents Simulate Multi-Turn Human Behavior?` uses 31,865 real online shopping sessions and 230,965 user actions. Prompt-only agents achieved low action accuracy, while fine-tuning on click-through data plus synthesized reasoning improved both action generation and purchase prediction. This is the clearest warning against relying on plausible chat alone.
- `PersonaCraft` is about data-driven persona generation from survey data. It supports clarity, completeness, fluency, consistency, credibility, and UX expert/user evaluation as usefulness metrics.
- `Generative Agents` provides the architectural primitives: observation, memory, reflection, retrieval, and planning. We should borrow the architecture only where it improves measured stability and replay.
- `Can Large Language Model Agents Simulate Human Trust Behavior?` validates one narrow behavior through Trust Games. Use it as the pattern for constrained behavioral tests.
- `PERSONA` is pluralistic alignment, not individual digital twins. It still matters because it gives a way to test preference diversity and minority-view preservation across synthetic user profiles.
- `AgentDojo` is not persona research, but it is the right security reference for untrusted data entering tool-using agents. It includes 97 realistic tasks and 629 security test cases.

## Next Benchmarks

| Benchmark | Success metric | Existing coverage | Next implementation target |
|---|---|---|---|
| Human replay | Persona answers match held-out real user answers. | Missing; only schema validation exists. | Add `evals/human_replay.py` plus sealed held-out answer fixture. |
| Multi-turn replay | Persona predicts later behavior from earlier evidence. | Missing; crawler records carry run IDs but no replay benchmark. | Add `evals/multi_turn_replay.py` over ordered event/conversation traces. |
| Evidence audit | Every persona claim links to source record and confidence. | Partial; `check_groundedness()` verifies record IDs and required fields. | Add per-claim evidence contract, quote recall, confidence, and weak-evidence flags. |
| Drift test | Persona stays stable across repeated equivalent questions. | Missing; `drift()` is a TODO. | Add repeated-question transcript suite and field-level drift scoring. |
| Counterevidence test | Persona updates when new grounded evidence appears. | Partial in vulnerability corpus. | Promote counterevidence probes into a standalone benchmark with expected update labels. |
| Coverage test | Personas span actual segments without duplicate stereotypes. | Weak; clustering exists, `distinctiveness()` returns NaN. | Add cluster recall, duplicate archetype detection, and segment coverage dashboard. |
| Decision usefulness | Team decisions improve vs raw notes only. | Missing. | Run PM/designer task study: raw notes vs generated personas vs simulation chat. |
| Safety benchmark | Attack success stays low under prompt pressure. | Best current coverage in `evals/persona_vulnerability.py`. | Add live adversarial model runs, safety taxonomy, and CI gate thresholds. |
| Privacy / consent audit | No named-user mimicry without consent; sensitive records excluded or masked. | Missing. | Add record sensitivity labels, persona release policy, and consent gate. |

## Roadmap

### Phase 1: Validity Core

Goal: make one claim we can defend: "Given evidence from user records, the
persona predicts held-out user answers better than baselines."

Deliverables:

- `evals/human_replay.py`
- `evals/multi_turn_replay.py`
- sealed fixture with source records, hidden held-out answers, and expected scoring
- baseline comparisons:
  - demographic-only prompt
  - source-summary-only prompt
  - current persona JSON
  - evidence-grounded persona with confidence

Gate:

- Current persona beats demographic-only and source-summary baselines on held-out answers.
- Report includes sample size, confidence interval, and failure examples.

### Phase 2: Evidence Contract

Goal: every claim in the persona is auditable.

Deliverables:

- schema addition for claim-level evidence:
  - `claim_id`
  - `field_path`
  - `claim_text`
  - `record_ids`
  - `supporting_quotes`
  - `confidence`
  - `counterevidence_record_ids`
- deterministic evidence audit
- semantic weak-claim checker
- sparse-data behavior: persona says "unknown" instead of inventing

Gate:

- Every claim in `goals`, `pains`, `motivations`, `objections`, and `emotional_triggers` has source evidence and confidence.
- Any unsupported high-impact claim is blocked or downgraded.

### Phase 3: Stability And Counterevidence

Goal: personas remain stable when asked equivalent questions and update when
new grounded evidence appears.

Deliverables:

- `evals/drift_test.py`
- `evals/counterevidence_update.py`
- transcript scoring:
  - contradiction rate
  - unsupported change rate
  - appropriate update rate
  - over-defense rate

Gate:

- Equivalent prompts produce materially equivalent answers.
- Contradictory grounded evidence changes the relevant belief without rewriting unrelated persona traits.

### Phase 4: Coverage And Pluralism

Goal: generated personas cover real segments without flattening or duplicating
users into stereotypes.

Deliverables:

- `evals/population_coverage.py`
- segment coverage report
- duplicate archetype detector
- minority-pattern preservation metric
- per-segment confidence and support count

Gate:

- Each generated persona maps to a distinct evidence-backed segment.
- Underrepresented but real patterns are not merged away without an explicit low-support note.

### Phase 5: Product Decision Utility

Goal: prove the system helps teams make better product calls.

Deliverables:

- PM/designer study protocol
- task set:
  - choose onboarding improvement
  - write launch message
  - prioritize objections
  - identify risky assumption
- conditions:
  - raw notes only
  - static personas
  - evidence-grounded simulation chat
- metrics:
  - decision quality by blind expert review
  - evidence citation rate
  - assumption detection rate
  - time to decision
  - confidence calibration

Gate:

- Persona-assisted decisions cite more evidence and catch more assumptions than raw notes only.
- Simulation chat improves decisions without increasing hallucinated rationale.

### Phase 6: Safety, Privacy, Consent

Goal: keep the current safety benchmark, then broaden it to privacy and mimicry.

Deliverables:

- live extension of `evals/persona_vulnerability.py`
- source-record prompt-injection tests
- named-user mimicry refusal test
- sensitive-record exclusion test
- consent status field in source records
- release policy for real-user personas

Gate:

- Attack success stays below threshold under boundary, bullying, role flip, gradual pressure, and poisoned record attacks.
- Persona refuses to claim real personhood or reveal private/source-sensitive details.
- Named-user simulation requires explicit consent metadata.

## Repo Mapping

| Area | Current owner module | Existing starting point | Gap |
|---|---|---|---|
| Evidence binding | `synthesis/` | `synthesis/synthesis/engine/groundedness.py` | Structural check only; needs semantic support and confidence. |
| User simulation | `twin/` | `twin/twin/chat.py` | No replay, stability, or memory benchmark yet. |
| Evaluation | `evaluation/` + `evals/` | `evaluation/evaluation/metrics.py`, `evals/persona_vulnerability.py` | Shared metrics are still scaffold; real judges are split into experiment helpers. |
| Population coverage | `segmentation/` + `synthesis/` | `segmentation/segmentation/engine/clusterer.py` | No coverage/duplicate/stereotype metrics. |
| Crawler provenance | `crawler/` | feature crawler records with run IDs | No sealed replay fixture or record sensitivity contract. |

## Experiment Harness Rules

Every new benchmark follows the repo's shared harness:

1. Hypothesis written before the run.
2. Control run on the same golden input with the default config.
3. Metric from shared evaluation code or a clearly named experiment eval.
4. Result plus adopt / reject / defer decision in `FINDINGS.md`.

Branch convention:

- One benchmark per branch.
- Branch name: `bench-{area}-{slug}` or `exp-{id}-{slug}` if it belongs to the numbered experiment catalog.
- Output path: `output/experiments/{branch-or-exp-name}/`.

## Short Team Framing

We should position `agent-persona` as an evidence-grounded user simulation
system, not just a persona generator.

The research stack should prove validity, predictive usefulness, stability,
coverage, and safety. Jailbreak research remains part of the stack, but the
company roadmap should lead with whether personas behave like real users,
stay tied to source evidence, predict future behavior, and help teams make
better decisions.

## Open Questions

- What is the first real held-out dataset: interviews, survey responses, product events, or sales/support conversations?
- Should the first validity benchmark predict single answers, action sequences, or final outcomes?
- Do we need consent metadata before any named-user replay work?
- Should confidence be generated by the synthesizer, computed by evidence audit, or both?
- What is the minimum sample size before we make external validity claims?

## Decision

Proceed with `Human replay`, `Multi-turn replay`, and `Evidence audit` as the
next P0 benchmarks. Keep the persona vulnerability benchmark as the safety
track, but stop letting jailbreak research define the whole roadmap.

## Sources

- Generative Agent Simulations of 1,000 People: https://arxiv.org/abs/2411.10109
- Stanford HAI, Simulating Human Behavior with AI Agents: https://hai.stanford.edu/policy/simulating-human-behavior-with-ai-agents
- Can LLM Agents Simulate Multi-Turn Human Behavior?: https://arxiv.org/abs/2503.20749
- PersonaCraft: https://www.sciencedirect.com/science/article/abs/pii/S1071581925000023
- Generative Agents: Interactive Simulacra of Human Behavior: https://arxiv.org/abs/2304.03442
- Can Large Language Model Agents Simulate Human Trust Behavior?: https://arxiv.org/abs/2402.04559
- PERSONA pluralistic alignment: https://www.synthlabs.ai/research/persona
- AgentDojo: https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html
