# personas-pipeline

End-to-end persona-framework pipeline: turn raw behavioral records (chat logs,
analytics events, CRM notes, Discord activity) into grounded, interactive
persona twins you can chat with — with every claim bound to a source record.

Every default in this codebase — schema shape, prompt structure, retry policy,
judge rubric, clustering knobs, temperature, model tier — is the product of
**hypothesis-driven experiments** run against a frozen golden set. The repo
doubles as the artifact and the harness: ship with it as-is, or pull the same
levers the experiments pulled and iterate further.

---

## What makes this different

Most "AI persona" tools today are prompt wrappers around ChatGPT that output
plausible-sounding marketing personas from a one-line brief. The commercial
category splits into two camps:

- **Synthetic-user SaaS** — Delve AI, Synthetic Users, Viewpoints.ai, Ditto:
  hosted, per-seat or per-persona pricing (Delve charges $0.99/synthetic user),
  closed-source, you can't audit how a claim was derived.
- **Character platforms** — Character.AI, Poe, MyShell, Janitor: strong for
  entertainment-style roleplay, but no grounding in your data and no
  evaluation harness for drift, refusal, or hallucination.

personas-pipeline targets the gap between them. The concrete differentiators:

| Axis | Typical SaaS / prompt wrapper | personas-pipeline |
|---|---|---|
| **Grounding** | Free-form LLM output, no citations | Every claim carries `source_evidence` record IDs; structural groundedness is measured on every synthesis |
| **Methodology** | Vendor-decided defaults, no visibility | ~30 logged experiments under `output/experiments/` with hypothesis → control → metric → decision |
| **Openness** | Closed SaaS, black-box prompts | MIT-licensed, modular (6 packages, stable JSON contracts between stages), self-host with your own API key |
| **Cost** | $0.99+/persona hosted | ~$0.027/persona at Haiku + `temperature=0.0` (exp-2.06); re-runs are free |
| **Runtime** | Static persona PDFs, or unbounded chat | Twin runtime with tested boundary / refusal / meta-question behavior (exp-4.05, 4.16, 4.20) |
| **Eval** | "Looks reasonable" | Shared metrics (schema validity, groundedness, distinctiveness, judge score, drift) and a frozen golden set to re-run them against |

The `evaluation/` package is the load-bearing piece. We've already used it to
reject ideas that sound good on paper — e.g., reference-based judging tightened
variance 44% but introduced a -0.33 anchoring bias and was rejected
(exp-5.11); pairwise-vs-absolute judging failed inter-judge agreement at 43%
position bias and was deferred (exp-5.10). Those negative results are in the
repo so you don't pay to rediscover them.

---

## Who it's for

**Discord community owners.** Point the crawler at your server's message
history and get back 3–5 grounded archetypes of the people actually in your
channel — "the lurker who DMs mods", "the question-asker", "the power-user
running tournaments" — with quote-level evidence for each trait. Spin up twin
bots that answer in-character for moderation playbooks, onboarding flows, or
pre-testing announcements before you post them.

**LinkedIn / creator-economy influencers.** Feed in comment threads, DM
transcripts, or subscriber survey exports. Synthesis produces distinct
audience personas with `goals`, `pains`, and `sample_quotes` drawn directly
from your audience's own language. Draft posts against a twin, see which
archetype each section lands with, iterate before you publish.

**Chatbot developers.** The schema, synthesis pipeline, and twin runtime are
designed to be lifted into a product. Stable JSON contracts between every
stage mean you can replace the crawler with your own data source, swap model
backends, or plug the twin into an existing chat UI without touching the rest.
The experimental record tells you which knobs matter and which don't —
`goals` and `sample_quotes` are load-bearing (exp-1.07); `channels`,
`vocabulary`, `journey_stages` can be dropped in cost-constrained builds.

---

## Selected results

All numbers below come from `output/experiments/` on the golden tenant
(`tenant_acme_corp`, 37 records, 2 clusters). Sample sizes are small — each
FINDINGS.md reports its own caveats — but the decisions encoded in the shipped
defaults are the ones the data supported.

| Experiment | Result | Impact on defaults |
|---|---|---|
| **2.06** temperature sweep | `temp=0.0` reached groundedness 1.0, cut retry rate 50%, saved 30% cost ($0.027 vs $0.039/persona) | Production default is `temperature=0.0` |
| **1.07** field interdependence | Removing `goals` drops 3 quality dims by -1 each; removing `sample_quotes` drops voice fidelity by -2 | QA prioritizes these two fields; others are optional |
| **5.11** reference-based judging | Variance -44% but mean -0.33 (anchoring bias), rank distortion | **Rejected** — did not ship |
| **5.10** pairwise vs absolute judging | Inter-judge Spearman ≈ 0.0 (absolute) vs 0.06 (pairwise win-count); 43% position bias | **Deferred** — honest negative, don't rely on cross-judge absolute scores |
| **6.02** coverage gap analysis | Current persona set represents 5.3% of population records (strong negative) | Drives next batch: outlier personas (exp-6.11), hierarchical archetypes (exp-6.23), clusterer sweep (exp-6.03) |
| **1.11** negative-space probing | Surfaces claims a persona would *deny* — used to validate distinctiveness | Wired into judge rubric |

The cost numbers above are real run costs at Haiku pricing. A full end-to-end
pipeline run (crawl → segment → 2 personas synthesized → twin smoke chat) lands
at cents, not dollars.

---

## Module layout

```
personas-pipeline/
├── crawler/          # Stage 1: pull behavioral records from sources
├── segmentation/     # Stage 2: cluster records into behavioral segments
├── synthesis/        # Stage 3: turn clusters into structured personas
├── twin/             # Stage 4: chat in character as a persona
├── orchestration/    # Glue: sequential DAG runner that wires the stages
├── evaluation/       # Judges, golden set, shared metrics
├── scripts/          # run_full_pipeline.py — end-to-end demo
└── output/           # persona_*.json + experiments/ — the empirical record
```

Each module is an independent Python package with its own `pyproject.toml`
and `README.md`. `pip install -e .` one at a time to work on it in isolation,
or run the whole pipeline end-to-end with the top-level script.

---

## Problem space → module map

| # | Problem space | Primary module | Primary files |
|---|---|---|---|
| 1 | Persona representation & schema | `synthesis/` | `synthesis/models/persona.py`, `synthesis/models/evidence.py` |
| 2 | Synthesis pipeline architecture | `synthesis/` | `synthesis/engine/synthesizer.py`, `synthesis/engine/prompt_builder.py`, `synthesis/engine/model_backend.py` |
| 3 | Groundedness & evidence binding | `synthesis/` | `synthesis/engine/groundedness.py`, `synthesis/engine/prompt_builder.py` |
| 4 | Twin runtime: consistency & drift | `twin/` | `twin/twin/chat.py` |
| 5 | Evaluation & judge methodology | `evaluation/` | `evaluation/judges.py`, `evaluation/metrics.py`, `evaluation/golden_set.py` |
| 6 | Population distinctiveness & coverage | `segmentation/` + `synthesis/` | `segmentation/engine/clusterer.py`, `synthesis/engine/synthesizer.py` |

---

## Quickstart

```bash
cd personas-pipeline
pip install -e crawler -e segmentation -e synthesis -e twin -e orchestration -e evaluation
pip install python-dotenv
cp synthesis/.env.example synthesis/.env   # set ANTHROPIC_API_KEY=...
python scripts/run_full_pipeline.py
```

Output personas land in `output/persona_*.json`. Haiku is the default model
based on exp-2.06; swap `default_model=claude-sonnet-4-6` in `synthesis/.env`
to re-evaluate the tradeoff on your own inputs.

---

## Pipeline flow

```
crawler.fetch_all(tenant_id)
        │ list[crawler.Record]
        ▼
segmentation.segment(records, ...)
        │ list[dict]  — conforms to synthesis.models.ClusterData
        ▼
synthesis.engine.synthesizer.synthesize(cluster, backend)
        │ SynthesisResult { persona: PersonaV1, groundedness, cost }
        ▼
twin.TwinChat(persona, client).reply(user_message)
        │ TwinReply { text, cost }
        ▼
output/persona_XX.json
```

Every arrow is a stable JSON contract — swap any single stage as long as the
contract holds.

---

## The evaluation harness

Shipped defaults were validated against a shared harness:

1. **A hypothesis**, written before the run.
2. **A control** — same golden input, default config.
3. **A metric** — one of the shared metrics in `evaluation/metrics.py`
   (schema validity, groundedness, distinctiveness, judge score, drift, cost).
4. **A result + decision** — adopt / reject / defer, recorded under
   `output/experiments/<exp-id>/`.

`evaluation/golden_set.py` + the mock tenant in `crawler/crawler/connectors/`
give you the exact control input the existing findings were measured against.

---

## Where to find things

- `output/experiments/` — findings, raw outputs, per-experiment write-ups.
- `docs/plans/` — batch research strategy and results summaries.
- `PRD_SYNTHESIS.md`, `PRD_SEGMENTATION.md`, `PRD_TESTING.md` — product context
  and hypotheses that shaped each module.
