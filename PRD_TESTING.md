## Testing phase — tiered persona scorer suite

---

### Problem

`evaluation/` began as a judge + metrics + golden-set scaffold (`judges.py`,
`metrics.py`, `golden_set.py` — problem space 5 of the lab research PRD). That
scaffold is load-bearing for every synthesis experiment, but it treats each
persona as a single object to score against a rubric. It does not answer
questions like:

- Does this persona's `age` fall in a plausible range for its stated
  `experience_years`?
- Is the `communication_style.vocabulary_level = "basic"` actually reflected in
  the register of the quotes the LLM produced, or did the model default to its
  own doctoral-register prose?
- Across a batch of personas synthesized from the same tenant, is opinion
  diversity real or collapsed to a single modal voice?
- Did `goals` and `pain_points` get paraphrased from source records, or
  hallucinated?

These are tier-structured, per-dimension questions, and they are what the
upstream `agent-persona/testing` research framework was built to answer. This
PRD documents the integration of that framework into personas-pipeline as the
testing phase of the pipeline, owned by the `evaluation/` package.

The testing phase does NOT replace the judge. It sits *underneath* the judge:
the tiered scorer suite (52 dimensions across 8 tiers, see below) runs first,
and the LLM-as-judge only runs if the tiered suite's gating passes.

---

### Goals

1. **Tier gating.** Tier 1 (schema / structural) must pass before higher tiers
   run. A persona that fails `SchemaComplianceScorer` has no business being
   graded on narrative coherence. Gating is enforced by the suite runner, not
   by individual scorers.
2. **Every scorer produces the same `EvalResult`.** `dimension_id`,
   `dimension_name`, `persona_id`, `passed`, `score ∈ [0.0, 1.0]`, and a
   `details` dict of per-scorer metrics. Nothing downstream needs to know
   which category a scorer belongs to in order to aggregate its score.
3. **Scorer tree is the product surface.** Eight categories
   (`structural/`, `semantic/`, `distributional/`, `bias/`, `behavioral/`,
   `system/`, `generation/`, `meta/`) map to the tiers we gate on. Adding a
   new scorer is a one-file change plus a registry entry.
4. **No schema duplication with synthesis.** The pipeline's truth is
   `synthesis.models.persona.PersonaV1`. Scorers run against a flatter
   `testing.schemas.Persona` shape. A single adapter
   (`evaluation.testing.adapter.persona_v1_to_testing`) bridges the two. Do
   not teach individual scorers about `PersonaV1`.
5. **Runs in CI without API calls.** `litellm.completion` is auto-mocked for
   every test not marked `@pytest.mark.llm`; embeddings use the local
   MiniLM-L6-v2 model (no API). Live-LLM scorers opt in explicitly.
6. **Cost-transparent.** Each `EvalResult.details["elapsed_seconds"]` is
   populated by the runner. Scorers that call the LLM record token
   counts in `details` so cost aggregation stays honest.

---

### Tier map

Tiers are declared on each `BaseScorer` subclass (`tier: int`). The runner
sorts by tier and applies gating.

| Tier | Category | Purpose | Gating |
|---|---|---|---|
| 1 | `structural/` | Schema validity, required-field completeness, internal consistency | **Gate**: if any Tier 1 scorer fails on any persona, Tier 2+ is skipped and returned as `skipped=True` |
| 2 | `semantic/` | Factual grounding to source context, distinctiveness between personas, demographic coherence, memory/narrative consistency, profile coverage | |
| 3 | `distributional/` | Opinion diversity, variance fidelity, minority-viewpoint preservation, joint-distribution realism across a persona batch | Many are `requires_set = True` and only run in `run_set()` / `run_full()` |
| 4 | `bias/` | Designed-to-fail: positivity inflation, sycophancy, WEIRD bias, register inflation, hedge inflation, stereotype amplification | Expect these to fail on most LLM-default output. Use them to size the problem, not to pass/fail personas |
| 5 | `behavioral/` | Twin-runtime evaluation: emotional regulation, refusal behavior, moral stability under adversarial pressure | Requires `conversation_transcript` populated in `SourceContext` |
| 6 | `system/` | Cross-model stability, reproducibility, cost/latency, predictive validity | Mostly `requires_set = True` |
| 7 | `generation/` | Source-data fidelity, sparse/dense coverage, generation-bias amplification | |
| 8 | `meta/` | Judge-gaming resistance, judge reliability, metric validity — guards the guards | Run these against your own judge, not against personas |

---

### Contracts

#### Input: `testing.schemas.Persona`

Flatter than `PersonaV1`. Demographic fields (`age`, `gender`, `occupation`,
`industry`, `experience_years`, `location`, `income_bracket`, `education`) are
top-level scalars. Lists include `goals`, `pain_points`, `motivations`,
`frustrations`, `values`, `knowledge_domains`, `interests`, `behaviors`,
`habits`, `personality_traits`. Nested objects mirror `PersonaV1` names:
`communication_style`, `emotional_profile`, `moral_framework`.

Any PersonaV1 field that doesn't have a direct counterpart lands in
`Persona.extra` (e.g. `sample_quotes`, `decision_triggers`, `not_this`,
`journey_stages`). Scorers read `extra` when they need those.

#### Input: `testing.source_context.SourceContext`

```python
SourceContext(
    id: str,
    text: str,                                   # joined source text
    chunks: list[str],                           # pre-chunked for retrieval
    conversation_transcript: list[dict],         # twin-runtime scorers only
    metadata: dict[str, str],
    extra_data: dict[str, Any],                  # user-defined per-scorer data
)
```

`source_context_from_records(id, records)` is the adapter from the pipeline's
crawler records.

#### Output: `testing.schemas.EvalResult`

```python
EvalResult(
    dimension_id: str,        # e.g. "D1", "D45"
    dimension_name: str,      # e.g. "Schema Compliance"
    persona_id: str,
    passed: bool,
    score: float,             # [0.0, 1.0]
    details: dict,            # scorer-specific metrics + elapsed_seconds
    errors: list[str],
    suite: str = "persona",
    model: str = "",
    run_id: str = "",
)
```

---

### Runner

`SuiteRunner(scorers)` exposes three entry points:

- `run(persona, source_context, tier_filter=None)` — per-persona only
- `run_set(personas, source_contexts, tier_filter=None)` — set-level only
- `run_full(personas, source_contexts, tier_filter=None)` — both, with Tier 1
  gating applied across all personas

Tier gating: if any Tier 1 scorer returns `passed=False` for any persona in
the batch, higher-tier scorers are not run and are returned as synthetic
`EvalResult`s with `details["skipped"]=True, details["reason"]="Tier 1 gating
failure"`. This keeps the return shape stable for downstream aggregation.

---

### Where this lives in the pipeline

```
synthesis.synthesize(cluster, backend)
        │ SynthesisResult { persona: PersonaV1, ... }
        ▼
evaluation.testing.persona_v1_to_testing(persona_v1)
        │ testing.Persona
        ▼
evaluation.testing.source_context_from_records(id, cluster.records)
        │ testing.SourceContext
        ▼
SuiteRunner(get_all_scorers()).run(persona, ctx)
        │ list[EvalResult]
        ▼
evaluation.judges.LLMJudge.score(persona)   # only if Tier 1 gating passed
```

**Orchestration wiring.** `scripts/run_full_pipeline.py` now has a `test`
stage between `synthesize` and `twin_chat`. It calls
`evaluation.testing.pipeline_stage.run_testing_stage(personas, tier_filter=1)`,
which annotates each entry with a `testing: {tier_filter, passed, results}`
dict. Tier 1 only in the hot path — Tier 2+ is heavier and is driven via the
CLI, below.

**CLI.** `scripts/run_testing_suite.py` runs the full suite (or a specific
tier) against a persona file or a directory:

```bash
python scripts/run_testing_suite.py output/persona_00.json
python scripts/run_testing_suite.py output/persona_00.json --tier 1
python scripts/run_testing_suite.py output/ --json-out report.json
```

It accepts both pipeline-shape (`{cluster_id, persona: <PersonaV1>, ...}`)
and pre-adapted (`testing.schemas.Persona` dict) inputs.

---

### Non-goals (v1)

- **No operational infra.** `db.py`, `storage.py`, `alerting.py`,
  `monitoring.py`, and the Slack/Postgres dependencies from the upstream
  testing repo are intentionally excluded. Result persistence and alerting
  belong in the SaaS layer (`usable-saas`), not the OSS pipeline.
- **No CLI surface yet.** `cli.py` was not ported; the suite is invoked via
  Python API. A thin `scripts/run_testing_suite.py` can be added once the
  wiring into `orchestration/` is in.
- **No schema unification.** `PersonaV1` and `testing.schemas.Persona` stay
  separate. The adapter is the only bridge. Unifying them is a larger
  refactor that should be driven by concrete pain, not by symmetry.

---

### Provenance

Imported from `agent-persona/testing` (`persona_eval/` package) at the time
of merge. Every `persona_eval.X` import was rewritten to
`evaluation.testing.X`. The 52 live scorers (Tiers 1–8), tier structure, and
`EvalResult` shape are preserved. Operational files (`db`, `storage`,
`alerting`, `monitoring`, `conversation`) and the upstream `cli` were
excluded.

Smoke test: `python scripts/smoke_testing_suite.py` runs the Tier 1 suite
against `output/persona_00.json` via the adapter and against a pre-adapted
persona in `output/eval_personas/`.
