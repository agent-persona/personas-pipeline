# scripts

End-to-end runner(s) that wire every module together.

## `run_full_pipeline.py`

Demos the whole flow: **crawler → segment → synthesize → twin chat → persist**.
Uses the mock tenant (`tenant_acme_corp`, B2B SaaS / engineering PM tool)
and Haiku by default, so a full run costs cents.

### Prerequisites

```bash
# From personas-pipeline/
pip install -e crawler -e segmentation -e synthesis -e twin -e orchestration
pip install python-dotenv
cp synthesis/.env.example synthesis/.env
# edit synthesis/.env and set ANTHROPIC_API_KEY=...
```

### Run it

```bash
python scripts/run_full_pipeline.py
```

Output: `output/persona_00.json`, `output/persona_01.json` (one per cluster
from the mock fixture — engineer cluster and designer cluster).

The script logs per-stage duration, per-persona groundedness score and
cost, and a twin-reply demo for each persona.

### What the script does, stage by stage

1. **`stage_ingest`** — calls `crawler.fetch_all("tenant_acme_corp")`.
2. **`stage_segment`** — round-trips to `segmentation.models.RawRecord` and
   calls `segment(...)` with `similarity_threshold=0.15`, `min_cluster_size=2`.
3. **`stage_synthesize`** — for each cluster, calls
   `synthesis.engine.synthesize(cluster, AnthropicBackend(...))`. Logs cost
   and groundedness per persona.
4. **`stage_twin_chat`** — for each persona, asks one hard-coded question
   (`"What's the single biggest frustration with your current tools?"`)
   and prints the reply. Dirt-cheap liveness check.
5. **`stage_persist`** — writes the persona JSON files to `output/`.

All five stages run through the `orchestration.Pipeline` DAG runner, which
logs per-stage duration and captures a `RunState` for telemetry.

## Using this script as an experiment harness

The simplest way to compare two pipeline variants is to duplicate
`run_full_pipeline.py`, change one stage, and run both. But prefer to build
your experiment *inside* the pipeline by:

1. Keeping the stage list identical to control.
2. Passing your variant as a keyword argument (e.g. a different prompt
   builder, a different persona schema version, a different clustering
   threshold).
3. Writing results to `output/experiments/<experiment_id>/` so control
   and variant runs sit side-by-side.

If you need branching or parallel stage execution, **don't add it to
`orchestration/`** — that module is explicitly out of scope for the lab
program. Either call the stages directly from your script, or build a
second `Pipeline` and run both.

## `run_baseline.py`

Generates a frozen baseline JSON under `evaluation/baselines/`.

```bash
python scripts/run_baseline.py --num-runs 3
python scripts/run_baseline.py --schema-version v2 --birth-year 1988 --eval-year 2026
```

Default outputs:

- `v1` -> `evaluation/baselines/p1_baseline.json`
- `v2` -> `evaluation/baselines/p2_baseline.json`

The baseline file now stores:

- aggregate metrics for the first-run personas
- per-persona `developmental_fit`, `historical_fit`, `capability_coherence`,
  and `relational_realism` scores alongside groundedness and cost
- `stability_breakdown` with per-field similarity scores
- `runs`, containing each synthesis pass so drift can be inspected later
- `run_metadata` with schema version plus `birth_year` / `eval_year` for v2
- the rerun-aware summaries used by compare scripts and temp sweeps

This matters because `stability` alone is only one scalar; debugging drift
needs both the field-level breakdown and the raw rerun outputs.

## `run_temp_sweep.py`

Runs experiment `2.06` against the selected schema pipeline.

```bash
python scripts/run_temp_sweep.py --num-runs 3
python scripts/run_temp_sweep.py --schema-version v2 --birth-year 1988 --eval-year 2026
```

Default sweep temperatures:

- `0.0`
- `0.3`
- `0.5`
- `0.7`
- `1.0`

The script:

1. ingests + segments once
2. re-synthesizes the shared clusters at each temperature
3. scores groundedness, developmental fit, historical fit, capability
   coherence, relational realism, stability, and mean judge score
4. selects the best temperature by highest stability without groundedness regression
5. saves the full report to `evaluation/baselines/temp_sweep.json`

## `freeze_temp_sweep_winner.py`

Exports the selected `2.06` winner from `temp_sweep.json` into a
baseline-like artifact so it can be compared with `compare_versions.py`.

```bash
python scripts/freeze_temp_sweep_winner.py
```

Default output: `evaluation/baselines/p1_temp_winner.json`

## `run_field_interdependence.py`

Runs experiment `1.07` on a frozen artifact by swapping dependent field
bundles between personas and measuring how much the deterministic
field-interdependence score drops.

```bash
python scripts/run_field_interdependence.py --input evaluation/baselines/p1_baseline.json
```

Optional: add `--run-judge` to compare original vs mutated personas pairwise.

## `run_rubric_ablation.py`

Runs experiment `5.05` by rescoring the same personas with the full judge
rubric and with one dimension removed at a time.

```bash
python scripts/run_rubric_ablation.py --input evaluation/baselines/p1_baseline.json
```

Default output: `evaluation/baselines/rubric_ablation.json`

## `run_reference_judging.py`

Runs experiment `5.11` by scoring the same personas twice: once free-form
and once with reconstructed cluster/source context attached to the judge
prompt.

```bash
python scripts/run_reference_judging.py --input evaluation/baselines/p1_baseline.json
```

Default output: `evaluation/baselines/reference_vs_free_judging.json`

Default outputs:

- `v1` -> `evaluation/baselines/temp_sweep.json`
- `v2` -> `evaluation/baselines/temp_sweep_v2.json`
