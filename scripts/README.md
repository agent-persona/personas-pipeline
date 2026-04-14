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

## `convert_to_eval_personas.py`

Converts `PersonaV1` JSON (the synthesis pipeline's output shape) into
`persona_eval.Persona` JSON (the shape each eval scorer expects). Pure
field-mapping — no LLM calls, no network.

### Run it

```bash
# Convert everything under output/ (default), write to output/eval_personas/
python scripts/convert_to_eval_personas.py

# Custom paths
python scripts/convert_to_eval_personas.py \
  --input-dir tests/fixtures \
  --output-dir /tmp/eval_out
```

Input glob: `<input-dir>/persona_*.json`. Output file naming: one file per
input, keyed by `cluster_id` (falling back to the source filename stem) as
`<output-dir>/<cluster_id>.json`. Overwrites existing files in `--output-dir`
without warning.

Use this after `run_full_pipeline.py` to hand off personas to `persona_eval`
for scoring, or run it against `tests/fixtures/` to regenerate the example
eval inputs without a live pipeline run.

## Using this script as an iteration harness

The defaults in this repo were validated by running control vs variant
pipelines through this script. The pattern that produced the write-ups
under `output/experiments/`:

1. Keep the stage list identical to control.
2. Pass the variant as a keyword argument (a different prompt builder, a
   different persona schema version, a different clustering threshold).
3. Write results to `output/experiments/<experiment_id>/` so control and
   variant runs sit side-by-side.

Branching or parallel stage execution is deliberately **not** in
`orchestration/` — that module is out of scope for the iteration program.
Either call the stages directly from a driver script, or build a second
`Pipeline` and run both.
