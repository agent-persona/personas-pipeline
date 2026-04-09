# orchestration

**Glue.** A minimal sequential DAG runner that wires the four pipeline stages
together (`crawler → segment → synthesize → twin → persist`). Stands in for
what a production system would run on Temporal / LangGraph / Prefect.

This module is **not** a primary lab target. Per `PRD_LAB_RESEARCH.md`, the
choice of orchestration engine (Temporal vs LangGraph vs …) is explicitly
**out of scope** for the lab program:

> *"Infrastructure benchmarking (Temporal vs LangGraph etc — that's an
> orchestration call, not a research one)."*

Treat this module as **stable shared infrastructure**. Don't edit it to run
an experiment — add your experiment as a new `Stage` in the pipeline, or
wrap an existing stage, rather than changing the runner.

## What lives here

```
orchestration/
├── orchestration/
│   ├── dag.py         # Pipeline, Stage, StageResult, RunState, PipelineError
│   └── __init__.py
└── pyproject.toml
```

## I/O contract

```python
from orchestration import Pipeline, Stage

pipeline = Pipeline([
    Stage(name="ingest",     fn=stage_ingest,     description="Pull from connectors"),
    Stage(name="segment",    fn=stage_segment,    description="Cluster by behavior"),
    Stage(name="synthesize", fn=stage_synthesize, description="Generate personas"),
    Stage(name="twin_chat",  fn=stage_twin_chat,  description="Demo twin replies"),
    Stage(name="persist",    fn=stage_persist,    description="Save outputs"),
])

state = await pipeline.run(initial_input=None, tenant_id="tenant_acme_corp")
```

Each stage function takes the previous stage's output and returns a new value.
Both sync and async functions are accepted — the runner detects `asyncio`
coroutines via `inspect.iscoroutinefunction`.

`RunState` captures per-stage success, duration, and an output summary. A
failed stage raises `PipelineError` carrying the partial state.

See `scripts/run_full_pipeline.py` for the reference wiring.

## Where this module shows up in the experiment catalog

Nowhere as a variable under test — but two places as a hosting layer:

- **Experiment harness.** Any experiment that compares a variant stage
  against a control stage can build two `Pipeline` instances that share
  every stage except one. Good way to A/B without duplicating the whole
  script.
- **Telemetry collection.** `RunState.stages` gives per-stage timing and
  success. If the evaluation module needs to log cost and latency per run,
  read them out of `RunState` after `pipeline.run()` returns.

## Knobs you *shouldn't* turn

- Parallel execution, conditional branching, resume-from-stage, Postgres
  persistence — all explicitly out of scope. If you need them, talk to
  the owner of `PRD_ORCHESTRATION.md`; the lab program is not where
  orchestration features ship.
