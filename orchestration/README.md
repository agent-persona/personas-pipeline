# orchestration

**Glue.** A minimal sequential DAG runner that wires the four pipeline stages
together (`crawler → segment → synthesize → twin → persist`). Stands in for
what a production system would run on Temporal / LangGraph / Prefect.

This module was deliberately kept out of the iteration program. Per
`PRD_LAB_RESEARCH.md`, the choice of orchestration engine (Temporal vs
LangGraph vs …) is an engineering call, not a research one:

> *"Infrastructure benchmarking (Temporal vs LangGraph etc — that's an
> orchestration call, not a research one)."*

Treat this module as **stable shared infrastructure**. Variants of a
pipeline stage arrive as new `Stage` values or wrappers around existing
stages — the runner itself doesn't change.

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

## Role in the iteration harness

Never a variable under test — but two places as a hosting layer:

- **Variant A/B runner.** A/B comparisons between a variant stage and a
  control stage build two `Pipeline` instances that share every stage
  except one. `output/experiments/` write-ups use this pattern.
- **Telemetry collection.** `RunState.stages` carries per-stage timing and
  success. Cost and latency per run are read out of `RunState` after
  `pipeline.run()` returns.

## Out of scope

- Parallel execution, conditional branching, resume-from-stage, Postgres
  persistence — all deliberately out of scope. Those are orchestration
  engineering calls, tracked in `PRD_ORCHESTRATION.md`, not validated
  through the iteration program here.
