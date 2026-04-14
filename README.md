# personas-pipeline

End-to-end persona-framework pipeline. Every default in this codebase — the
schema shape, the prompt structure, the retry policy, the judge rubric, the
clustering knobs — is the product of **hypothesis-driven experiments** run
against a frozen golden set. The repo doubles as the artifact and the harness:
you can ship with it as-is, or pull the same levers the experiments pulled and
iterate further.

Evidence of the iteration lives under `output/experiments/` (raw results,
findings) and `docs/plans/` (batch strategies and write-ups). Module layout:

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
and `README.md`. You can `pip install -e .` one module at a time and work on
it in isolation, or run the whole pipeline end-to-end with the top-level
script.

---

## Problem space → module map

The pipeline is organized around six problem spaces. Each space has a body
of experimental work that shaped the defaults in the corresponding module.

| # | Problem space | Primary module | Primary files |
|---|---|---|---|
| 1 | Persona representation & schema | `synthesis/` | `synthesis/models/persona.py`, `synthesis/models/evidence.py` |
| 2 | Synthesis pipeline architecture | `synthesis/` | `synthesis/engine/synthesizer.py`, `synthesis/engine/prompt_builder.py`, `synthesis/engine/model_backend.py` |
| 3 | Groundedness & evidence binding | `synthesis/` | `synthesis/engine/groundedness.py`, `synthesis/engine/prompt_builder.py` |
| 4 | Twin runtime: consistency & drift | `twin/` | `twin/twin/chat.py` |
| 5 | Evaluation & judge methodology | `evaluation/` | `evaluation/judges.py`, `evaluation/metrics.py`, `evaluation/golden_set.py` |
| 6 | Population distinctiveness & coverage | `segmentation/` + `synthesis/` | `segmentation/engine/clusterer.py`, `synthesis/engine/synthesizer.py` (fan-out / contrast prompting) |

Spaces 1, 2, 3 all live inside `synthesis/`. New knobs are added as keyword
arguments with defaults that preserve current behavior, so prior results stay
reproducible and further iteration stays cheap.

---

## Quickstart

```bash
# 1. Clone / cd into personas-pipeline
cd personas-pipeline

# 2. Install every module in editable mode so imports resolve without PYTHONPATH hacks
pip install -e crawler -e segmentation -e synthesis -e twin -e orchestration -e evaluation
pip install python-dotenv                    # used by the end-to-end script

# 3. Drop in your Anthropic API key
cp synthesis/.env.example synthesis/.env
# then edit synthesis/.env and set ANTHROPIC_API_KEY=...

# 4. Run the full pipeline (crawler → segment → synthesize → twin → persist)
python scripts/run_full_pipeline.py
```

Output personas land in `output/persona_*.json`. The script uses Haiku by
default — the tiering choice came out of model-mix experiments in space 2, and
keeps per-run cost in the cents. Swap the model in `synthesis/.env`
(`default_model=claude-sonnet-4-6` etc.) to re-evaluate the tradeoff on your
own inputs.

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

Every arrow is a stable JSON contract. Any single stage can be swapped for an
alternate implementation as long as the contract holds. Several of the
experiments under `output/experiments/` do exactly this — replace one stage,
hold the others fixed, measure. See each module's README for its I/O shape.

---

## The evaluation harness

The defaults shipped here were validated against a shared harness:

1. **A hypothesis**, written before the run.
2. **A control** — the same golden input with the default config.
3. **A metric** — one of the shared metrics in `evaluation/metrics.py`
   (schema validity, groundedness, distinctiveness, judge score, drift, cost).
4. **A result + decision** — adopt / reject / defer, recorded alongside the
   run under `output/experiments/<exp-id>/`.

The harness isn't scaffolding that's still being built — it's the thing that
produced the current defaults, and it's the thing you re-run when you want to
push them further. `evaluation/golden_set.py` + the mock tenant in
`crawler/crawler/connectors/` give you the exact control input the existing
findings were measured against.

---

## Conventions

These fall out of the iterative approach — they exist so each round of
iteration can be compared cleanly against the last.

- **One module = one owner at a time.** Changes that cross modules (space 6
  does) need a heads-up before the second module is touched.
- **Default behavior is sacred.** New knobs arrive as flags with
  behavior-preserving defaults. A downstream control run depends on yours
  still matching it.
- **Variants live on branches or behind config.** `main` carries validated
  defaults, not in-flight experiments.
- **Cost is recorded.** Every run's Anthropic spend is part of the result —
  the model-mix findings turn on it.

---

## Where to find things

- `output/experiments/` — the empirical record. Findings, raw outputs, and
  per-experiment write-ups.
- `docs/plans/` — batch-level research strategy and experiment results
  summaries.
- `PRD_SYNTHESIS.md`, `PRD_SEGMENTATION.md`, `PRD_TESTING.md` — product
  context for the modules, including the hypotheses that shaped them.
