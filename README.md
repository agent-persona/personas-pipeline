# personas-pipeline

End-to-end persona-framework pipeline, packaged for the lab research program.

Six researchers each own one of the six problem spaces from `PRD_LAB_RESEARCH.md`
(and the `lab-research.html` dashboard). Every problem space maps to exactly one
module directory below — that's where you make changes, run experiments, and
record results.

```
personas-pipeline/
├── crawler/          # Stage 1: pull behavioral records from sources
├── segmentation/     # Stage 2: cluster records into behavioral segments
├── synthesis/        # Stage 3: turn clusters into structured personas
├── twin/             # Stage 4: chat in character as a persona
├── orchestration/    # Glue: sequential DAG runner that wires the stages
├── evaluation/       # Judges, golden set, metrics (scaffold — researcher #5)
├── scripts/          # run_full_pipeline.py — end-to-end demo
└── output/           # persona_*.json dumped by the pipeline
```

Each module is an independent Python package with its own `pyproject.toml`
and `README.md`. You can `pip install -e .` one module at a time and work on
it in isolation, or run the whole pipeline end-to-end with the top-level
script.

---

## Problem space → module map

Use this when picking where to put your experiment code.

| # | Problem space | Primary module | Primary files |
|---|---|---|---|
| 1 | Persona representation & schema | `synthesis/` | `synthesis/models/persona.py`, `synthesis/models/evidence.py` |
| 2 | Synthesis pipeline architecture | `synthesis/` | `synthesis/engine/synthesizer.py`, `synthesis/engine/prompt_builder.py`, `synthesis/engine/model_backend.py` |
| 3 | Groundedness & evidence binding | `synthesis/` | `synthesis/engine/groundedness.py`, `synthesis/engine/prompt_builder.py` |
| 4 | Twin runtime: consistency & drift | `twin/` | `twin/twin/chat.py` |
| 5 | Evaluation & judge methodology | `evaluation/` | `evaluation/judges.py`, `evaluation/metrics.py`, `evaluation/golden_set.py` |
| 6 | Population distinctiveness & coverage | `segmentation/` + `synthesis/` | `segmentation/engine/clusterer.py`, `synthesis/engine/synthesizer.py` (fan-out / contrast prompting) |

Spaces 1, 2, 3 all live inside `synthesis/` — coordinate on merges so you're not
stepping on each other. If your experiment needs a new knob, add it as a
keyword argument with a default that preserves current behavior, not a rewrite.

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
default to keep per-run cost in the cents. Swap the model in
`synthesis/.env` (`default_model=claude-sonnet-4-6` etc.) to test a different
tier.

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

Every arrow is a stable JSON contract. A researcher can replace any one stage
with an alternate implementation as long as they honor the contract with the
stages on either side. See each module's README for its I/O shape.

---

## Shared harness (required for all experiments)

Per `PRD_LAB_RESEARCH.md`, no experiment lands without:

1. **A hypothesis** — written before the run.
2. **A control** — a run on the same golden input with the default config.
3. **A metric** — one of the shared metrics in `evaluation/metrics.py`
   (schema validity, groundedness, distinctiveness, judge score, drift, cost).
4. **A result + decision** — adopt / reject / defer, written up in your space's
   results file.

Until the golden set is frozen, use the mock tenant in
`crawler/crawler/connectors/` (engineers + designers, two clean clusters) as
the control input.

---

## Coordination rules

- **One module = one researcher at a time.** If your experiment crosses
  modules (space 6 does), open a thread before editing the other module.
- **Default behavior is sacred.** Add flags, don't rewrite defaults. Someone
  else's control run depends on yours.
- **Every experiment gets a branch or a config tag.** Don't mutate `main`
  with an experiment-specific default.
- **Record the cost** of every run. All experiments draw from the same
  Anthropic budget.

---

## Where to find things

- `PRD_LAB_RESEARCH.md` (parent repo) — the full experiment catalog.
- `lab-research.html` (parent repo) — the six-space dashboard the researchers
  work from day-to-day.
- `PRD_SYNTHESIS.md`, `PRD_SEGMENTATION.md`, `PRD_TESTING.md` — product context
  for the modules you'll be modifying.
