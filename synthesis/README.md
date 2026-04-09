# synthesis

**Stage 3.** Turn one cluster summary into one structured `PersonaV1` JSON
object. This module is the **highest-traffic lab module** — three of the six
problem spaces (1, 2, 3) live here, and parts of space 6 reach in too.

## What lives here

```
synthesis/
├── synthesis/
│   ├── config.py              # Settings (API key, model, max_retries)
│   ├── models/
│   │   ├── persona.py         # PersonaV1 — THE schema. Space 1 lives here.
│   │   ├── evidence.py        # SourceEvidence — claim → record_ids mapping
│   │   └── cluster.py         # ClusterData — the input contract
│   └── engine/
│       ├── synthesizer.py     # synthesize() — orchestrates the call + retry
│       ├── prompt_builder.py  # SYSTEM_PROMPT, build_messages, tool def
│       ├── model_backend.py   # ModelBackend protocol + AnthropicBackend
│       └── groundedness.py    # Deterministic post-generation check
├── .env.example               # Copy to .env and fill ANTHROPIC_API_KEY
└── pyproject.toml
```

## I/O contract

Input: a `ClusterData` instance (or any dict that validates against it).
See `models/cluster.py`.

Output: a `SynthesisResult`:

```python
@dataclass
class SynthesisResult:
    persona: PersonaV1
    groundedness: GroundednessReport   # score ∈ [0,1] + violation list
    total_cost_usd: float
    model_used: str
    attempts: int                      # 1 on first-try success, ≤ max_retries+1
```

`PersonaV1` is the structured JSON schema. Today it has ~12 fields:
`name`, `summary`, `demographics`, `firmographics`, `goals`, `pains`,
`motivations`, `objections`, `channels`, `vocabulary`, `decision_triggers`,
`sample_quotes`, `journey_stages`, `source_evidence`.

Groundedness is enforced by two deterministic checks:
1. Every `source_evidence.record_ids` entry must reference a real record
   from the cluster.
2. Every item in `goals`, `pains`, `motivations`, `objections` must have
   at least one valid `source_evidence` entry pointing at it via
   `field_path` (e.g. `"goals.0"`).

A report with `score >= 0.9` passes. Failures trigger a retry with the
violation list injected into the next prompt.

## How to run standalone

```python
import asyncio
from anthropic import AsyncAnthropic
from synthesis.config import settings
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import synthesize
from synthesis.models.cluster import ClusterData

async def main():
    cluster = ClusterData.model_validate({...})   # or load from segmentation
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)
    result = await synthesize(cluster, backend)
    print(result.persona.name, result.groundedness.score, f"${result.total_cost_usd:.4f}")

asyncio.run(main())
```

## Knobs you can turn

**In `models/persona.py` (Space 1 — schema):**
- Add or remove fields on `PersonaV1`.
- Change `min_length` / `max_length` budgets per field.
- Split or merge substructures (`Demographics`, `Firmographics`, `JourneyStage`).
- Introduce a numeric psychometric backbone (Big Five, attachment style).
- Create `PersonaV1_1`, `PersonaV2`, etc. as parallel schemas for A/B runs.

**In `engine/prompt_builder.py` (Space 2 — pipeline architecture):**
- `SYSTEM_PROMPT` — rewrite the quality criteria / evidence rules.
- `build_user_message()` — change the order fields are presented in (tests
  the "order-of-fields effect" hypothesis).
- `build_tool_definition()` — today it emits the whole `PersonaV1` schema.
  Decompose into per-section tools for the monolithic-vs-decomposed experiment.
- `build_retry_messages()` — change how critique feedback is injected.

**In `engine/synthesizer.py` (Space 2 — critique loops, model-mix tiering):**
- `MAX_RETRIES` — how many revision rounds.
- Bolt on a critic model pass between attempts (evaluator-optimizer loop).
- Swap `AnthropicBackend` for a fan-out backend that calls one section at
  a time and stitches.

**In `engine/model_backend.py` (Space 2 — tool-use vs JSON mode vs free):**
- `AnthropicBackend.generate()` currently uses `tool_choice={"type":"tool"}`.
  Variants: drop the forcing, switch to plain JSON mode, parse from raw text.

**In `engine/groundedness.py` (Space 3 — groundedness & evidence):**
- `EVIDENCE_REQUIRED_FIELDS` — which fields must carry citations.
- The 0.9 pass threshold in `GroundednessReport.passed`.
- Add an NLI / entailment check that actually reads the cited record payload
  instead of just checking that the ID exists (experiment 3.5).
- Retrieval-augmented synthesis (3.3): before calling the backend, retrieve
  top-k records per persona section via embedding search.

## Where this module shows up in the experiment catalog

This is ground zero for three problem spaces. **Coordinate before merging.**

| Space | Experiments | Primary files |
|---|---|---|
| 1 (schema) | 1.1 schema width · 1.2 structured vs narrative · 1.3 vocabulary anchoring · 1.4 trait crystallization · 1.5 schema versioning · 1.6 self-describing · 1.11 negative space · 1.14 belief/value split · 1.17 length budgets · 1.20 verbs vs adjectives | `models/persona.py` |
| 2 (pipeline) | 2.1 monolithic vs decomposed · 2.2 critique loops · 2.3 model-mix tiering · 2.4 tool-use vs JSON · 2.5 few-shot examples · 2.6 temperature sweep · 2.7 order-of-fields · 2.8 synthetic warmstart | `engine/prompt_builder.py`, `engine/synthesizer.py`, `engine/model_backend.py` |
| 3 (grounding) | 3.1 citation-required · 3.2 prompt vs post-hoc validation · 3.3 retrieval-augmented · 3.4 evidence-first · 3.5 NLI check · 3.6 sparse-data ablation · 3.7 adversarial injection | `engine/groundedness.py`, `engine/prompt_builder.py` |

Space 6 (distinctiveness) also edits this module for **6.4 cross-persona
contrast prompting**: when synthesizing persona N+1, include personas 1..N
with an instruction to differ on specific axes. This means the synthesizer
needs to accept an `existing_personas` argument and build that into the
prompt. Coordinate with space 1/2 on the API change.

## Default-is-sacred rule

Space 1/2/3 researchers are all editing files in this directory. Please:

1. **Don't change the default behavior** of `synthesize()`. Add new keyword
   arguments with defaults that preserve today's behavior.
2. **Don't rename** `PersonaV1`. If you need a new schema, add `PersonaV1_1`,
   `PersonaV2`, etc. alongside.
3. **Branch per experiment.** Never merge an experiment-specific default
   into main without a recorded decision in the space's results file.

## Tests

Parent repo has `synthesis/synthesis/tests/` covering models, prompt_builder,
groundedness, and the synthesizer retry loop. Copy them into this bundle
before starting — they're your safety net when you're ripping up the schema.
