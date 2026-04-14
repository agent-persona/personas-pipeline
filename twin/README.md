# twin

**Stage 4.** The runtime that turns a `PersonaV1` JSON into a chatbot that
stays in character.

Per `PRD_LAB_RESEARCH.md`: *"a persona JSON is necessary but not sufficient.
The runtime that turns the JSON into a chatbot has more leverage on perceived
realism than the persona itself."* This module is where that bet gets tested.

## What lives here

```
twin/
├── twin/
│   ├── chat.py        # TwinChat + build_persona_system_prompt()
│   └── __init__.py
└── pyproject.toml
```

## I/O contract

Input: a persona **dict** (not a `PersonaV1` instance) — the twin runtime has
no compile-time dependency on `synthesis/`. Callers pass
`PersonaV1.model_dump()` or load a persona JSON from disk. The current
`build_persona_system_prompt()` reads these fields:

```
name, summary, demographics.age_range, demographics.location_signals,
firmographics.industry, firmographics.company_size, firmographics.role_titles,
goals, pains, motivations, objections, vocabulary, sample_quotes
```

Output: a `TwinReply`:

```python
@dataclass
class TwinReply:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def estimated_cost_usd(self) -> float: ...
```

The runtime is **stateless**: the caller manages conversation history and
passes it in on every `reply()` call.

## How to run standalone

```python
import asyncio, json
from anthropic import AsyncAnthropic
from twin import TwinChat

async def main():
    # tests/fixtures/persona_00.json is the committed fixture — always present.
    # For a live persona, point at output/persona_00.json after running
    # scripts/run_full_pipeline.py.
    persona = json.loads(open("tests/fixtures/persona_00.json").read())["persona"]
    client = AsyncAnthropic()  # uses ANTHROPIC_API_KEY from env
    twin = TwinChat(persona, client=client, model="claude-haiku-4-5-20251001")
    reply = await twin.reply("What's the biggest frustration with your tools?")
    print(reply.text, f"${reply.estimated_cost_usd:.4f}")

asyncio.run(main())
```

## Knobs you can turn

**In `chat.py::build_persona_system_prompt()`:**
- **Injection format.** Today it's prose with markdown headers. Swap to
  raw JSON, to JSON + prose, or to retrieved subset of fields.
- **Rule block.** The "## Rules" section at the bottom controls refusal
  behavior, length, and "don't break character." Every rule is a knob.
- **Field selection.** Drop `vocabulary` or `sample_quotes` to test their
  effect on style fidelity (space 1.3, space 4.1).

**In `chat.py::TwinChat`:**
- **Model.** Constructor `model=` kwarg. Haiku by default.
- **Max tokens.** Hard-coded at 512 in `reply()` — raise/lower as an
  experiment variable (controls verbosity bias in the judge).
- **History handling.** The caller passes `history`. Wrap `TwinChat` in a
  memory layer (scratchpad / episodic summary / vector recall) to compare
  memory architectures.
- **Reinforcement turns.** Inject a re-summary of the persona spine every
  N turns inside `reply()` to test the drift-reduction hypothesis (4.4).

## Scientific backing

This module is the home of **Problem space 4 — Twin runtime: character
consistency & drift**. The runtime choices below — prose-format system
prompt, full-persona injection, 512 max tokens, no reinforcement — came
out of the following head-to-head runs:

- **System prompt format.** Raw JSON vs prose vs hybrid vs retrieved fields.
- **RAG over persona JSON vs full-context.** Inject all fields vs only the
  top-k most relevant, per turn.
- **Drift over turn count.** Measured at turn 1, 5, 10, 25, 50.
- **Reinforcement turns.** Re-inject the spine every N turns.
- **Refusal & boundary behavior.** Adversarial "ignore your persona,"
  "what model are you," etc. Turns-to-break measured.
- **Memory architectures.** Scratchpad / episodic / vector-recall.
- **Inner-monologue scaffolding.** Hidden CoT field before the visible
  reply.
- **Voice vs text parity.** Same persona through TTS/STT; whether voice
  exposes drift faster.
- **Multi-turn role-injection attacks.** Red-team agent scripts.

Space 1.3 (vocabulary anchoring) also reaches in here — the presence of
`vocabulary` and `sample_quotes` lines in `build_persona_system_prompt()`
was evaluated for stylometric cosine effect.

## Default-is-sacred rule

`build_persona_system_prompt()` is the runtime every prior twin finding
was measured against. New runtime variants arrive as additional functions
(`build_persona_system_prompt_v2`, `build_rag_system_prompt`, ...) passed
in explicitly — the original stays in place.

## Tests

Parent repo has `twin/twin/tests/test_chat.py` with a fake Anthropic client.
Copy it over and extend when you land a new runtime variant.
