# exp-4.15 — Cold-Start vs Warm Conversation

**Branch:** `exp-4.15-cold-start-warmup`
**Guide:** Guide 4 — Twin Runtime
**Date:** 2026-04-12
**Status:** FAIL — No cold-start gap exists; warmup prefix slightly degrades late-turn realism

## Hypothesis

Warmup prefix reduces cold-start delta by >=50% — turn 1 realism with warmup approaches turn 5 realism without warmup.

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The control condition in this experiment (no warmup) matches the default pipeline TwinChat behavior. The shared baseline personas confirm the pipeline produces valid, grounded personas at low cost. The treatment tests whether `warmup_turns` improve early-turn realism in TwinChat conversations.

## Method

Synthesized 1 persona ("Alex Chen, The Infrastructure Architect") from the largest cluster. Two TwinChat conditions:

- **Condition A (cold/baseline):** 10-turn product-research interview, no warmup (default pipeline behavior)
- **Condition B (warm/treatment):** Same 10 prompts, same persona, with 3-exchange warmup prefix injected via `TwinChat(warmup_turns=...)`

Claude-as-judge rated realism (1-5) at turns 1, 3, 5, 7, 10. Model: `claude-haiku-4-5-20251001`.

## Results

### Quantitative

| Turn | Cold Realism | Warm Realism |
|------|-------------|-------------|
| 1    | 5           | 5           |
| 3    | 5           | 5           |
| 5    | 5           | 4           |
| 7    | 4           | 4           |
| 10   | 5           | 4           |

| Metric | Cold (baseline) | Warm (treatment) | Delta |
|---|---|---|---|
| delta (turn 1 - turn 5) | 0 | 1 | -- |
| Reduction | N/A (cold delta is 0) | -- | -- |
| Turn 1 warm boost | -- | 0 | -- |
| Late-turn mean (7, 10) | 4.5 | 4.0 | -0.5 |

### Key findings

1. **No cold-start gap exists.** The cold condition achieves realism=5 at turn 1, leaving no deficit for warmup to fix. The persona system prompt alone establishes character immediately.
2. **Warmup prefix slightly degrades late turns.** Warm condition averaged 4.0 vs cold's 4.5 at turns 7-10, possibly because extra prefix tokens diluted attention on accumulated conversation history.
3. **Hypothesis is moot.** With delta_cold=0, the 50% reduction target cannot be evaluated. Cold-start is not a meaningful problem with well-constructed persona system prompts on Haiku.

## Recommendation

FAIL — No action needed. The `build_persona_system_prompt()` function is sufficient. Do not add warmup prefix complexity to the pipeline.

**Action items:**
1. Close this line of investigation; warmup is unnecessary with current persona prompts
2. If future models exhibit cold-start behavior, revisit with a model that actually shows turn-1 degradation

## Cost

- Total API cost: included in summary (1 synthesis + 20 conversation turns + 10 judge calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — cold vs warm realism scores per turn
- `conversations.json` — full transcripts for both conditions
