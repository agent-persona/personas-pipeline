# Experiment 4.15 — Cold-Start vs Warm Conversation

## Hypothesis

Warmup prefix reduces cold-start delta by ≥50% — turn 1 realism with warmup approaches turn 5 realism without warmup.

## Method

- Synthesized 1 persona ("Alex Chen, The Infrastructure Architect") from the largest cluster
- **Condition A (cold):** 10-turn product-research interview, no warmup
- **Condition B (warm):** Same 10 prompts, same persona, with 3-exchange warmup prefix injected via `TwinChat(warmup_turns=...)`
- Claude-as-judge rated realism (1-5) at turns 1, 3, 5, 7, 10
- Model: `claude-haiku-4-5-20251001`

## Results

| Turn | Cold Realism | Warm Realism |
|------|-------------|-------------|
| 1    | 5           | 5           |
| 3    | 5           | 5           |
| 5    | 5           | 4           |
| 7    | 4           | 4           |
| 10   | 5           | 4           |

- **Cold delta (turn 1 - turn 5):** 0 (no cold-start gap detected)
- **Warm delta (turn 1 - turn 5):** 1
- **Reduction:** N/A (cold delta is 0, cannot divide)
- **Turn 1 warm boost:** 0 (both start at 5)
- **Late-turn mean (7, 10):** Cold = 4.5, Warm = 4.0

## Verdict

**FAIL** — but for an unexpected reason. The hypothesis assumed a cold-start quality gap (turn 1 worse than later turns). In practice, Haiku produces high-realism responses from turn 1 even without warmup. The persona system prompt alone is sufficient to establish character immediately.

The warmup prefix actually introduced slight degradation at later turns (warm condition averaged 4.0 vs cold's 4.5 at turns 7-10), possibly because the extra prefix tokens diluted attention on the accumulated conversation history.

## Implications

- Cold-start is not a meaningful problem with well-constructed persona system prompts on Haiku
- Warmup prefixes may be counterproductive for short conversations
- The persona `build_persona_system_prompt()` function is doing its job: establishing character from the first turn
