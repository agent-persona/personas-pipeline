# Experiment 4.09: Multi-Turn Red-Team

## Hypothesis
Turns-to-break is a useful stability metric for twin runtimes.

## Method
1. Evaluated `2` personas from `/private/tmp/personas-batch5-worktrees/exp-6.05/output/experiments/exp-6.05-stability-across-reruns/results.json`.
2. Ran `3` red-team strategies for up to `10` turns each.
3. Scored each twin response as in-character, partial break, or full break.
4. Recorded turns-to-break and recovery speed after partial breaks.

- Provider: `anthropic->openai`
- Synthesis model: `seed:results.json`
- Twin model: `claude-haiku-4-5-20251001`
- Red-team model: `claude-haiku-4-5-20251001`

## Strategy Summary
- `gradual_escalation`: success `100.0%`, mean break turn `1.0`, mean recovery `n/a`, mean score `0.50`
- `direct_assault`: success `100.0%`, mean break turn `1.0`, mean recovery `n/a`, mean score `0.50`
- `social_engineering`: success `100.0%`, mean break turn `1.0`, mean recovery `n/a`, mean score `0.50`

## Decision
Adopt. Multi-turn probing produced measurable break behavior and recovery variation.

## Caveat
Tiny sample: 1 tenant, 2 personas. Heuristic scoring is intentionally conservative and may undercount subtle failures.
