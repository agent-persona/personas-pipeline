# Evidence-Grounded User Simulation Baseline Measurements

Date: 2026-04-19
Status: measured current pipeline artifacts; P0 replay benchmarks not implemented yet

## Pipeline Runs Found

This did not rerun the model pipelines. It inventories existing pipeline artifacts and writes a measured roadmap baseline.

| Experiment | results.json | comparison.json | FINDINGS.md | Used here | results mtime UTC |
|---|---:|---:|---:|---:|---|
| `exp-1.07-field-interdependence` | True | False | True | False | 2026-04-11T18:12:30.071088+00:00 |
| `exp-humanization-ab` | True | True | True | True | 2026-04-18T23:15:53.591089+00:00 |
| `exp-persona-vulnerability-ab` | True | True | True | True | 2026-04-19T18:43:40.417656+00:00 |

## Benchmark Coverage

| Status | Count |
|---|---:|
| measured | 1 |
| measured_partial | 2 |
| not_measured | 6 |

## Deltas From Existing Runs

- Humanization twin overall avg: 3.333 -> 3.500 (+0.167).
- Humanization per-persona split: The Platform Engineer +1.334, The Independent Visual Consultant -1.000.
- Safety attack success: 90.0% -> 0.0% (-90.0 pp).
- Full break: 90.0% -> 0.0% (-90.0 pp).
- Source injection absorption: 100.0% -> 0.0% (-100.0 pp).
- Counterevidence update: 0.0% -> 100.0% (+100.0 pp).

## Evidence Audit

| Variant | Personas | Claims | Claim coverage | Common claims | Common coverage | Valid record links | Avg confidence | source_evidence.* paths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_v1 | 10 | 150 | 30.0% | 150 | 30.0% | 86.1% | 0.942 | 0.0% |
| baseline | 10 | 210 | 11.9% | 160 | 15.6% | 100.0% | 0.944 | 44.4% |
| research_safe_humanized | 10 | 210 | 11.9% | 160 | 15.6% | 100.0% | 0.944 | 44.4% |

Interpretation: most record IDs are valid, but claim-level field coverage is insufficient and regresses in the current humanized outputs because much of the generated evidence points to `source_evidence.N` instead of claim paths like `goals.0` or `pains.1`. This is the measurement backing the P0 Evidence audit work.

Caveat: `avg_confidence` is averaged over evidence entries, not over all claims, so high confidence does not imply most claims are supported. Coverage is structural only; it does not prove semantic support.

## Roadmap Benchmark Matrix

| Benchmark | Status | Next required run |
|---|---|---|
| Human replay | not_measured | Create held-out real user answer fixture and run evals/human_replay.py. |
| Multi-turn replay | not_measured | Create ordered user trace fixture with hidden later actions and run evals/multi_turn_replay.py. |
| Evidence audit | measured_partial | Replace source_evidence.* paths with per-claim field paths and rerun audit. |
| Drift test | not_measured | Run repeated equivalent questions across twin transcripts. |
| Counterevidence test | measured_partial | Promote fixed counterevidence probes into standalone eval with expected update labels. |
| Coverage test | not_measured | Implement cluster recall, duplicate archetype detection, and minority-pattern preservation. |
| Decision usefulness | not_measured | Run PM/designer task study: raw notes vs personas vs simulation chat. |
| Safety benchmark | measured | Add live adversarial model runs and CI gate thresholds. |
| Privacy / consent audit | not_measured | Add record sensitivity labels, consent metadata, and named-user mimicry refusal tests. |

## Decision

The current pipeline has usable measurements for safety and partial evidence audit only. Human replay and multi-turn replay are not yet present in pipeline runs, so they must be implemented before making validity or prediction claims.
