# Handoff: 4 Experiment Branches (2.06, 1.07, 5.05, 5.11)

Repo: `/Users/maxpetrusenko/Desktop/Gauntlet/personas-pipline`

Goal: create and run 4 separate experiment branches for guide tasks 2.06, 1.07, 5.05, and 5.11.

---

## Results

See **[batch4-experiment-results.md](2026-04-11-batch4-experiment-results.md)** for full analysis, interpretation, caveats, and next steps.

| Branch | Decision | Confidence |
|--------|----------|------------|
| exp-1.07 field interdependence | **Adopt** | Narrow (1 tenant, coarse scores) |
| exp-2.06 temperature sweep | **Adopt** | Moderate (2 clusters, one 429 noise) |
| exp-5.11 reference vs free judging | **Reject** | High (clearest negative) |
| exp-5.05 rubric ablation | **Defer** | Low (ceiling effect, degenerate stats) |

---

## Hidden dependencies — read this first

- On `origin/main`, `evaluation/evaluation/judges.py` is a **stub** (~80 lines, `score_persona()` returns NaN).
- The real working judge implementation lives on `origin/exp-5.13` (341 lines: `JudgeBackend` class, rubric prompts, calibration anchors, working `score()` via Anthropic API).
- Experiments **5.05 and 5.11 must branch from `origin/exp-5.13`**.
- If 1.07 or 2.06 needs judge scoring, **reuse the exp-5.13 judge implementation intact** or in a local helper module. Do not partially port a reduced `judges.py` onto main.
- `evaluation/evaluation/golden_set.py` does **not** contain real gold personas yet (`ideal_personas=[]`). For 5.11, use a clearly labeled **proxy reference persona** (derived from a control run or hand-crafted), not a claimed gold-standard artifact.
- The golden tenant `tenant_acme_corp` produces only ~2 personas. If sample size is too small for stable rank metrics (Kendall tau, Spearman), run multiple synthesis repeats or generate multiple candidate personas per cluster. Report simpler deltas/std-dev if rank stats are degenerate.

---

## Shared rules

- Follow the shared harness: every experiment needs a written hypothesis, a control on the same input, explicit metrics, saved results under `output/experiments/exp-{id}/`, and an adopt/reject/defer decision.
- Default behavior is sacred. Add optional flags/knobs only. Do not change current defaults in any shared module.
- Keep each experiment isolated on its own branch. Do not stack.
- Prefer full slugged branch names (`exp-X.XX-descriptive-name`).
- Add a test file when new logic is nontrivial.
- End each branch with: committed code, run artifacts (results.json), findings markdown, and a short summary of what commands were run.
- Commit format: `"exp X.XX: description"` (append `"— result [adopt/reject/defer]"` after running).

---

## Before coding each branch

Inspect the template branches listed below. Use `git show origin/BRANCH:path/to/file` to read their patterns. Do NOT blindly copy — understand the structure, then write your own code following the same conventions.

---

## Branch creation commands

Run these exact commands. Do not improvise remote ref handling.

```bash
git fetch origin

# Branches 3 and 4 base from exp-5.13 (real judge)
git switch -c exp-5.05-rubric-ablation origin/exp-5.13
# ... work, commit, push, then:
git switch main

git switch -c exp-5.11-reference-based-vs-reference-free origin/exp-5.13
# ... work, commit, push, then:
git switch main

# Branches 1 and 2 base from main
git pull --ff-only
git switch -c exp-1.07-field-interdependence
# ... work, commit, push, then:
git switch main

git pull --ff-only
git switch -c exp-2.06-temperature-sweep
# ... work, commit, push, then:
git switch main
```

---

## Branch 1: exp-2.06-temperature-sweep

**Base:** `origin/main`

**Templates to study:**
- `origin/exp-1.17-length-budgets-per-field` — multi-variant synthesis runner with sweep grid (`scripts/experiment_1_17.py`, 433 lines)
- `origin/exp-1.19-schema-artifact-format` — alternate variant pattern
- `origin/exp-1.20` — short-style experiment for reference

**What to do:**
- Add optional `temperature` and `top_p` parameters to the synthesis backend WITHOUT changing defaults. Subclass `AnthropicBackend` or add optional kwargs that default to `None` (meaning API defaults apply).
- Run a 2D sweep: `temperature ∈ {0.0, 0.4, 0.7, 1.0} × top_p ∈ {0.8, 1.0}` on the golden tenant (`tenant_acme_corp`). If runtime is prohibitive, run temperature sweep first; add top_p as phase 2 if signal is weak.
- Control = no temperature/top_p specified (API defaults).
- Measure per variant: groundedness score (from synthesizer result), schema validity / success rate, retry rate (attempts > 1), cost per persona.
- Distinctiveness can be measured heuristically (e.g., vocabulary overlap between personas within tenant). Judge scoring is optional — do not require it unless the heuristic proves insufficient.

**Result: ADOPT** — temp=0.0 saves 30% cost, 50% fewer retries. top_p had no signal. API forbids simultaneous temperature+top_p so sweep was run independently.

**Deliverables:**
- `scripts/experiment_2_06.py` (runner)
- Minimal changes in `synthesis/synthesis/engine/model_backend.py` (optional kwargs, defaults unchanged)
- `output/experiments/exp-2.06-temperature-sweep/results.json`
- `output/experiments/exp-2.06-temperature-sweep/FINDINGS.md`
- `tests/test_exp_2_06.py` (optional)

---

## Branch 2: exp-1.07-field-interdependence

**Base:** `origin/main`

**Templates to study:**
- `origin/exp-1.17-length-budgets-per-field` — multi-variant harness pattern
- `origin/exp-6.04` — baseline vs variant comparison structure

**What to do:**
- Generate a full persona (control) using the standard pipeline on golden tenant.
- Ablate persona fields **in a copy of the output artifact** and judge the degraded result. This is NOT rerunning synthesis with schema changes — it's post-hoc field removal + re-scoring.
- For each removable field (`goals`, `pains`, `motivations`, `objections`, `channels`, `vocabulary`, `decision_triggers`, `sample_quotes`, `journey_stages`): create an ablated copy with that field set to empty list `[]`, then re-score with the judge.
- If judge scoring is needed, either cherry-pick/copy the exp-5.13 judge implementation intact, or create a local experiment-only judge helper derived from it. Do not partially reimplement `judges.py` on main.
- Build a dependency matrix: rows = removed field, columns = scoring dimension, cells = score delta from control.
- Classify fields as load-bearing (removal drops 2+ dimensions significantly) vs decorative.

**Result: ADOPT** — goals + sample_quotes are load-bearing (2+ dim drops). channels/vocabulary/journey_stages are decorative. Surprise: pains had zero net impact (coherent -1 offset by actionable +1).

**Deliverables:**
- `evals/field_interdependence.py` (ablation harness + dependency graph builder)
- `scripts/experiment_1_07.py` (runner)
- `output/experiments/exp-1.07-field-interdependence/results.json`
- `output/experiments/exp-1.07-field-interdependence/FINDINGS.md`
- `tests/test_exp_1_07.py` (optional)

---

## Branch 3: exp-5.05-rubric-ablation

**Base:** `origin/exp-5.13` (to get the real judges.py implementation)

**Templates to study:**
- `origin/exp-5.02-cross-judge-agreement` — eval harness structure (`evals/judge_harness.py`, ~510 lines)
- `origin/exp-5.04-position-verbosity-bias` — eval reporting pattern
- `origin/exp-3.19b-recency-real-fixture` — mature `FINDINGS.md` format

**What to do:**
- Base from exp-5.13 so you start with a working judge.
- Add rubric override support: the judge should accept a custom dimension list so it can run full rubric vs one-dimension-dropped variants.
- Generate personas from golden tenant. If sample size is too small for stable rank metrics, run multiple synthesis repeats or multiple candidate personas per cluster.
- Score each persona with full 5-dimension rubric (control).
- For each dimension: build an ablated rubric prompt excluding that dimension, re-score all personas.
- Compute: pairwise correlation between dimensions, ranking stability (Kendall tau if sample permits, otherwise simpler deltas/std-dev), score shift in surviving dimensions.
- Identify redundant dimensions (correlation > 0.95) and inert dimensions (removal doesn't change rankings).

**Result: DEFER** — Ceiling effect (all 5 personas scored 4-5) prevents definitive correlation analysis. Discovered two-tier rubric structure: anchor dims (distinctive, actionable, voice_fidelity) depress other scores when removed; independent dims (grounded, coherent) have zero cross-impact. Need intentionally degraded personas for wider quality range.

**Deliverables:**
- `evals/rubric_ablation.py` (parameterized rubric builder + ablation harness)
- `scripts/experiment_5_05.py` (runner)
- `output/experiments/exp-5.05-rubric-ablation/results.json`
- `output/experiments/exp-5.05-rubric-ablation/FINDINGS.md`
- `tests/test_exp_5_05.py` (optional)

---

## Branch 4: exp-5.11-reference-based-vs-reference-free

**Base:** `origin/exp-5.13` (to get the real judges.py implementation)

**Templates to study:**
- `origin/exp-5.02-cross-judge-agreement` — dual-mode comparison pattern
- `origin/exp-6.09-color-palette-view` — mature `FINDINGS.md`
- `origin/exp-5.04-position-verbosity-bias` — bias/anchoring measurement

**What to do:**
- Base from exp-5.13 so you start with a working judge.
- Add an optional reference-persona judging mode: a modified rubric prompt that includes a **proxy reference persona** (derived from a control run or hand-crafted) alongside the candidate. Clearly label it as a proxy, not a true gold persona.
- Generate personas from golden tenant. If sample size is too small for stable rank metrics, run multiple synthesis repeats or generate multiple candidate personas per cluster.
- Score each persona twice: **free mode** (standard rubric, no reference) and **reference mode** (rubric + proxy reference as calibration anchor).
- Compare: score distributions (mean, std) per mode, variance reduction ratio, rank correlation between modes (Spearman if sample permits, otherwise simpler deltas/std-dev), anchoring detection (do reference-mode scores cluster around the anchor's quality?).

**Result: REJECT** — 44% variance reduction (std 0.548→0.408) but 100% anchoring bias (all reference-mode scores within 0.5 of declared anchor quality 4.5). Spearman rho=0.657 shows rank flattening. Mean shift -0.33 penalizes high-quality personas. Few-shot calibration (exp-5.13) remains the better approach.

**Deliverables:**
- `evals/reference_judging.py` (free vs reference prompt builders + comparison stats)
- `scripts/experiment_5_11.py` (runner)
- `output/experiments/exp-5.11-reference-vs-free-judging/results.json`
- `output/experiments/exp-5.11-reference-vs-free-judging/FINDINGS.md`
- `tests/test_exp_5_11.py` (optional)

---

## Execution order

**Recommended: 5.05 → 5.11 → 1.07 → 2.06**

Eval experiments (5.05, 5.11) first since they share the exp-5.13 base and don't touch synthesis. Then cross-module experiments (1.07, 2.06) on main.

**For each branch:**
1. Create branch using the exact commands above
2. Inspect the listed template branches (`git show origin/BRANCH:file`) to understand patterns
3. Implement all files
4. Run: `python scripts/experiment_X_XX.py`
5. Write FINDINGS.md from results
6. Run tests if created: `pytest tests/test_exp_X_XX.py -v`
7. `git add` and commit
8. `git push origin exp-X.XX-name`
9. `git switch main` before starting next branch

**If blocked by missing API key or model access:**
- Still commit all code and harness
- Note in FINDINGS.md: "harness ready, not yet run — missing: [specific thing]"
- Commit as: `"exp X.XX: description — harness ready"`

---

## Verification checklist

After all 4 branches are complete:
1. `git branch -r | grep -E "exp-(2.06|1.07|5.05|5.11)"` shows all 4 branches ✅
2. Each branch has: runner script, results.json, FINDINGS.md ✅
3. exp-5.05 and exp-5.11 branches include the real judges.py from exp-5.13 ✅
4. No shared module defaults were changed ✅
5. 5.11 does not claim any "gold-standard" persona — uses proxy reference only ✅

---

## Reference: guide and sources

- Lab guide: https://gardnermcintyre.com/wips/personas/lab-research.html#guide-1
- Repo README, synthesis/README.md, evaluation/README.md
- Batch 4 strategy: docs/plans/2026-04-10-batch4-research-strategy.md
