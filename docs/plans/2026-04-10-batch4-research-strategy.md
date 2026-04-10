# Batch 4 Research Strategy & Groundedness Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a final tier of independent variable experiments before stacking schema additions, and fix the structural groundedness ceiling that blocked 3 experiments across batches 1–3.

**Architecture:** Two parallel tracks — (A) independent experiments that extend batch 3 coverage with cleaner fixtures and metric fixes, (B) an infra sprint that adds a semantic groundedness proxy to `evaluation/metrics.py` so space 2/3 experiments get real signal.

**Tech Stack:** Python, existing personas-pipeline modules, git worktrees per experiment, Claude Code as LLM/judge (zero API cost).

---

## Context

### Experiment state (as of 2026-04-10)

| Batch | Experiments run | Strong adopts | PRs open |
|-------|----------------|---------------|----------|
| 1 | 4 (2.16, 2.17, 5.08, 6.09) | 3 | #1–4 |
| 2 | 9 (2.18–6.14) | 3 | #5–13 |
| 3 | 9 (1.12–6.17) | 8 | #14–22 |

Original 22-experiment queue is **fully exhausted**. Batch 4 requires new experiments designed from learnings.

### Why independent experiments first

Batch 3 added 4 schema fields independently (temporal layering, belief/values, cross-references, contradictions) and each showed strong isolated signal. Before promoting to v1.1 or testing combinations, we need:

1. **Cleaner re-runs** of experiments whose metrics were flawed (3.19, 4.19)
2. **Coverage of gaps** the original catalog didn't address (semantic grounding, metric fix validation)
3. **Generalization tests** for batch 3 findings before trusting them at scale

Combination experiments (stacking 2+ schema fields) come in **batch 5**, after independent signals are confirmed.

---

## Track A: Independent Experiments (Batch 4)

### Experiment A1: Counterfactual grounding swap
**Rationale:** Quantify how broken `check_groundedness()` is. Swap each persona's source_evidence citations with another persona's records and run the structural check. It should fail — if it passes, the checker is proven blind to content.

- **ID**: 3.24
- **Branch**: `exp-3.24-counterfactual-grounding`
- **Files**: `evals/counterfactual_grounding.py` (new)
- **Metric**: False-pass rate (% of swapped personas that still pass groundedness check)
- **Hypothesis**: Structural checker passes >80% of evidence-swapped personas — proving it cannot distinguish grounded from hallucinated citations
- **Expected outcome**: STRONG signal confirming ceiling, which justifies the Track B infra work

**Implementation:**
```python
# evals/counterfactual_grounding.py
def swap_evidence(persona_a: dict, persona_b: dict) -> dict:
    """Return persona_a with persona_b's record_ids injected into source_evidence."""
    ...

def run_counterfactual(persona_a, persona_b, cluster_a) -> dict:
    swapped = swap_evidence(persona_a, persona_b)
    report = check_groundedness(swapped, cluster_a)
    return {"passed": report.passed, "score": report.score, "expected_to_fail": True}
```

---

### Experiment A2: Recency weighting with real temporal fixture
**Rationale:** Experiment 3.19 concluded WEAK/DEFER because all cluster records shared synthetic same-day timestamps. The feature is implemented and correct — it needs real data.

- **ID**: 3.19b
- **Branch**: `exp-3.19b-recency-real-fixture`
- **Files**: `synthesis/fixtures/temporal_tenant/records.json` (new), `evals/recency_rerun.py`
- **Metric**: Freshness delta (weighted minus baseline) on fixture with 60-day temporal spread
- **Hypothesis**: Recency weighting causes a measurable shift in which traits dominate when records span a meaningful time window
- **Fixture design**: 12 records per cluster, timestamps spread over 60 days:
  - Months 1–2: Broad exploratory behaviors (feature discovery, onboarding pages)
  - Month 3: Specific technical behaviors (API config, webhook setup)
  - This ensures the HIGH WEIGHT records are meaningfully different from LOW WEIGHT

---

### Experiment A3: Multilingual coherence with semantic judge
**Rationale:** Experiment 4.19 concluded WEAK/DEFER because the substring-match metric was wrong — Spanish technical loanwords preserved perfectly but weren't counted. Re-run with an LLM judge rating in-language trait preservation.

- **ID**: 4.19b
- **Branch**: `exp-4.19b-multilingual-semantic-judge`
- **Files**: `evals/multilingual_semantic.py` (new)
- **Metric**: Judge-rated trait preservation (1–5 per language, averaged)
- **Hypothesis**: With a semantic judge, Spanish shows near-parity (loanwords preserve) but Mandarin shows meaningful degradation (semantic translation vs code-switching)
- **Method**: You are the judge. Read each multilingual twin response and score: does the persona's core professional identity come through in this language? (1=collapsed to generic LLM, 5=full persona preserved)

---

### Experiment A4: Off-label adversarial probes for contradictions
**Rationale:** Experiment 1.23 scored 1.00 adversarial coherence — but all probes were directly matched to a listed contradiction. This tests generalization: do contradictions help when the probe doesn't directly map to any named one?

- **ID**: 1.23b
- **Branch**: `exp-1.23b-off-label-adversarial`
- **Files**: `evals/off_label_probes.py` (new)
- **Metric**: Off-label coherence rate (probes not matched to any listed contradiction)
- **Hypothesis**: The benefit of named contradictions generalizes — even off-label probes are handled with more nuance because the persona has an established self-aware register
- **Method**: Design 3 probes per persona that target character traits NOT listed as contradictions. Score coherence vs a baseline persona (no contradictions field).

---

### Experiment A5: Curiosity throttle
**Rationale:** Experiment 4.21 showed question_rate_delta = 1.00 but flagged that the model treats "when natural" as "always." This risks mechanical question-appending in structured interview contexts.

- **ID**: 4.21b
- **Branch**: `exp-4.21b-curiosity-throttle`
- **Files**: `twin/twin/chat.py` (modify `curiosity_mode` instruction)
- **Metric**: Question rate in structured vs open contexts, and realism score in each
- **Hypothesis**: Adding "skip if the user is asking a direct closed question" to the curiosity instruction reduces mechanical appending without reducing realism in open contexts
- **Method**: Run 5-turn conversations where turns alternate: 2 open-ended, 3 closed/direct. Count question rate and score realism per turn type.

---

### Experiment A6: Semantic groundedness validation
**Rationale:** After Track B adds `semantic_groundedness_proxy()`, validate it actually catches what the structural checker misses. Compare structural scores vs semantic scores on the same personas.

- **ID**: 3.25
- **Branch**: `exp-3.25-semantic-groundedness-validation`
- **Files**: `evals/semantic_vs_structural.py` (new)
- **Metric**: Divergence rate (% of claims where semantic and structural scores disagree)
- **Hypothesis**: Structural checker passes >90% of claims that semantic proxy flags as weakly grounded — confirming the proxy adds real signal
- **Depends on**: Track B infra (semantic proxy must exist first)

---

## Track B: Infra Sprint — Semantic Groundedness Proxy

**Priority:** High. Unblocks re-runs of 2.20 (transcript-first) and 3.22 (domain rules) with real signal.

### Task B1: Add `semantic_groundedness_proxy()` to evaluation/metrics.py

**Files:**
- Modify: `evaluation/evaluation/metrics.py`
- Test: `tests/test_semantic_groundedness.py` (new)

**Implementation — vocabulary overlap approach (zero API cost):**

```python
STOPWORDS = {
    "the","a","an","and","or","of","in","for","to","with","that",
    "their","our","is","are","was","were","be","been","have","has",
    "this","it","its","at","by","from","on","as","not","but","so"
}

def _claim_tokens(text: str) -> set[str]:
    return {
        w.lower().strip(".,;:'\"()")
        for w in text.split()
        if len(w) > 3 and w.lower() not in STOPWORDS
    }

def semantic_groundedness_proxy(persona: dict, cluster: dict) -> dict:
    """
    Vocabulary-overlap semantic groundedness check.
    For each claim in goals/pains/motivations/objections, compute
    token overlap with its cited record payloads.

    Returns:
    {
      "semantic_score": float,       # mean overlap across all claim-evidence pairs
      "weak_pairs": list[dict],      # pairs below threshold (overlap < 0.1)
      "claim_count": int,
      "weak_count": int,
    }
    """
    records_by_id = {r["record_id"]: r for r in cluster.get("sample_records", [])}
    evidence_map = {
        e["field_path"]: e["record_ids"]
        for e in persona.get("source_evidence", [])
    }
    FIELDS = ["goals", "pains", "motivations", "objections"]
    pairs, weak = [], []

    for field in FIELDS:
        for i, item in enumerate(persona.get(field, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            field_path = f"{field}.{i}"
            record_ids = evidence_map.get(field_path, [])
            if not record_ids:
                continue
            claim_tokens = _claim_tokens(text)
            record_tokens = set()
            for rid in record_ids:
                rec = records_by_id.get(rid, {})
                payload = rec.get("payload", {})
                for v in payload.values():
                    record_tokens |= _claim_tokens(str(v))
            overlap = len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
            pairs.append(overlap)
            if overlap < 0.1:
                weak.append({"field_path": field_path, "claim": text[:80], "overlap": overlap})

    score = sum(pairs) / len(pairs) if pairs else 0.0
    return {
        "semantic_score": round(score, 4),
        "weak_pairs": weak,
        "claim_count": len(pairs),
        "weak_count": len(weak),
    }
```

**Steps:**

1. Write failing test:
```python
# tests/test_semantic_groundedness.py
def test_semantic_proxy_catches_unrelated_evidence():
    persona = {...}  # claim: "loves GraphQL"
    cluster = {...}  # record payload: {"behavior": "email_open"}
    result = semantic_groundedness_proxy(persona, cluster)
    assert result["weak_count"] > 0

def test_semantic_proxy_passes_matching_evidence():
    persona = {...}  # claim: "struggles with webhook configuration"
    cluster = {...}  # record payload: {"behavior": "webhook_config", "page": "/settings/webhooks"}
    result = semantic_groundedness_proxy(persona, cluster)
    assert result["semantic_score"] > 0.1
```

2. Run: `pytest tests/test_semantic_groundedness.py -v` → expect FAIL
3. Implement `semantic_groundedness_proxy()` in `evaluation/evaluation/metrics.py`
4. Run tests → expect PASS
5. Commit: `git add evaluation/ tests/ && git commit -m "feat: add semantic_groundedness_proxy() to evaluation metrics"`

---

### Task B2: Re-run exp-2.20 with semantic proxy

**Files:**
- `output/experiments/exp-2.20-reverse-engineered-persona/FINDINGS_v2.md` (new)

**Method:** Load the transcript-first personas already generated in exp-2.20 branch. Run both `check_groundedness()` (structural) and `semantic_groundedness_proxy()` (semantic) on them. Compare the two scores.

**Expected:** Transcript-first approach shows higher semantic_score than direct synthesis, even though both score 1.0 on structural. This validates the original 2.20 hypothesis with a real metric.

**Steps:**
1. Checkout: `git checkout exp-2.20-reverse-engineered-persona`
2. Run both checks on `output/transcript_first_persona_00.json` and `output/baseline_persona_00.json`
3. Record: `semantic_score_transcript_first` vs `semantic_score_baseline`
4. Write `FINDINGS_v2.md` with updated results
5. Commit and push update to PR #10

---

### Task B3: Re-run exp-3.22 with semantic proxy

**Files:**
- `output/experiments/exp-3.22-domain-specific-grounding-rules/FINDINGS_v2.md` (new)

**Method:** Run `semantic_groundedness_proxy()` on flagged claims from 3.22's domain rules engine. Compare: do flagged "sensitive" claims have lower semantic overlap than non-flagged claims? This validates whether the N+1 rule targets the right claims.

**Steps:**
1. Checkout: `git checkout exp-3.22-domain-specific-grounding-rules`
2. Run semantic proxy on both personas
3. Cross-reference: do the flagged claims (from `check_domain_rules()`) have lower semantic_score than the non-flagged ones?
4. Write `FINDINGS_v2.md`
5. Commit and push update to PR #8

---

## Sequence

```
Track B (infra) runs first or in parallel with early Track A experiments.
Track A6 (semantic validation) depends on Track B completion.

Week 1:
  B1 → implement semantic_groundedness_proxy + tests
  A1 → counterfactual grounding swap (no infra dependency)
  A2 → recency real fixture (no infra dependency)
  A3 → multilingual semantic judge (no infra dependency)

Week 2:
  B2, B3 → re-run 2.20 and 3.22 with new metric
  A4 → off-label adversarial probes (no infra dependency)
  A5 → curiosity throttle (no infra dependency)
  A6 → semantic groundedness validation (depends on B1)

After batch 4 completes:
  → Code review all batch 4 PRs
  → Batch 5: combination experiments (stack 2+ schema additions)
  → Schema v1.1 promotion from highest-signal combination
```

---

## Schema v1.1 Promotion Criteria

**Do NOT promote to v1.1 until:**
- [ ] All batch 4 independent experiments complete
- [ ] At least one combination experiment (batch 5) shows the fields compound positively
- [ ] Semantic groundedness proxy implemented and validating claims
- [ ] 1.14 (beliefs/values) re-tested with the proxy to confirm grounding holds with new fields

**v1.1 fields to add (pending combination confirmation):**
```python
class PersonaV1_1(PersonaV1):
    beliefs: list[str] = []           # exp-1.14
    values: list[str] = []            # exp-1.14
    contradictions: list[str] = []    # exp-1.23
    relates_to: list[dict] = []       # exp-1.16
    # temporal layering (exp-1.12) promoted separately as PersonaTemporal wrapper
```

**Promotion gate:** A combination experiment stacking any 2 of the above fields must show distinctiveness or depth score ≥ the sum of individual deltas (synergy check).

---

## Open PR Merge Order (recommended)

Merge in this order to minimize conflicts (all touch different modules):

1. **#3** exp-6.09 color palette view (segmentation only)
2. **#2** exp-5.08 adversarial detector (evaluation only)
3. **#5** exp-2.18 prompt prefix caching (synthesis/engine)
4. **#12** exp-6.14 persona graph relationships (evals only)
5. **#11** exp-4.12 twin-to-twin conversations (evals only)
6. **#1** exp-2.16 prompt compression (synthesis/engine)
7. **#14** exp-6.17 naming distinctiveness (evals only)
8. **#15** exp-4.21 curiosity behavior (twin/)
9. **#16** exp-1.12 temporal layering (synthesis/models)
10. **#18** exp-1.14 belief/value separation (synthesis/models)
11. **#17** exp-1.16 persona-to-persona references (synthesis/models)
12. **#21** exp-1.23 internalized contradictions (synthesis/models + twin/)
13. **#20** exp-3.23 predictive grounding (evals + fixtures)
14. **#22** exp-3.13 temporal grounding (synthesis/models)
15. **#6** exp-5.09 eval drift (evals only)

Defer merging until batch 4 complete (signal unclear or needs re-run):
- #7 exp-4.19 (4.19b re-run first)
- #8 exp-3.22 (B3 re-run first)
- #9 exp-5.19 (REJECT — do not merge)
- #10 exp-2.20 (B2 re-run first)
- #13 exp-5.23 (needs 10+ personas)
- #4 exp-2.17 (NOISE)
- #19 exp-3.19 (3.19b re-run first)
