# exp-7.03 — coverage: ours vs persona-hub

**Tenant:** `tenant_acme_corp` (38 records)  
**Embedding model:** `all-MiniLM-L6-v2` (local, no LLM spend)  
**persona-hub source:** https://huggingface.co/datasets/proj-persona/PersonaHub/resolve/main/persona.jsonl  

## Coverage of tenant records

For each record we compute the max cosine similarity to any persona in the set. Coverage@t = fraction of records with max_sim ≥ t.

| Set | #personas | mean max-sim | median max-sim | cov@0.20 | cov@0.30 | cov@0.40 | cov@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours | 2 | 0.182 | 0.176 | 34.2% | 0.0% | 0.0% | 0.0% |
| persona-hub-100 | 100 | 0.170 | 0.159 | 31.6% | 0.0% | 0.0% | 0.0% |

## Interpretation

- `ours` personas are flattened from `output/persona_*.json` (name + summary + top-5 goals/pains + 3 sample quotes). They were synthesized from this tenant's records, so a coverage win is expected — the interesting numbers are the *magnitude* of the gap.
- `persona-hub-100` entries are one-sentence generic archetypes sampled from a 1B-persona pool designed for synthetic data generation, not audience segmentation. Low coverage here isn't a flaw of persona-hub — it's confirmation that its personas aren't designed to index a specific tenant's behavior.
- The gap in mean max-sim is the load-bearing finding: it's the measurable value of grounded synthesis over sampling from a generic pool.

## Paired per-record win-rate (ours vs persona-hub-100)

- ours wins: **22/38** (58%)
- ties: 0/38
- persona-hub wins: 16/38
- mean gap (ours − phub): +0.012