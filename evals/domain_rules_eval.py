#!/usr/bin/env python3
"""Reproducible eval runner for exp-3.22 domain-specific grounding rules."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'synthesis'))
from synthesis.engine.domain_rules import check_domain_rules

results = {}
for i in ['00', '01']:
    persona = json.load(open(f'output/persona_{i}.json'))['persona']
    cluster = json.load(open(f'output/clusters/cluster_{i}.json'))
    results[f'cluster_{i}'] = check_domain_rules(persona, cluster)
    print(f'=== cluster_{i} ===')
    print(json.dumps(results[f'cluster_{i}'], indent=2))

json.dump(results, open('output/experiments/exp-3.22-domain-specific-grounding-rules/eval_results.json', 'w'), indent=2)
print('\nResults saved to output/experiments/exp-3.22-domain-specific-grounding-rules/eval_results.json')
