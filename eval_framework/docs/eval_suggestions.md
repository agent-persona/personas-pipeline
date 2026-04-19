  ┌─────────────────────────────┬─────────────────────────────────────┐
  │           Metric            │               Method                │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Persona identification      │ Blind matching task                 │
  │ accuracy                    │                                     │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Time-to-identify            │ Stopwatch on evaluator              │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Consistency score           │ Contradiction detection across      │
  │                             │ turns                               │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Drift rate                  │ Consistency score over turn count   │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Adversarial robustness      │ # of red-team prompts before break  │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Human-likeness              │ Binary classifier by evaluators     │
  │ (Turing-style)              │                                     │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Style fidelity              │ Cosine similarity of stylometric    │
  │                             │ vectors                             │
  ├─────────────────────────────┼─────────────────────────────────────┤
  │ Cross-platform coherence    │ Same evaluator rates persona on 2+  │
  │                             │ channels                            │
  └─────────────────────────────┴─────────────────────────────────────┘