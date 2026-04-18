"""Per-experiment cost + token ledger. Persisted to JSONL so a crashed
run leaves an audit trail."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Haiku-4.5 pricing as of 2026-04 (USD per 1M tokens).
# Source: anthropic.com/pricing. Cached inputs ignored — we do not use caching.
HAIKU_INPUT_PER_M = 1.00
HAIKU_OUTPUT_PER_M = 5.00


def haiku_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * HAIKU_INPUT_PER_M + output_tokens * HAIKU_OUTPUT_PER_M) / 1_000_000


@dataclass
class LedgerEntry:
    experiment: str
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ts: float = field(default_factory=time.time)


class CostLedger:
    def __init__(self, experiment: str, out_dir: Path):
        self.experiment = experiment
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "ledger.jsonl"
        self.entries: list[LedgerEntry] = []

    def record(self, stage: str, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = haiku_cost(input_tokens, output_tokens) if "haiku" in model else 0.0
        entry = LedgerEntry(self.experiment, stage, model, input_tokens, output_tokens, cost)
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        return cost

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def total_by_stage(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.stage] = out.get(e.stage, 0.0) + e.cost_usd
        return out


# Global kill-switch. If total session spend exceeds this, raise.
SESSION_BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "15.0"))
_global_spend = 0.0


def register_spend(cost: float) -> None:
    global _global_spend
    _global_spend += cost
    if _global_spend > SESSION_BUDGET_USD:
        raise RuntimeError(
            f"Session budget exceeded: ${_global_spend:.4f} > ${SESSION_BUDGET_USD:.2f}. "
            f"Raise BENCH_BUDGET_USD env var to continue."
        )


def session_spend() -> float:
    return _global_spend
