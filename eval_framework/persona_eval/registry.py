from __future__ import annotations
from typing import TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from persona_eval.scorer import BaseScorer

_SUITES: dict[str, list["BaseScorer"]] = {}
_LOCK = threading.Lock()


def register(suite: str, scorer: "BaseScorer") -> None:
    with _LOCK:
        _SUITES.setdefault(suite, []).append(scorer)


def get_suite(suite: str) -> list["BaseScorer"]:
    with _LOCK:
        if suite not in _SUITES:
            raise KeyError(f"Suite '{suite}' not found. Available: {list(_SUITES)}")
        return list(_SUITES[suite])


def list_suites() -> list[str]:
    with _LOCK:
        return list(_SUITES.keys())


def clear() -> None:
    with _LOCK:
        _SUITES.clear()
