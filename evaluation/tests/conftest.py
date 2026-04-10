from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "evaluation"
SYNTHESIS_ROOT = REPO_ROOT / "synthesis"

for candidate in (REPO_ROOT, EVALUATION_ROOT, SYNTHESIS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
