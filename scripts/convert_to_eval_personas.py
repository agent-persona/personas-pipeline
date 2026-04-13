"""Convert PersonaV1 JSON outputs to persona_eval.Persona JSON.

Usage (from personas-pipeline/ root):
    python -m scripts.convert_to_eval_personas [--input-dir DIR] [--output-dir DIR]

Defaults:
    --input-dir:  output/
    --output-dir: output/eval_personas/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.persona import PersonaV1


def convert_file(src: Path, dst_dir: Path) -> Path:
    data = json.loads(src.read_text())
    persona = PersonaV1.model_validate(data["persona"])
    cluster_id = data.get("cluster_id") or src.stem
    eval_persona = persona_v1_to_eval(persona, persona_id=cluster_id)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{cluster_id}.json"
    dst.write_text(json.dumps(eval_persona.model_dump(), indent=2))
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_in = Path(__file__).resolve().parents[1] / "output"
    parser.add_argument("--input-dir", type=Path, default=default_in)
    parser.add_argument("--output-dir", type=Path, default=default_in / "eval_personas")
    args = parser.parse_args(argv)

    src_files = sorted(args.input_dir.glob("persona_*.json"))
    if not src_files:
        print(f"No persona_*.json files found in {args.input_dir}", file=sys.stderr)
        return 1

    for src in src_files:
        dst = convert_file(src, args.output_dir)
        try:
            shown = dst.relative_to(Path.cwd())
        except ValueError:
            shown = dst
        print(f"  {src.name} -> {shown}")

    print(f"\nConverted {len(src_files)} persona(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
