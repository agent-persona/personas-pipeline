"""
Persona set visualization for experiment 6.09.

Computes a 2D projection of personas using bag-of-words cosine distance
and a simple stress-minimization layout. Renders as self-contained HTML.
No heavy ML dependencies required.
"""
from __future__ import annotations
import html as html_lib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _extract_text(persona: dict[str, Any]) -> str:
    """Flatten all string fields of a persona into one text blob."""
    parts = []
    for key, val in persona.items():
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if isinstance(v, str))
        elif isinstance(val, dict):
            parts.extend(str(v) for v in val.values() if isinstance(v, str))
    return " ".join(parts).lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text)


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}


def _cosine_distance(v1: dict, v2: dict) -> float:
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1 = math.sqrt(sum(x**2 for x in v1.values()))
    n2 = math.sqrt(sum(x**2 for x in v2.values()))
    if n1 == 0 or n2 == 0:
        return 1.0
    return max(0.0, 1.0 - (dot / (n1 * n2)))


def _mds_2d(dist_matrix: list[list[float]]) -> list[tuple[float, float]]:
    """Very simple MDS: place points to approximate distance matrix."""
    n = len(dist_matrix)
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        d = dist_matrix[0][1]
        return [(-d / 2, 0.0), (d / 2, 0.0)]
    # Gradient descent approximation for small n
    random.seed(42)
    coords = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]
    for _ in range(500):
        for i in range(n):
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                actual = math.sqrt(dx**2 + dy**2) + 1e-9
                target = dist_matrix[i][j]
                force = (actual - target) / actual
                fx -= force * dx * 0.01
                fy -= force * dy * 0.01
            coords[i] = (coords[i][0] + fx, coords[i][1] + fy)
    return coords


def _spread_score(coords: list[tuple[float, float]]) -> float:
    """Mean pairwise distance between all points (higher = more spread)."""
    if len(coords) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            total += math.sqrt(dx**2 + dy**2)
            count += 1
    return round(total / count, 4) if count else 0.0


COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
          "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def render_html(personas: list[dict], coords: list[tuple[float, float]],
                spread: float, output_path: str) -> None:
    if not coords:
        return
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xrange = (xmax - xmin) or 1
    yrange = (ymax - ymin) or 1

    def norm_x(x):
        return 60 + int(((x - xmin) / xrange) * 480)

    def norm_y(y):
        return 60 + int(((y - ymin) / yrange) * 380)

    dots = ""
    legend = ""
    for i, (p, (x, y)) in enumerate(zip(personas, coords)):
        color = COLORS[i % len(COLORS)]
        cx, cy = norm_x(x), norm_y(y)
        name = html_lib.escape(p.get("name", f"Persona {i}"))
        summary = html_lib.escape(p.get("summary", "")[:80])
        dots += f'<circle cx="{cx}" cy="{cy}" r="14" fill="{color}" opacity="0.85" />'
        dots += f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-size="9" fill="white" font-family="sans-serif">{i+1}</text>'
        legend += f'<div style="margin:4px 0"><span style="background:{color};display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px"></span><b>{i+1}. {name}</b> — {summary}</div>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Persona Set — 2D Projection</title></head>
<body style="font-family:sans-serif;padding:24px;background:#f9f9f9">
<h2>Persona Set — 2D Trait Projection</h2>
<p style="color:#666">Spread score: <b>{spread}</b> (higher = more distinct personas)</p>
<svg width="600" height="500" style="background:white;border:1px solid #ddd;border-radius:4px">
  <text x="300" y="30" text-anchor="middle" font-size="12" fill="#888" font-family="sans-serif">Trait Space Projection (cosine distance)</text>
  {dots}
</svg>
<div style="margin-top:16px;background:white;padding:16px;border-radius:4px;border:1px solid #ddd">
  <h3 style="margin-top:0">Legend</h3>{legend}
</div>
</body></html>"""

    Path(output_path).write_text(html)


def run_visualization(output_dir: str = "output", html_out: str = None) -> dict:
    output_path = Path(output_dir)
    persona_files = sorted(output_path.glob("persona_*.json"))
    if not persona_files:
        return {"error": "no persona files found"}

    personas = []
    for f in persona_files:
        data = json.loads(f.read_text())
        personas.append(data.get("persona", data))

    # Build TF-IDF
    all_tokens = [_tokenize(_extract_text(p)) for p in personas]
    doc_freq: Counter = Counter()
    for tokens in all_tokens:
        doc_freq.update(set(tokens))
    n = len(personas)
    idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}
    vectors = [_tfidf_vector(tokens, idf) for tokens in all_tokens]

    # Distance matrix
    dist = [[_cosine_distance(vectors[i], vectors[j]) for j in range(n)] for i in range(n)]

    # 2D coords + spread
    coords = _mds_2d(dist)
    spread = _spread_score(coords)

    # Render HTML
    if html_out is None:
        html_out = str(output_path / "persona_projection.html")
    render_html(personas, coords, spread, html_out)

    return {
        "personas_visualized": n,
        "spread_score": spread,
        "html_output": html_out,
        "distance_matrix": [[round(d, 3) for d in row] for row in dist],
        "coords_2d": [(round(x, 3), round(y, 3)) for x, y in coords],
    }


if __name__ == "__main__":
    result = run_visualization()
    print(json.dumps({k: v for k, v in result.items() if k != "distance_matrix"}, indent=2))
    print(f"\nVisualization saved to: {result.get('html_output')}")
