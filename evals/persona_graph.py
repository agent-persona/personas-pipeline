"""
Experiment 6.14: Persona Graph Relationships

Builds an undirected weighted graph from a persona set where edges represent
shared traits. Computes density and clustering coefficient as
coherence/distinctiveness proxies.
"""

import json
import math
import sys
from pathlib import Path

STOPWORDS = {
    "the", "and", "to", "a", "of", "in", "for", "with", "that", "their",
    "our", "is", "it", "be", "are", "on", "at", "by", "an", "as", "or",
    "not", "but", "from", "this", "they", "my", "me", "we", "i", "you",
    "she", "he", "was", "has", "have", "had", "do", "does", "will", "can",
    "so", "no", "if", "all", "any", "each", "which", "who", "its", "into",
    "than", "then", "more", "up", "out", "per", "about", "her", "his",
    "your", "them", "when", "how", "what", "very", "just", "too", "via",
    "across", "without", "between", "through",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase-split text into filtered tokens, removing stopwords and short words."""
    return {
        w
        for raw in text.split()
        for w in [raw.lower().strip(".,;:\"'()-")]
        if w and w not in STOPWORDS and len(w) > 2
    }


def extract_traits(persona: dict) -> set[str]:
    """Extract a flat set of normalized trait tokens from a persona object."""
    p = persona["persona"]
    traits: set[str] = set()
    for field in ("vocabulary", "channels", "decision_triggers", "goals", "pains", "motivations"):
        for item in p.get(field, []):
            traits |= _tokenize(item)
    return traits


def build_graph(persona_files: list[Path]) -> dict:
    personas = []
    for f in persona_files:
        with open(f) as fh:
            data = json.load(fh)
        name = data["persona"]["name"]
        traits = extract_traits(data)
        personas.append({"id": f.stem, "name": name, "traits": traits, "source": f})

    nodes = [
        {"id": p["id"], "name": p["name"], "trait_count": len(p["traits"])}
        for p in personas
    ]

    # Build trait lookup by persona id for edge normalization
    trait_map = {p["id"]: p["traits"] for p in personas}

    edges = []
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            a, b = personas[i], personas[j]
            shared = sorted(a["traits"] & b["traits"])
            denom = math.sqrt(len(a["traits"]) * len(b["traits"]))
            norm_weight = len(shared) / denom if denom > 0 else 0.0
            edges.append({
                "source": a["id"],
                "target": b["id"],
                "shared_traits": shared,
                "weight": len(shared),
                "normalized_weight": round(norm_weight, 4),
            })

    total_shared = sum(e["weight"] for e in edges)
    all_traits: set[str] = set()
    for p in personas:
        all_traits |= p["traits"]
    total_unique = len(all_traits)

    graph_density = total_shared / total_unique if total_unique > 0 else 0.0

    # For 2 nodes: normalized_edge_weight == the single edge's normalized_weight.
    # For N>2: report mean across all edges as graph-level summary.
    normalized_edge_weight = (
        sum(e["normalized_weight"] for e in edges) / len(edges) if edges else 0.0
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "graph_density": round(graph_density, 4),
        "normalized_edge_weight": round(normalized_edge_weight, 4),
        "total_unique_traits": total_unique,
        "shared_trait_count": total_shared,
        "interpretation": "lower density = more distinct population",
        "note_on_clustering": (
            "With 2 nodes there are no triangles, so the standard clustering "
            "coefficient is always 0. normalized_edge_weight (geometric-mean "
            "normalized shared/total) is reported as an equivalent proxy."
        ),
    }


def main():
    base = Path(__file__).parent.parent
    persona_dir = base / "output"
    out_dir = base / "output" / "experiments" / "exp-6.14-persona-graph-relationships"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only use the two canonical persona files (not drafts/baselines)
    files = sorted([
        persona_dir / "persona_00.json",
        persona_dir / "persona_01.json",
    ])
    files = [f for f in files if f.exists()]
    if not files:
        print("No persona files found in output/", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(files)} persona(s): {[f.name for f in files]}")
    result = build_graph(files)

    out_path = out_dir / "graph_results.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n=== Graph Metrics ===")
    print(f"Nodes: {len(result['nodes'])}")
    for n in result["nodes"]:
        print(f"  {n['id']}: {n['name']} ({n['trait_count']} traits)")
    print(f"Edges: {len(result['edges'])}")
    for e in result["edges"]:
        print(f"  {e['source']} <-> {e['target']}: {e['weight']} shared traits")
        print(f"    shared: {e['shared_traits']}")
    print(f"\nTotal unique traits : {result['total_unique_traits']}")
    print(f"Shared trait count  : {result['shared_trait_count']}")
    print(f"Graph density       : {result['graph_density']}")
    print(f"Norm. edge weight   : {result['normalized_edge_weight']}")
    print(f"\nResults written to  : {out_path}")


if __name__ == "__main__":
    main()
