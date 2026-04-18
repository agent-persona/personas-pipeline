"""exp-7.04 intrinsic: segmentation quality WITHOUT any LLM call.

Clusters N users derived from tenant_acme_corp records under four
methods:
  1. ours        — greedy Jaccard on behavior/page/source sets
  2. bertopic    — sentence-transformers + UMAP + HDBSCAN + c-TF-IDF
  3. top2vec     — joint doc/topic embedding clustering (optional)
  4. kmeans-emb  — baseline: sentence-transformers + sklearn KMeans

Metrics:
  - coverage_pct          fraction of users assigned a non-noise label
  - n_clusters            distinct non-noise labels
  - stability_ari_mean    mean Adjusted Rand Index over 5 permuted reruns

Outputs:
  output/experiments/exp-7.04-oss-bench-segment/intrinsic_results.json
  output/experiments/exp-7.04-oss-bench-segment/intrinsic_FINDINGS.md
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import adjusted_rand_score

from crawler import fetch_all
from segmentation.models.record import RawRecord

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "experiments" / "exp-7.04-oss-bench-segment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]  # five permutations for stability ARI
TENANT = "tenant_acme_corp"


def _load_users() -> tuple[list[str], list[str], list[dict]]:
    """Aggregate records by user_id into (user_ids, documents, raw_records).

    `documents` is a single text blob per user — the natural input shape
    for the topic-modeling comparators.
    """
    records = fetch_all(TENANT)
    by_user: dict[str, list[RawRecord]] = defaultdict(list)
    raw_dicts = []
    for r in records:
        raw = RawRecord(**r.model_dump())
        if raw.user_id:
            by_user[raw.user_id].append(raw)
        raw_dicts.append(raw.model_dump())

    user_ids = sorted(by_user.keys())
    docs = []
    for uid in user_ids:
        blobs = []
        for r in by_user[uid]:
            blobs.append(f"source={r.source}")
            if r.behaviors:
                blobs.append("behaviors=" + ",".join(r.behaviors))
            if r.pages:
                blobs.append("pages=" + ",".join(r.pages))
        docs.append(" ".join(blobs))
    return user_ids, docs, raw_dicts


def _labels_from_our_segmenter(raw_records: list[dict]) -> dict[str, int]:
    """Return {user_id: cluster_int_label}. Users not placed get -1."""
    from segmentation.engine.clusterer import cluster_users
    from segmentation.engine.featurizer import featurize_records

    raws = [RawRecord(**r) for r in raw_records]
    features = featurize_records(raws)
    groups = cluster_users(features, threshold=0.15, min_cluster_size=2)

    label_map: dict[str, int] = {}
    for label, group in enumerate(groups):
        for uf in group:
            label_map[uf.user_id] = label
    # Any user not placed → -1
    for uf in features:
        if uf.user_id not in label_map:
            label_map[uf.user_id] = -1
    return label_map


def _labels_from_bertopic(user_ids: list[str], docs: list[str], seed: int) -> dict[str, int]:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Tiny-corpus friendly knobs: we have ~8 user docs.
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(
        n_neighbors=min(5, max(2, len(docs) - 1)),
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdb = HDBSCAN(min_cluster_size=2, metric="euclidean", prediction_data=True)
    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=umap_model,
        hdbscan_model=hdb,
        min_topic_size=2,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs)
    return {uid: int(t) for uid, t in zip(user_ids, topics)}


def _labels_from_top2vec(user_ids: list[str], docs: list[str], seed: int) -> dict[str, int]:
    # Top2Vec requires a corpus large enough for doc2vec; it errors out
    # on tiny inputs. We still try so the failure mode is recorded.
    from top2vec import Top2Vec

    model = Top2Vec(
        documents=docs,
        speed="fast-learn",
        workers=1,
        min_count=1,
        embedding_model="doc2vec",  # no tensorflow dep
    )
    doc_topics, _, _ = model.get_documents_topics(list(range(len(docs))))
    return {uid: int(t) for uid, t in zip(user_ids, doc_topics)}


def _labels_from_kmeans_emb(user_ids: list[str], docs: list[str], seed: int, k: int = 2) -> dict[str, int]:
    """Baseline: sentence embeddings + sklearn KMeans with k fixed."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = emb_model.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    km = KMeans(n_clusters=min(k, len(docs)), random_state=seed, n_init=10)
    labels = km.fit_predict(embs)
    return {uid: int(l) for uid, l in zip(user_ids, labels)}


def _coverage(labels: dict[str, int]) -> float:
    if not labels:
        return 0.0
    n_non_noise = sum(1 for v in labels.values() if v != -1)
    return n_non_noise / len(labels)


def _n_clusters(labels: dict[str, int]) -> int:
    return len({v for v in labels.values() if v != -1})


def _ari(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    return float(adjusted_rand_score([a[k] for k in keys], [b[k] for k in keys]))


@dataclass
class MethodResult:
    method: str
    coverage_pct: float
    n_clusters: int
    stability_ari_mean: float
    stability_ari_per_run: list[float]
    per_run_n_clusters: list[int]
    error: str | None = None


def run() -> None:
    user_ids, docs, raw_records = _load_users()
    print(f"[exp-7.04 intrinsic] n_users={len(user_ids)} n_records={len(raw_records)}")

    methods = {
        "ours-jaccard": lambda seed: _labels_from_our_segmenter(_shuffled(raw_records, seed)),
        "bertopic": lambda seed: _labels_from_bertopic(*_shuffled_users(user_ids, docs, seed), seed=seed),
        "top2vec": lambda seed: _labels_from_top2vec(*_shuffled_users(user_ids, docs, seed), seed=seed),
        "kmeans-emb": lambda seed: _labels_from_kmeans_emb(*_shuffled_users(user_ids, docs, seed), seed=seed),
    }

    results: list[MethodResult] = []
    for name, fn in methods.items():
        print(f"\n[exp-7.04 intrinsic] running {name}...")
        per_seed_labels: list[dict[str, int]] = []
        per_run_nc: list[int] = []
        err = None
        for seed in SEEDS:
            try:
                lbl = fn(seed)
                # normalize to consistent user-key basis
                lbl = {uid: lbl.get(uid, -1) for uid in user_ids}
                per_seed_labels.append(lbl)
                per_run_nc.append(_n_clusters(lbl))
                print(f"  seed={seed} coverage={_coverage(lbl):.2f} n_clusters={_n_clusters(lbl)}")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"  seed={seed} ERROR {err}")
                break
        if err:
            results.append(MethodResult(name, 0.0, 0, float("nan"), [], [], error=err))
            continue
        cov = _coverage(per_seed_labels[0])
        nc = _n_clusters(per_seed_labels[0])
        aris = [_ari(per_seed_labels[0], per_seed_labels[i]) for i in range(1, len(SEEDS))]
        aris_clean = [a for a in aris if not (a != a)]  # drop NaN
        ari_mean = float(sum(aris_clean) / len(aris_clean)) if aris_clean else float("nan")
        results.append(MethodResult(name, cov, nc, ari_mean, aris, per_run_nc))

    # Cross-method agreement on seed-0 labelings (ARI between each pair
    # of successful methods). Tells us whether the methods *agreed* on
    # which users go together, independent of cluster-count choices.
    seed0_labelings: dict[str, dict[str, int]] = {}
    for name, fn in methods.items():
        if any(r.method == name and r.error for r in results):
            continue
        try:
            lbl = fn(0)
            seed0_labelings[name] = {uid: lbl.get(uid, -1) for uid in user_ids}
        except Exception:
            pass
    cross = []
    names_ok = list(seed0_labelings.keys())
    for i, a in enumerate(names_ok):
        for b in names_ok[i + 1:]:
            cross.append({
                "pair": f"{a} vs {b}",
                "ari": _ari(seed0_labelings[a], seed0_labelings[b]),
            })

    out = {
        "experiment": "exp-7.04-oss-bench-segment-intrinsic",
        "tenant": TENANT,
        "n_users": len(user_ids),
        "n_records": len(raw_records),
        "seeds": SEEDS,
        "results": [asdict(r) for r in results],
        "cross_method_ari_seed0": cross,
    }
    (OUT_DIR / "intrinsic_results.json").write_text(json.dumps(out, indent=2))
    _write_findings(out)
    print(f"\n[exp-7.04 intrinsic] wrote {OUT_DIR / 'intrinsic_results.json'}")


def _shuffled(items: list, seed: int):
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def _shuffled_users(user_ids: list[str], docs: list[str], seed: int):
    rng = random.Random(seed)
    pairs = list(zip(user_ids, docs))
    rng.shuffle(pairs)
    uids, ds = zip(*pairs)
    return list(uids), list(ds)


def _write_findings(out: dict) -> None:
    md = [
        "# exp-7.04 intrinsic — segmentation head-to-head (no LLM)",
        "",
        f"**Tenant:** `{out['tenant']}`  ",
        f"**Input:** {out['n_users']} users / {out['n_records']} records  ",
        f"**Seeds:** {out['seeds']} (input order permuted per seed for stability ARI)  ",
        "",
        "## Results",
        "",
        "| Method | Coverage% | #clusters | Stability ARI (mean over 4 perms vs seed 0) | Error |",
        "|---|---:|---:|---:|---|",
    ]
    for r in out["results"]:
        cov = f"{r['coverage_pct']*100:.1f}%" if not r.get("error") else "—"
        nc = str(r["n_clusters"]) if not r.get("error") else "—"
        ari = f"{r['stability_ari_mean']:.3f}" if not r.get("error") and r['stability_ari_mean'] == r['stability_ari_mean'] else "—"
        err = r.get("error") or ""
        md.append(f"| {r['method']} | {cov} | {nc} | {ari} | {err} |")
    md.append("")
    md.append("## Cross-method agreement on seed-0 labelings")
    md.append("")
    md.append("ARI = 1.0 means two methods put every user in the same group.")
    md.append("")
    md.append("| Pair | ARI |")
    md.append("|---|---:|")
    for c in out.get("cross_method_ari_seed0", []):
        ari = c["ari"]
        ari_s = f"{ari:.3f}" if ari == ari else "NaN"
        md.append(f"| {c['pair']} | {ari_s} |")
    md.append("")
    md.append("## Honest caveats")
    md.append("")
    md.append(
        "- The golden tenant has only ~8 aggregated users. Topic-modeling "
        "methods (BERTopic, Top2Vec) are designed for corpora of hundreds+; "
        "small-corpus behavior is informative but not representative."
    )
    md.append(
        "- Stability ARI is intra-method (same method, permuted input). "
        "The cross-method section above is a separate measure — ARI between "
        "different methods' seed-0 assignments."
    )
    md.append("- No LLM was called in this benchmark. Cost: $0.")
    (OUT_DIR / "intrinsic_FINDINGS.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    run()
