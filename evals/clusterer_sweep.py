"""Experiment 6.03: Clusterer parameter sweep."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev

from segmentation.engine.clusterer import cluster_users, jaccard_similarity
from segmentation.engine.featurizer import featurize_records
from segmentation.engine.summarizer import build_cluster_data
from segmentation.models.record import RawRecord

THRESHOLDS = (0.1, 0.2, 0.4, 0.6, 0.8)
MIN_CLUSTER_SIZES = (1, 2, 3, 5)
SYNTHESIS_THRESHOLDS = THRESHOLDS
SYNTHESIS_MIN_CLUSTER_SIZE = 2


@dataclass
class ClusterConfigResult:
    threshold: float
    min_cluster_size: int
    n_features: int
    n_clusters: int
    n_noise: int
    noise_rate: float
    cluster_sizes: list[int]
    mean_cluster_size: float
    size_stdev: float
    compactness: float
    balance_score: float


@dataclass
class SynthesizedConfigResult:
    threshold: float
    min_cluster_size: int
    selected_cluster_size: int
    persona_name: str
    judge_overall: float
    judge_dimensions: dict[str, float]
    compactness: float
    cluster_count: int
    noise_rate: float


@dataclass
class ClusterSweepReport:
    rows: list[ClusterConfigResult] = field(default_factory=list)
    synthesis_rows: list[SynthesizedConfigResult] = field(default_factory=list)
    best_by_compactness: ClusterConfigResult | None = None
    best_by_cluster_count: ClusterConfigResult | None = None
    knee_candidates: list[ClusterConfigResult] = field(default_factory=list)


def _compactness_for_cluster(cluster: list, feature_map: dict[str, object]) -> float:
    centroid = set().union(*(member.behaviors for member in cluster))
    sims = [jaccard_similarity(member.behaviors, centroid) for member in cluster]
    return mean(sims) if sims else 0.0


def _balance_score(cluster_sizes: list[int]) -> float:
    if not cluster_sizes:
        return 0.0
    if len(cluster_sizes) == 1:
        return 1.0
    avg = mean(cluster_sizes)
    if avg == 0:
        return 0.0
    spread = pstdev(cluster_sizes) if len(cluster_sizes) > 1 else 0.0
    return max(0.0, 1.0 - (spread / avg))


def compute_cluster_config(records: list[RawRecord], threshold: float, min_cluster_size: int) -> ClusterConfigResult:
    features = featurize_records(records)
    clusters = cluster_users(features, threshold=threshold, min_cluster_size=min_cluster_size)
    cluster_sizes = [len(cluster) for cluster in clusters]
    n_features = len(features)
    n_clusters = len(clusters)
    n_noise = max(0, n_features - sum(cluster_sizes))
    noise_rate = n_noise / n_features if n_features else 0.0
    compactness = mean(
        _compactness_for_cluster(cluster, {})
        for cluster in clusters
    ) if clusters else 0.0
    return ClusterConfigResult(
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        n_features=n_features,
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_rate=noise_rate,
        cluster_sizes=sorted(cluster_sizes, reverse=True),
        mean_cluster_size=mean(cluster_sizes) if cluster_sizes else 0.0,
        size_stdev=pstdev(cluster_sizes) if len(cluster_sizes) > 1 else 0.0,
        compactness=compactness,
        balance_score=_balance_score(cluster_sizes),
    )


def run_cluster_sweep(records: list[RawRecord]) -> ClusterSweepReport:
    report = ClusterSweepReport()
    for threshold in THRESHOLDS:
        for min_cluster_size in MIN_CLUSTER_SIZES:
            report.rows.append(
                compute_cluster_config(
                    records=records,
                    threshold=threshold,
                    min_cluster_size=min_cluster_size,
                )
            )

    report.best_by_compactness = max(
        report.rows,
        key=lambda row: (row.compactness, -row.noise_rate, row.n_clusters),
        default=None,
    )
    report.best_by_cluster_count = max(
        report.rows,
        key=lambda row: (row.n_clusters, -row.noise_rate, row.compactness),
        default=None,
    )
    report.knee_candidates = sorted(
        (row for row in report.rows if row.min_cluster_size == SYNTHESIS_MIN_CLUSTER_SIZE),
        key=lambda row: row.threshold,
    )
    return report


def pick_synthesis_configs(report: ClusterSweepReport, limit: int = 5) -> list[ClusterConfigResult]:
    selected = [row for row in report.rows if row.min_cluster_size == SYNTHESIS_MIN_CLUSTER_SIZE]
    selected = sorted(selected, key=lambda row: row.threshold)
    return selected[:limit]


def build_cluster_summaries(
    records: list[RawRecord],
    threshold: float,
    min_cluster_size: int,
    tenant_industry: str | None,
    tenant_product: str | None,
) -> list[dict]:
    return [
        build_cluster_data(
            cluster_users=cluster,
            all_records=records,
            tenant_id=records[0].tenant_id if records else "unknown",
            tenant_industry=tenant_industry,
            tenant_product=tenant_product,
            existing_persona_names=[],
        )
        for cluster in cluster_users(
            featurize_records(records),
            threshold=threshold,
            min_cluster_size=min_cluster_size,
        )
    ]


def report_to_dict(report: ClusterSweepReport) -> dict:
    return {
        "rows": [asdict(row) for row in report.rows],
        "synthesis_rows": [asdict(row) for row in report.synthesis_rows],
        "best_by_compactness": asdict(report.best_by_compactness) if report.best_by_compactness else None,
        "best_by_cluster_count": asdict(report.best_by_cluster_count) if report.best_by_cluster_count else None,
        "knee_candidates": [asdict(row) for row in report.knee_candidates],
    }


def format_report(report: ClusterSweepReport) -> str:
    lines = []
    lines.append("=== GRID ===")
    lines.append(f"{'threshold':>10} {'min':>4} {'clusters':>8} {'noise':>6} {'compact':>8} {'balance':>8} sizes")
    for row in report.rows:
        lines.append(
            f"{row.threshold:>10.1f} {row.min_cluster_size:>4d} {row.n_clusters:>8d}"
            f" {row.noise_rate:>6.2f} {row.compactness:>8.2f} {row.balance_score:>8.2f}"
            f" {row.cluster_sizes}"
        )
    if report.best_by_compactness:
        lines.append("")
        lines.append(
            f"Best compactness: t={report.best_by_compactness.threshold:.1f} "
            f"m={report.best_by_compactness.min_cluster_size} "
            f"compact={report.best_by_compactness.compactness:.2f}"
        )
    if report.best_by_cluster_count:
        lines.append(
            f"Best cluster count: t={report.best_by_cluster_count.threshold:.1f} "
            f"m={report.best_by_cluster_count.min_cluster_size} "
            f"clusters={report.best_by_cluster_count.n_clusters}"
        )
    return "\n".join(lines)
