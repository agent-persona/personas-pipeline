from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return json.loads(path.read_text())


def load_persona_entries(
    path: Path,
    *,
    temperature: float | None = None,
    use_best_temperature: bool = True,
) -> tuple[dict, list[dict]]:
    artifact = load_json(path)

    if "per_persona" in artifact:
        return artifact, artifact["per_persona"]

    if artifact.get("experiment_id") == "2.06":
        selected_temperature = temperature
        if selected_temperature is None and use_best_temperature:
            selected_temperature = artifact.get("selection", {}).get("best_temperature")
        if selected_temperature is None:
            raise ValueError("Temperature sweep artifact requires --temperature or best_temperature")

        for result in artifact.get("results", []):
            if result.get("temperature") == selected_temperature:
                if result.get("status") == "failed":
                    raise ValueError(f"Temperature {selected_temperature} failed and has no personas")
                return artifact, result["per_persona"]

        raise ValueError(f"Temperature {selected_temperature} not found in {path}")

    raise ValueError(f"Unsupported artifact format: {path}")


def load_cluster_map() -> dict[str, ClusterData]:
    crawler_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = [ClusterData.model_validate(cluster_dict) for cluster_dict in cluster_dicts]
    return {cluster.cluster_id: cluster for cluster in clusters}


def load_record_map() -> dict[str, dict]:
    crawler_records = fetch_all(TENANT_ID)
    return {
        record.record_id: record.model_dump(mode="json")
        for record in crawler_records
    }


def build_reference_context(
    entry: dict,
    cluster_map: dict[str, ClusterData] | None = None,
    record_map: dict[str, dict] | None = None,
) -> dict:
    cluster_id = entry.get("cluster_id")
    if cluster_map is not None and cluster_id in cluster_map:
        cluster = cluster_map[cluster_id]
        return {
            "cluster_id": cluster.cluster_id,
            "tenant": cluster.tenant.model_dump(mode="json"),
            "summary": cluster.summary.model_dump(mode="json"),
            "sample_records": [record.model_dump(mode="json") for record in cluster.sample_records],
            "enrichment": cluster.enrichment.model_dump(mode="json"),
            "all_record_ids": cluster.all_record_ids,
        }

    persona = entry.get("persona", {})
    evidence = persona.get("source_evidence", [])
    record_ids: list[str] = []
    for item in evidence:
        record_ids.extend(item.get("record_ids", []))
    unique_record_ids = list(dict.fromkeys(record_ids))
    if record_map is None:
        record_map = load_record_map()

    sample_records = [
        record_map[record_id]
        for record_id in unique_record_ids
        if record_id in record_map
    ]
    if not sample_records:
        raise KeyError(f"Missing reference records for cluster_id={cluster_id}")

    return {
        "cluster_id": cluster_id,
        "tenant": {
            "tenant_id": TENANT_ID,
            "industry": TENANT_INDUSTRY,
            "product_description": TENANT_PRODUCT,
        },
        "summary": {
            "source_evidence_claims": [item.get("claim") for item in evidence],
        },
        "sample_records": sample_records,
        "enrichment": {},
        "all_record_ids": unique_record_ids,
    }
