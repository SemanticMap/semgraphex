"""Cross-stage contracts for reproducible ConceptNet preparation and M2 smoke runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_PREP_DATASET_FIELDS = (
    "source",
    "version",
    "language",
    "relations",
    "min_weight",
    "max_nodes",
    "component",
    "max_rows",
)
_PREP_GRAPH_FIELDS = ("directed", "self_loop_policy", "weight_transform")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_file_identity(path: str | Path) -> dict[str, object]:
    source = Path(path)
    stat = source.stat()
    return {
        "input_sha256": sha256_file(source),
        "input_size_bytes": int(stat.st_size),
    }


def _normalized(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value


def validate_prepared_graph_contract(prepared_graph_dir: str | Path, config: Any) -> dict[str, object]:
    """Reject a prepared graph whose scientific preparation differs from the run config.

    Local source paths are intentionally excluded from equality checks; the exact
    input bytes are instead bound by the preparation SHA-256 and byte size.
    """
    directory = Path(prepared_graph_dir)
    resolved_path = directory / "resolved_config.json"
    metadata_path = directory / "graph_metadata.json"
    completed = directory / "COMPLETED"
    if not completed.is_file() or not resolved_path.is_file() or not metadata_path.is_file():
        raise ValueError("prepared graph must contain COMPLETED, resolved_config.json, and graph_metadata.json")
    prepared = json.loads(resolved_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_identity = metadata.get("source_identity", {})
    if not isinstance(source_identity, dict):
        raise ValueError("prepared graph source_identity is missing or malformed")
    digest = source_identity.get("input_sha256")
    byte_size = source_identity.get("input_size_bytes")
    if not isinstance(digest, str) or len(digest) != 64 or not isinstance(byte_size, int) or byte_size <= 0:
        raise ValueError("prepared graph predates the reportable input fingerprint contract; re-run prepare")

    expected_dataset = config.resolved["dataset"]
    expected_graph = config.resolved["graph"]
    prepared_dataset = prepared.get("dataset", {})
    prepared_graph = prepared.get("graph", {})
    mismatches: list[str] = []
    for field in _PREP_DATASET_FIELDS:
        left = _normalized(prepared_dataset.get(field))
        right = _normalized(expected_dataset.get(field))
        if left != right:
            mismatches.append(f"dataset.{field}: prepared={left!r}, run={right!r}")
    for field in _PREP_GRAPH_FIELDS:
        left = _normalized(prepared_graph.get(field))
        right = _normalized(expected_graph.get(field))
        if left != right:
            mismatches.append(f"graph.{field}: prepared={left!r}, run={right!r}")
    if mismatches:
        raise ValueError("prepared graph/run scientific contract mismatch: " + "; ".join(mismatches))
    return {
        "prepared_graph_dir": str(directory),
        "input_sha256": digest,
        "input_size_bytes": byte_size,
        "dataset_source": prepared_dataset.get("source"),
        "dataset_version": prepared_dataset.get("version"),
        "language": prepared_dataset.get("language"),
        "relations": prepared_dataset.get("relations", []),
    }


def validate_start_run(run_dir: str | Path) -> dict[str, object]:
    """Validate the mandatory M2 start-smoke evidence bundle."""
    directory = Path(run_dir)
    if not (directory / "COMPLETED").is_file():
        raise ValueError("run is not complete")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    missing_stages = [name for name in ("spectral", "dynamics", "coarsening") if stages.get(name) != "completed"]
    if missing_stages:
        raise ValueError("start smoke missing completed stages: " + ", ".join(missing_stages))
    resolved = manifest.get("resolved_config", {})
    methods = resolved.get("coarsening", {}).get("methods", ["connectivity_matching", "unconstrained_matching"])
    method_summaries: dict[str, object] = {}
    for method in methods:
        metrics_path = directory / "coarsening" / method / "metrics.json"
        if not metrics_path.is_file():
            raise ValueError(f"start smoke missing {method} metrics.json")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("result", payload)
        fine = metrics.get("fine_node_count")
        coarse = metrics.get("coarse_node_count")
        reduction = metrics.get("achieved_reduction")
        if not isinstance(fine, int) or not isinstance(coarse, int) or fine <= coarse or not isinstance(reduction, (int, float)) or reduction <= 0:
            raise ValueError(f"{method} did not produce a valid contraction")
        evidence = (
            metrics.get("subspace_projection_distance"),
            metrics.get("slow_eigenvalue_max_abs_error"),
            metrics.get("mean_trajectory_relative_error"),
        )
        if all(value is None for value in evidence):
            raise ValueError(f"{method} produced no cross-scale distortion evidence")
        method_summaries[method] = {
            "fine_node_count": fine,
            "coarse_node_count": coarse,
            "achieved_reduction": float(reduction),
            "subspace_projection_distance": evidence[0],
            "slow_eigenvalue_max_abs_error": evidence[1],
            "mean_trajectory_relative_error": evidence[2],
        }
    return {
        "run_dir": str(directory),
        "status": "start_smoke_passed",
        "methods": method_summaries,
    }
