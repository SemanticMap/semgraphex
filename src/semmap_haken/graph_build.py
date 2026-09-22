"""Deterministic, sparse-only ConceptNet graph preparation and persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .conceptnet import Assertion


@dataclass(frozen=True)
class PreparedGraph:
    adjacency: sparse.csr_matrix
    node_ids: tuple[str, ...]
    report: dict[str, object]
    selected_edges: tuple[dict[str, object], ...] = ()


def transform_weight(weight: float, transform: str) -> float:
    if transform == "binary":
        return 1.0
    if transform == "raw":
        return weight
    if transform == "log1p":
        return math.log1p(weight)
    raise ValueError(f"unsupported transform for this preparation stage: {transform}")


def _select_nodes(adjacency: sparse.csr_matrix, node_ids: list[str], *, component: str, max_nodes: int) -> tuple[sparse.csr_matrix, list[str]]:
    selected = np.arange(adjacency.shape[0])
    if component == "largest" and adjacency.shape[0]:
        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        sizes = np.bincount(labels)
        largest = np.flatnonzero(sizes == sizes.max())
        chosen_label = min(largest, key=lambda label: min(node_ids[index] for index in np.flatnonzero(labels == label)))
        selected = np.flatnonzero(labels == chosen_label)
    selected = sorted(selected, key=lambda index: node_ids[index])[:max_nodes]
    return adjacency[selected][:, selected].tocsr(), [node_ids[index] for index in selected]


def build_sparse_graph(assertions: Iterable[Assertion], *, directed: bool, weight_transform: str, component: str, max_nodes: int, self_loop_policy: str = "exclude") -> PreparedGraph:
    """Build a URI-lexicographically indexed CSR matrix without dense materialization."""
    # Parsing remains streamed; this aggregate map is the explicit M0 memory cost.
    if self_loop_policy not in {"exclude", "include"}:
        raise ValueError("self_loop_policy must be 'exclude' or 'include'")
    records = [record for record in assertions if self_loop_policy == "include" or record.start_uri != record.end_uri]
    node_ids = sorted({uri for record in records for uri in (record.start_uri, record.end_uri)})
    index = {uri: position for position, uri in enumerate(node_ids)}
    weights: dict[tuple[int, int], float] = {}
    relation_counts: Counter[str] = Counter()
    for record in records:
        value = transform_weight(record.weight, weight_transform)
        start, end = index[record.start_uri], index[record.end_uri]
        weights[(start, end)] = weights.get((start, end), 0.0) + value
        if not directed and start != end:
            weights[(end, start)] = weights.get((end, start), 0.0) + value
        relation_counts[record.relation_name] += 1
    rows, columns, values = zip(*((row, column, value) for (row, column), value in sorted(weights.items()))) if weights else ((), (), ())
    adjacency = sparse.csr_matrix((values, (rows, columns)), shape=(len(node_ids), len(node_ids)), dtype=np.float64)
    adjacency, node_ids = _select_nodes(adjacency, node_ids, component=component, max_nodes=max_nodes)
    selected = set(node_ids)
    selected_records = [record for record in records if record.start_uri in selected and record.end_uri in selected]
    selected_counts = Counter(record.relation_name for record in selected_records)
    loops = int(adjacency.diagonal().astype(bool).sum())
    edge_count = int(adjacency.nnz if directed else (adjacency.nnz - loops) // 2 + loops)
    edge_rows: dict[tuple[object, ...], dict[str, object]] = {}
    for record in selected_records:
        key = (record.start_uri, record.end_uri, record.relation_uri, record.dataset, json.dumps(record.sources, sort_keys=True), record.license)
        row = edge_rows.setdefault(key, {"start_uri": record.start_uri, "end_uri": record.end_uri, "relation": record.relation_name, "relation_uri": record.relation_uri, "dataset": record.dataset, "sources": record.sources, "license": record.license, "assertion_count": 0, "aggregate_contribution": 0.0})
        row["assertion_count"] = int(row["assertion_count"]) + 1
        row["aggregate_contribution"] = float(row["aggregate_contribution"]) + transform_weight(record.weight, weight_transform)
    report = {"node_count": adjacency.shape[0], "edge_count": edge_count, "adjacency_nnz": int(adjacency.nnz), "self_loop_count": loops, "self_loop_policy": self_loop_policy, "relation_histogram": dict(sorted(selected_counts.items())), "selection_policy": "lexicographic URI order after optional deterministic LCC; retain first max_nodes induced nodes"}
    return PreparedGraph(adjacency, tuple(node_ids), report, tuple(edge_rows[key] for key in sorted(edge_rows)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_prepared_graph(graph: PreparedGraph, directory: str | Path, *, metadata: dict[str, object], resolved_config: dict[str, object]) -> dict[str, Path]:
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    adjacency_path, nodes_path = staging / "adjacency.npz", staging / "nodes.json"
    metadata_path, config_path, edges_path = staging / "graph_metadata.json", staging / "resolved_config.json", staging / "selected_edges.jsonl"
    sparse.save_npz(adjacency_path, graph.adjacency)
    nodes_path.write_text(json.dumps(list(graph.node_ids), indent=2) + "\n", encoding="utf-8")
    edges_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in graph.selected_edges), encoding="utf-8")
    payload = {**metadata, "data_report": graph.report, "checksums": {"adjacency.npz": _sha256(adjacency_path), "nodes.json": _sha256(nodes_path), "selected_edges.jsonl": _sha256(edges_path)}}
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backup = destination.with_name(f".{destination.name}.previous")
    backup_exists = False
    if destination.exists():
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(destination, backup)
        backup_exists = True
    try:
        os.replace(staging, destination)
    except Exception:
        if backup_exists:
            os.replace(backup, destination)
        raise
    if backup_exists:
        shutil.rmtree(backup)
    (destination / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return {"adjacency": destination / "adjacency.npz", "nodes": destination / "nodes.json", "metadata": destination / "graph_metadata.json", "resolved_config": destination / "resolved_config.json", "selected_edges": destination / "selected_edges.jsonl"}


def load_prepared_graph(directory: str | Path) -> PreparedGraph:
    source = Path(directory)
    if not (source / "COMPLETED").is_file():
        raise ValueError("prepared graph is missing completion marker")
    adjacency = sparse.load_npz(source / "adjacency.npz").tocsr()
    node_ids = tuple(json.loads((source / "nodes.json").read_text(encoding="utf-8")))
    metadata = json.loads((source / "graph_metadata.json").read_text(encoding="utf-8"))
    if adjacency.shape != (len(node_ids), len(node_ids)):
        raise ValueError("prepared graph adjacency and node mapping disagree")
    for name, digest in metadata.get("checksums", {}).items():
        if _sha256(source / name) != digest:
            raise ValueError(f"prepared graph checksum mismatch: {name}")
    edges = tuple(json.loads(line) for line in (source / "selected_edges.jsonl").read_text(encoding="utf-8").splitlines())
    return PreparedGraph(adjacency, node_ids, metadata["data_report"], edges)
