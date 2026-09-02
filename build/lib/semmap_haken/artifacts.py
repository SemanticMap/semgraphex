"""Versioned metadata-only contracts for foundation artifacts.

Sparse adjacency serialization deliberately belongs to the graph-builder workstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DATASET_METADATA_SCHEMA_VERSION = 1
GRAPH_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    source_url: str
    checksum_sha256: str
    expected_size_bytes: int | None = None
    actual_size_bytes: int | None = None
    cache_hit: bool | None = None
    schema_version: int = DATASET_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.checksum_sha256) != 64:
            raise ValueError("checksum_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphArtifactMetadata:
    dataset: DatasetMetadata
    node_count: int
    edge_count: int
    adjacency_format: str
    directed: bool
    weight_transform: str
    operator_input: str
    language: str = "en"
    relation_whitelist: tuple[str, ...] = ()
    component_policy: str = "largest"
    self_loop_policy: str = "exclude"
    parser_version: str | None = None
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    schema_version: int = GRAPH_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.node_count < 0 or self.edge_count < 0:
            raise ValueError("node_count and edge_count must be non-negative")
        if self.adjacency_format not in {"csr", "relation_aware_csr"}:
            raise ValueError("adjacency_format must describe a sparse payload")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relation_whitelist"] = list(self.relation_whitelist)
        return payload
