from __future__ import annotations

import json
from pathlib import Path

import pytest

from semmap_haken.artifacts import (
    DATASET_METADATA_SCHEMA_VERSION,
    GRAPH_ARTIFACT_SCHEMA_VERSION,
    DatasetMetadata,
    GraphArtifactMetadata,
)
from semmap_haken.config import ConfigurationError, load_config
from semmap_haken.manifest import RunManifest


def test_load_config_resolves_relative_paths_against_config_location(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "smoke.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
paths:
  workspace_root: ..
  data_root: data
  cache_root: cache
  runs_root: runs
dataset:
  source: conceptnet-5.7
  language: en
  relations: [RelatedTo]
  min_weight: 1.0
  max_nodes: 1000
  component: largest
graph:
  directed: false
  weight_transform: log1p
  operator: normalized_adjacency
runtime:
  profile: smoke
  random_seed: 7
  resource_profile: resource_profiles/smoke.yaml
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.paths.workspace_root == tmp_path
    assert config.paths.data_root == tmp_path / "data"
    assert config.runtime.resource_profile == config_path.parent / "resource_profiles/smoke.yaml"
    assert config.resolved["dataset"]["language"] == "en"


def test_load_config_rejects_invalid_graph_transform(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
dataset:
  source: conceptnet-5.7
  language: en
  relations: [RelatedTo]
  min_weight: 1.0
  max_nodes: 10
  component: largest
graph:
  directed: false
  weight_transform: invalid
  operator: normalized_adjacency
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="weight_transform"):
        load_config(config_path)


def test_manifest_json_round_trip_preserves_colab_environment(tmp_path: Path) -> None:
    manifest = RunManifest.create(
        run_id="smoke-001",
        resolved_config={"dataset": {"source": "conceptnet-5.7"}},
        execution_environment="colab",
        notebook_environment={
            "runtime_class": "T4",
            "ram_gib": 15.0,
            "drive_cache_location": "datasets/conceptnet/5.7/checksum",
        },
        git_commit="abc123",
    )
    destination = tmp_path / "manifest.json"

    manifest.write_json(destination)
    restored = RunManifest.read_json(destination)

    assert restored == manifest
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 1


def test_artifact_metadata_contracts_expose_schema_versions() -> None:
    dataset = DatasetMetadata(
        dataset_id="conceptnet-5.7",
        source_url="https://example.invalid/conceptnet.csv.gz",
        checksum_sha256="a" * 64,
    )
    graph = GraphArtifactMetadata(
        dataset=dataset,
        node_count=2,
        edge_count=1,
        adjacency_format="csr",
        directed=False,
        weight_transform="log1p",
        operator_input="weighted_adjacency",
    )

    assert dataset.schema_version == DATASET_METADATA_SCHEMA_VERSION
    assert graph.schema_version == GRAPH_ARTIFACT_SCHEMA_VERSION
    assert graph.to_dict()["dataset"]["dataset_id"] == "conceptnet-5.7"
