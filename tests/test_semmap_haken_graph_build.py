from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from semmap_haken.conceptnet import stream_assertions
from semmap_haken.graph_build import build_sparse_graph, load_prepared_graph, save_prepared_graph, transform_weight


FIXTURE = Path(__file__).parent / "fixtures" / "conceptnet_tiny.tsv"


def test_transforms_and_undirected_parallel_aggregation() -> None:
    graph = build_sparse_graph(stream_assertions(FIXTURE), directed=False, weight_transform="raw", component="all", max_nodes=10)
    dog, pet = graph.node_ids.index("/c/en/dog/n/animal"), graph.node_ids.index("/c/en/pet")
    assert graph.adjacency[dog, pet] == graph.adjacency[pet, dog] == 2.0
    assert transform_weight(2.0, "binary") == 1.0
    assert transform_weight(2.0, "log1p") == pytest.approx(math.log(3.0))


def test_directed_lcc_selection_and_round_trip_are_deterministic(tmp_path: Path) -> None:
    records = list(stream_assertions(FIXTURE))
    first = build_sparse_graph(records, directed=True, weight_transform="raw", component="largest", max_nodes=2)
    second = build_sparse_graph(reversed(records), directed=True, weight_transform="raw", component="largest", max_nodes=2)
    assert first.node_ids == second.node_ids == ("/c/en/animal", "/c/en/dog")
    assert first.adjacency[first.node_ids.index("/c/en/animal"), first.node_ids.index("/c/en/dog")] == 0
    save_prepared_graph(first, tmp_path, metadata={"dataset": "fixture"}, resolved_config={"a": 1})
    restored = load_prepared_graph(tmp_path)
    assert restored.node_ids == first.node_ids
    assert np.array_equal(restored.adjacency.toarray(), first.adjacency.toarray())
