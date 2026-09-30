"""FGW regression tests for the Colab experiment."""
import numpy as np
import pytest
from scipy import sparse

from semmap_haken.wishart_config import WishartOptions
from semmap_haken.wishart_metrics import (
    EgoCandidate, _transport_support, build_neighbor_graph,
)


def _candidate(center: int, edges: list[tuple[int, int]], size: int) -> EgoCandidate:
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    relation = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    return EgoCandidate(center, np.arange(size), adjacency, {"RelatedTo": relation})


def test_fgw_landmark_paths_can_cross_non_landmark_vertex():
    # Hubs 0 and 1 have degree 3; their only connecting path goes via node 2.
    candidate = _candidate(0, [(0, 2), (0, 3), (0, 4), (1, 2), (1, 5), (1, 6)], 7)
    kept, structure = _transport_support(candidate, rank=2)
    assert kept.tolist() == [0, 1]
    assert np.allclose(structure, [[0.0, 1.0], [1.0, 0.0]])


def test_fgw_config_is_bounded():
    options = WishartOptions.from_mapping({
        "metric": "fgw", "candidate_limit": 192,
        "transport_max_candidates": 192, "transport_rank": 16,
        "fgw_alpha": 0.5,
    })
    assert options.metric == "fgw"
    assert options.candidate_limit <= options.transport_max_candidates


def test_fgw_neighbor_graph_preserves_metadata_and_symmetry():
    pytest.importorskip("ot")
    candidates = [
        _candidate(0, [(0, 1), (1, 2)], 3),
        _candidate(1, [(0, 1), (1, 2)], 3),
        _candidate(2, [(0, 1), (0, 2)], 3),
    ]
    kwargs = dict(
        metric="fgw", k=1, wl_iterations=2, feature_dim=32,
        graphlet_size=3, graphlet_samples=16, transport_rank=3,
        transport_max_candidates=3, fgw_alpha=0.5,
        relation_js_block_size=32, seed=9, device="cpu",
    )
    result = build_neighbor_graph(candidates, **kwargs)
    assert result.indices.shape == (3, 1)
    assert np.all(np.isfinite(result.distances))
    assert np.all(result.distances >= 0)
    assert result.metadata["metric"] == "fgw"
    assert "relation frequency" in result.metadata["feature_source"]
    assert result.metadata["device"] == "cpu" if "device" in result.metadata else True


def test_fgw_rejects_candidate_overflow():
    pytest.importorskip("ot")
    candidates = [_candidate(i, [(0, 1)], 2) for i in range(3)]
    with pytest.raises(ValueError, match="transport_max_candidates"):
        build_neighbor_graph(
            candidates, metric="fgw", k=1, wl_iterations=2, feature_dim=32,
            graphlet_size=3, graphlet_samples=16, transport_rank=2,
            transport_max_candidates=2, fgw_alpha=0.5,
            relation_js_block_size=32, seed=9, device="cpu",
        )
