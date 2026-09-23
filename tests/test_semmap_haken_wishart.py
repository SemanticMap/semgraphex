from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from semmap_haken.wishart_cluster import wishart_cluster
from semmap_haken.wishart_metrics import (
    EgoCandidate,
    relation_js_neighbors,
    typed_wl_features,
)


def _layer(edges: list[tuple[int, int]], n: int = 3) -> sparse.csr_matrix:
    if not edges:
        return sparse.csr_matrix((n, n), dtype=float)
    rows, cols = zip(*edges)
    return sparse.csr_matrix(
        (np.ones(len(edges)), (rows, cols)), shape=(n, n), dtype=float
    )


def test_wishart_finds_two_separated_modes_without_predefined_cluster_count():
    points = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    nn = NearestNeighbors(n_neighbors=3).fit(points)
    distances, indices = nn.kneighbors(points)
    result = wishart_cluster(
        indices[:, 1:],
        distances[:, 1:],
        significance=0.2,
        min_cluster_size=2,
    )
    assert result.cluster_count == 2
    assert set(result.labels[:3]) == {result.labels[0]}
    assert set(result.labels[3:]) == {result.labels[3]}
    assert result.labels[0] != result.labels[3]


def test_relation_js_keeps_same_relation_profile_closest():
    adjacency = _layer([(0, 1), (0, 2)])
    a = EgoCandidate(0, np.arange(3), adjacency, {"IsA": adjacency})
    b = EgoCandidate(1, np.arange(3), adjacency, {"IsA": adjacency * 2})
    c = EgoCandidate(2, np.arange(3), adjacency, {"UsedFor": adjacency})
    neighbors = relation_js_neighbors((a, b, c), k=1)
    assert neighbors.indices[0, 0] == 1
    assert neighbors.distances[0, 0] == 0.0
    assert neighbors.distances[2, 0] > 0.0


def test_typed_wl_changes_when_relation_type_changes():
    adjacency = _layer([(0, 1), (1, 0), (0, 2), (2, 0)])
    a = EgoCandidate(0, np.arange(3), adjacency, {"IsA": adjacency})
    b = EgoCandidate(1, np.arange(3), adjacency, {"UsedFor": adjacency})
    features = typed_wl_features((a, b), iterations=2, dimension=128).toarray()
    assert not np.allclose(features[0], features[1])


def test_dynamic_snapshot_runs_on_small_sparse_graph():
    from semmap_haken.wishart_dynamics import compute_dynamic_snapshot

    adjacency = sparse.csr_matrix(
        np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], dtype=float)
    )
    snapshot, vectors = compute_dynamic_snapshot(
        adjacency,
        slow_modes=2,
        mfpt_pairs=2,
        mfpt_walks_per_pair=2,
        mfpt_max_steps=20,
        betweenness_samples=4,
        clustering_samples=4,
        distance_samples=4,
        seed=7,
    )
    assert snapshot.node_count == 4
    assert vectors.stationary_mass.shape == (4,)
    assert np.isclose(vectors.stationary_mass.sum(), 1.0)
    assert snapshot.mean_degree == 2.0


def test_partition_contracts_each_figure_instance_separately():
    from semmap_haken.wishart_hierarchy import (
        FigureOccurrence,
        _partition_from_occurrences,
    )

    occurrences = (
        FigureOccurrence(figure_type=3, candidate_index=0, center=0, nodes=(0, 1), kth_radius=0.1),
        FigureOccurrence(figure_type=3, candidate_index=4, center=4, nodes=(4, 5), kth_radius=0.1),
    )
    plan = _partition_from_occurrences(6, occurrences)
    assert plan.fine_to_coarse[0] == plan.fine_to_coarse[1]
    assert plan.fine_to_coarse[4] == plan.fine_to_coarse[5]
    assert plan.fine_to_coarse[0] != plan.fine_to_coarse[4]
    assert len(plan.figure_type_by_coarse) == 2
