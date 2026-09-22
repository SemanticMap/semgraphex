from __future__ import annotations

import numpy as np
from scipy import sparse

from semmap_haken.coarsen import build_partition


def _star(size: int) -> sparse.csr_matrix:
    leaves = np.arange(1, size, dtype=np.int64)
    rows = np.r_[np.zeros(size - 1, dtype=np.int64), leaves]
    cols = np.r_[leaves, np.zeros(size - 1, dtype=np.int64)]
    return sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(size, size))


def _is_connected_cluster(adjacency: sparse.csr_matrix, cluster: tuple[int, ...]) -> bool:
    if len(cluster) <= 1:
        return True
    allowed = set(cluster)
    seen = {cluster[0]}
    stack = [cluster[0]]
    while stack:
        node = stack.pop()
        neighbours = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        for raw in neighbours:
            neighbour = int(raw)
            if neighbour in allowed and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == allowed


def test_agglomerative_escapes_disjoint_pair_ceiling_on_star() -> None:
    adjacency = _star(10)
    coordinates = np.arange(10, dtype=np.float64)[:, None]

    pairwise = build_partition(
        adjacency, coordinates, method="connectivity_matching", target_reduction=0.5, seed=1729
    )
    agglomerative = build_partition(
        adjacency, coordinates, method="connectivity_agglomerative", target_reduction=0.5, seed=1729
    )

    assert pairwise.achieved_merges == 1
    assert pairwise.shortfall_reason == "insufficient_disjoint_adjacent_pairs"
    assert agglomerative.requested_merges == agglomerative.achieved_merges == 5
    assert agglomerative.coarse_node_count == 5
    assert np.isclose(agglomerative.achieved_reduction, 0.5)
    assert agglomerative.shortfall_reason is None
    assert max(map(len, agglomerative.clusters)) > 2
    assert all(_is_connected_cluster(adjacency, cluster) for cluster in agglomerative.clusters)


def test_agglomerative_is_deterministic_and_complete() -> None:
    adjacency = _star(12)
    coordinates = np.column_stack((np.arange(12, dtype=np.float64), np.zeros(12)))
    first = build_partition(
        adjacency, coordinates, method="connectivity_agglomerative", target_reduction=0.35, seed=7
    )
    second = build_partition(
        adjacency, coordinates, method="connectivity_agglomerative", target_reduction=0.35, seed=7
    )
    assert np.array_equal(first.fine_to_coarse, second.fine_to_coarse)
    assert first.clusters == second.clusters
    assert sorted(node for cluster in first.clusters for node in cluster) == list(range(12))
    assert first.achieved_merges == int(np.floor(12 * 0.35))
    assert all(_is_connected_cluster(adjacency, cluster) for cluster in first.clusters)
