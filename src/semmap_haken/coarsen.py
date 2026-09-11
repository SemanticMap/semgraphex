"""Deterministic, sparse-aware graph coarsening contracts."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

Method = Literal["connectivity_matching", "unconstrained_matching", "connectivity_agglomerative"]


@dataclass(frozen=True)
class PartitionResult:
    fine_to_coarse: np.ndarray
    clusters: tuple[tuple[int, ...], ...]
    merges: tuple[tuple[int, int, float], ...]
    method: Method
    requested_merges: int
    achieved_merges: int
    requested_reduction: float
    achieved_reduction: float
    coarse_node_count: int
    shortfall_reason: str | None
    seed: int
    tie_breaking: str
    task_identity: str


def _distance(coordinates: np.ndarray, left: int, right: int) -> float:
    return float(np.linalg.norm(coordinates[left] - coordinates[right]))


def _requested_merges(node_count: int, reduction: float) -> int:
    if not 0 < reduction <= 0.5:
        raise ValueError("target_reduction must be in (0, 0.5]")
    return min(node_count // 2, int(np.floor(node_count * reduction + 1e-12)))


def _partition_from_clusters(
    node_count: int,
    clusters: list[tuple[int, ...]],
    merge_log: list[tuple[int, int, float]],
    *,
    method: Method,
    reduction: float,
    seed: int,
    shortfall: str | None,
) -> PartitionResult:
    clusters = sorted((tuple(sorted(cluster)) for cluster in clusters), key=lambda cluster: cluster[0])
    assignment = np.empty(node_count, dtype=np.int64)
    for coarse, cluster in enumerate(clusters):
        assignment[list(cluster)] = coarse
    requested = _requested_merges(node_count, reduction)
    achieved = node_count - len(clusters)
    return PartitionResult(
        fine_to_coarse=assignment,
        clusters=tuple(clusters),
        merges=tuple(merge_log),
        method=method,
        requested_merges=requested,
        achieved_merges=achieved,
        requested_reduction=reduction,
        achieved_reduction=achieved / node_count if node_count else 0.0,
        coarse_node_count=len(clusters),
        shortfall_reason=shortfall,
        seed=seed,
        tie_breaking="distance_then_node_index",
        task_identity=f"m2:{method}:seed={seed}",
    )


def _finish(node_count: int, pairs: list[tuple[int, int, float]], *, method: Method, reduction: float, seed: int, shortfall: str | None) -> PartitionResult:
    claimed = {node for left, right, _ in pairs for node in (left, right)}
    clusters = [tuple(sorted((left, right))) for left, right, _ in pairs]
    clusters.extend((node,) for node in range(node_count) if node not in claimed)
    return _partition_from_clusters(
        node_count, clusters, pairs, method=method, reduction=reduction, seed=seed, shortfall=shortfall
    )


def _connectivity_candidates(adjacency: sparse.csr_matrix, coordinates: np.ndarray) -> list[tuple[float, int, int]]:
    upper = sparse.triu(adjacency, k=1, format="coo")
    return sorted((_distance(coordinates, int(left), int(right)), int(left), int(right)) for left, right in zip(upper.row, upper.col, strict=True))


def _unconstrained_candidates(coordinates: np.ndarray) -> list[tuple[float, int, int]]:
    """Return O(N) nearest-neighbour candidates, not an N-by-N distance matrix."""
    count = coordinates.shape[0]
    if count < 2:
        return []
    tree = cKDTree(coordinates)
    neighbour_count = min(8, count)
    distances, indices = tree.query(coordinates, k=neighbour_count)
    candidates = {
        (float(distances[row, column]), min(row, int(indices[row, column])), max(row, int(indices[row, column])))
        for row in range(count)
        for column in range(1, neighbour_count)
        if int(indices[row, column]) != row
    }
    return sorted(candidates)


def _find(parent: np.ndarray, node: int) -> int:
    root = int(node)
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[node]) != node:
        nxt = int(parent[node])
        parent[node] = root
        node = nxt
    return root


def _connectivity_agglomerative(
    adjacency: sparse.csr_matrix,
    coordinates: np.ndarray,
    *,
    target_reduction: float,
    seed: int,
    distance_threshold: float | None,
) -> PartitionResult:
    """Merge adjacent clusters by centroid distance until the requested reduction.

    Each node starts as its own cluster.  Candidate merges exist only between
    clusters that share at least one graph edge.  Cluster coordinates are the
    size-weighted mean of member Haken coordinates.  A priority queue selects
    the smallest current centroid distance; stale entries are discarded using
    per-cluster generation counters.  Because a previously merged cluster may
    merge again in the same coarsening step, this method is not limited by the
    disjoint-pair ceiling of ``connectivity_matching``.
    """
    node_count = adjacency.shape[0]
    requested = _requested_merges(node_count, target_reduction)
    if requested == 0 or node_count < 2:
        return _partition_from_clusters(
            node_count,
            [(node,) for node in range(node_count)],
            [],
            method="connectivity_agglomerative",
            reduction=target_reduction,
            seed=seed,
            shortfall=None,
        )

    points = np.asarray(coordinates, dtype=np.float64)
    parent = np.arange(node_count, dtype=np.int64)
    active = np.ones(node_count, dtype=bool)
    sizes = np.ones(node_count, dtype=np.int64)
    centroids = points.copy()
    generation = np.zeros(node_count, dtype=np.int64)

    neighbours: list[set[int]] = []
    for node in range(node_count):
        row = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        neighbours.append({int(item) for item in row if int(item) != node})

    upper = sparse.triu(adjacency, k=1, format="coo")
    heap: list[tuple[float, int, int, int, int]] = []
    for raw_left, raw_right in zip(upper.row, upper.col, strict=True):
        left, right = int(raw_left), int(raw_right)
        distance = float(np.linalg.norm(centroids[left] - centroids[right]))
        heap.append((distance, left, right, 0, 0))
    heapq.heapify(heap)

    merge_log: list[tuple[int, int, float]] = []
    achieved = 0
    threshold_blocked = False

    while achieved < requested and heap:
        distance, left, right, left_gen, right_gen = heapq.heappop(heap)
        if not active[left] or not active[right]:
            continue
        if int(generation[left]) != left_gen or int(generation[right]) != right_gen:
            continue
        if right not in neighbours[left]:
            continue
        if distance_threshold is not None and distance > distance_threshold:
            threshold_blocked = True
            break

        keep, drop = (left, right) if left < right else (right, left)
        keep_size, drop_size = int(sizes[keep]), int(sizes[drop])
        total_size = keep_size + drop_size
        centroids[keep] = (keep_size * centroids[keep] + drop_size * centroids[drop]) / total_size
        sizes[keep] = total_size
        sizes[drop] = 0
        parent[drop] = keep

        boundary = (neighbours[keep] | neighbours[drop]) - {keep, drop}
        active_boundary = {node for node in boundary if active[node]}
        for neighbour in active_boundary:
            neighbours[neighbour].discard(keep)
            neighbours[neighbour].discard(drop)
            neighbours[neighbour].add(keep)
        neighbours[keep] = active_boundary
        neighbours[drop].clear()
        active[drop] = False
        generation[keep] += 1
        generation[drop] += 1
        achieved += 1
        merge_log.append((keep, drop, float(distance)))

        keep_generation = int(generation[keep])
        for neighbour in active_boundary:
            left_id, right_id = (keep, neighbour) if keep < neighbour else (neighbour, keep)
            new_distance = float(np.linalg.norm(centroids[left_id] - centroids[right_id]))
            heapq.heappush(
                heap,
                (
                    new_distance,
                    left_id,
                    right_id,
                    int(generation[left_id]),
                    int(generation[right_id]),
                ),
            )

    grouped: dict[int, list[int]] = {}
    for node in range(node_count):
        root = _find(parent, node)
        grouped.setdefault(root, []).append(node)
    clusters = [tuple(members) for _, members in sorted(grouped.items())]

    shortfall: str | None = None
    if achieved < requested:
        shortfall = "distance_threshold_prevented_target" if threshold_blocked else "insufficient_connected_cluster_merges"
    return _partition_from_clusters(
        node_count,
        clusters,
        merge_log,
        method="connectivity_agglomerative",
        reduction=target_reduction,
        seed=seed,
        shortfall=shortfall,
    )


def build_partition(adjacency: sparse.spmatrix, coordinates: np.ndarray, *, method: Method, target_reduction: float, seed: int, distance_threshold: float | None = None) -> PartitionResult:
    """Build a deterministic partition in Haken coordinates."""
    matrix = adjacency.tocsr()
    points = np.asarray(coordinates, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1] or points.ndim != 2 or points.shape[0] != matrix.shape[0]:
        raise ValueError("adjacency must be square and coordinates must have one row per node")
    if method not in {"connectivity_matching", "unconstrained_matching", "connectivity_agglomerative"}:
        raise ValueError("unsupported matching method")
    if method == "connectivity_agglomerative":
        return _connectivity_agglomerative(
            matrix,
            points,
            target_reduction=target_reduction,
            seed=seed,
            distance_threshold=distance_threshold,
        )

    requested = _requested_merges(matrix.shape[0], target_reduction)
    candidates = _connectivity_candidates(matrix, points) if method == "connectivity_matching" else _unconstrained_candidates(points)
    claimed: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for distance, left, right in candidates:
        if len(pairs) == requested:
            break
        if distance_threshold is not None and distance > distance_threshold:
            break
        if left not in claimed and right not in claimed:
            claimed.update((left, right))
            pairs.append((left, right, distance))
    if len(pairs) < requested:
        reason = "insufficient_disjoint_adjacent_pairs" if method == "connectivity_matching" else "insufficient_disjoint_nearest_neighbor_pairs"
        if distance_threshold is not None:
            reason = "distance_threshold_prevented_target"
    else:
        reason = None
    return _finish(matrix.shape[0], pairs, method=method, reduction=target_reduction, seed=seed, shortfall=reason)
