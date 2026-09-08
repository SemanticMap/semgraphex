"""Deterministic, sparse-aware one-step topology-only matching contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

Method = Literal["connectivity_matching", "unconstrained_matching"]


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


def _finish(node_count: int, pairs: list[tuple[int, int, float]], *, method: Method, reduction: float, seed: int, shortfall: str | None) -> PartitionResult:
    claimed = {node for left, right, _ in pairs for node in (left, right)}
    clusters = [tuple(sorted((left, right))) for left, right, _ in pairs]
    clusters.extend((node,) for node in range(node_count) if node not in claimed)
    clusters.sort(key=lambda cluster: cluster[0])
    assignment = np.empty(node_count, dtype=np.int64)
    for coarse, cluster in enumerate(clusters):
        assignment[list(cluster)] = coarse
    requested = _requested_merges(node_count, reduction)
    return PartitionResult(
        fine_to_coarse=assignment, clusters=tuple(clusters), merges=tuple(pairs), method=method,
        requested_merges=requested, achieved_merges=len(pairs), requested_reduction=reduction,
        achieved_reduction=len(pairs) / node_count if node_count else 0.0,
        coarse_node_count=len(clusters), shortfall_reason=shortfall, seed=seed,
        tie_breaking="distance_then_node_index", task_identity=f"m2:{method}:seed={seed}",
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
    # A small constant neighbour fan-out permits greedy matching to recover from
    # mutual-nearest collisions while remaining O(N log N) / O(N) in memory.
    neighbour_count = min(8, count)
    distances, indices = tree.query(coordinates, k=neighbour_count)
    candidates = {
        (float(distances[row, column]), min(row, int(indices[row, column])), max(row, int(indices[row, column])))
        for row in range(count)
        for column in range(1, neighbour_count)
        if int(indices[row, column]) != row
    }
    return sorted(candidates)


def build_partition(adjacency: sparse.spmatrix, coordinates: np.ndarray, *, method: Method, target_reduction: float, seed: int, distance_threshold: float | None = None) -> PartitionResult:
    """Greedily choose stable disjoint pairs from sparse/topological candidates."""
    matrix = adjacency.tocsr()
    points = np.asarray(coordinates, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1] or points.ndim != 2 or points.shape[0] != matrix.shape[0]:
        raise ValueError("adjacency must be square and coordinates must have one row per node")
    if method not in {"connectivity_matching", "unconstrained_matching"}:
        raise ValueError("unsupported matching method")
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
