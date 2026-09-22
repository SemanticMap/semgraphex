"""k-NN Wishart mode-analysis clustering.

This is a deterministic, metric-agnostic implementation of the mode-analysis
idea used in modern Wishart variants: process observations from high to low
local density, grow connected modes, and refuse to bridge two modes when both
stand sufficiently above the current density saddle.

The density proxy is `-log(r_k + eps)`, where `r_k` is the distance to the
k-th nearest neighbour.  A cluster is height-significant at a saddle when
`peak_log_density - saddle_log_density >= h`.  Using a log-density proxy makes
`h` independent of an unknown embedding dimension, which is useful for
non-Euclidean graph distances.  This is an operational adaptation, not a claim
that graph-space density has a canonical Euclidean volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NOISE = -1
UNASSIGNED = -2


@dataclass(frozen=True)
class WishartClustering:
    labels: np.ndarray
    kth_radius: np.ndarray
    log_density: np.ndarray
    cluster_peaks: dict[int, float]
    completed_clusters: tuple[int, ...]
    cluster_sizes: dict[int, int]

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_sizes)


def wishart_cluster(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    *,
    significance: float,
    min_cluster_size: int = 2,
) -> WishartClustering:
    """Cluster observations using a k-NN Wishart density-mode sweep.

    `neighbor_indices[i]` and `neighbor_distances[i]` contain the k nearest
    *other* observations for item i in ascending distance order.
    """
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    if indices.ndim != 2 or distances.shape != indices.shape:
        raise ValueError("neighbor arrays must be equally shaped 2-D arrays")
    n, k = indices.shape
    if n == 0 or k == 0:
        return WishartClustering(
            labels=np.full(n, NOISE, dtype=np.int64),
            kth_radius=np.zeros(n),
            log_density=np.zeros(n),
            cluster_peaks={},
            completed_clusters=(),
            cluster_sizes={},
        )
    if np.any(distances < 0) or not np.all(np.isfinite(distances)):
        raise ValueError("neighbor distances must be finite and non-negative")
    if significance < 0:
        raise ValueError("significance must be non-negative")

    kth = distances[:, -1]
    eps = np.finfo(np.float64).eps
    density = -np.log(np.maximum(kth, eps))
    order = np.lexsort((np.arange(n, dtype=np.int64), kth))

    labels = np.full(n, UNASSIGNED, dtype=np.int64)
    processed = np.zeros(n, dtype=bool)
    members: dict[int, set[int]] = {}
    peaks: dict[int, float] = {}
    completed: set[int] = set()
    next_cluster = 0

    def new_cluster(point: int) -> int:
        nonlocal next_cluster
        cid = next_cluster
        next_cluster += 1
        labels[point] = cid
        members[cid] = {point}
        peaks[cid] = float(density[point])
        return cid

    def assign(point: int, cid: int) -> None:
        labels[point] = cid
        members.setdefault(cid, set()).add(point)
        peaks[cid] = max(peaks.get(cid, -np.inf), float(density[point]))

    def merge_clusters(cluster_ids: list[int], point: int) -> int:
        survivor = min(
            cluster_ids,
            key=lambda cid: (-peaks.get(cid, -np.inf), cid),
        )
        for cid in cluster_ids:
            if cid == survivor:
                continue
            for item in members.get(cid, ()):
                labels[item] = survivor
                members.setdefault(survivor, set()).add(item)
            peaks[survivor] = max(peaks.get(survivor, -np.inf), peaks.get(cid, -np.inf))
            members.pop(cid, None)
            peaks.pop(cid, None)
            completed.discard(cid)
        assign(point, survivor)
        return survivor

    for point in order:
        point = int(point)
        local_neighbors = [
            int(j)
            for j, d in zip(indices[point], distances[point], strict=True)
            if processed[int(j)] and d <= kth[point] + 1e-12
        ]
        cluster_ids = sorted(
            {int(labels[j]) for j in local_neighbors if labels[j] >= 0}
        )

        if not cluster_ids:
            new_cluster(point)
        elif len(cluster_ids) == 1:
            cid = cluster_ids[0]
            if cid in completed:
                labels[point] = NOISE
            else:
                assign(point, cid)
        else:
            active = [cid for cid in cluster_ids if cid not in completed]
            if not active:
                labels[point] = NOISE
            else:
                saddle = float(density[point])
                significant = [
                    cid for cid in active
                    if peaks.get(cid, saddle) - saddle >= significance
                ]
                if len(significant) >= 2:
                    # A low-density bridge between multiple established modes.
                    # Do not join the modes; less significant touching branches
                    # are completed and the bridge point is marked as noise.
                    for cid in active:
                        if cid not in significant:
                            completed.add(cid)
                    labels[point] = NOISE
                else:
                    merge_clusters(active, point)
        processed[point] = True

    # Tiny modes do not constitute compression-figure types.
    final_sizes: dict[int, int] = {}
    for cid in sorted(set(int(x) for x in labels if x >= 0)):
        size = int(np.sum(labels == cid))
        if size < min_cluster_size:
            labels[labels == cid] = NOISE
        else:
            final_sizes[cid] = size

    # Normalize surviving labels for stable artifact comparison.
    remap = {old: new for new, old in enumerate(sorted(final_sizes))}
    normalized = np.array(
        [remap.get(int(label), NOISE) for label in labels],
        dtype=np.int64,
    )
    normalized_peaks = {remap[cid]: float(peaks[cid]) for cid in remap}
    normalized_sizes = {
        remap[cid]: int(np.sum(normalized == remap[cid])) for cid in remap
    }
    normalized_completed = tuple(sorted(remap[cid] for cid in completed if cid in remap))
    return WishartClustering(
        labels=normalized,
        kth_radius=kth,
        log_density=density,
        cluster_peaks=normalized_peaks,
        completed_clusters=normalized_completed,
        cluster_sizes=normalized_sizes,
    )
