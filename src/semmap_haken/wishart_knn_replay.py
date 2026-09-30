"""Replay Wishart k-neighbor settings on one saved exact-type population.

No ego extraction, dictionary mutation, graph contraction or checkpoint writes.
The input is the *complete* ordered nearest-neighbor list for every eligible
type at a completed source level; this allows exact k-NN after 80% subsampling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .wishart_cluster import wishart_cluster


K_GRID: tuple[int, ...] = (1, 2, 3, 4, 6, 12)


def subset_neighbors(
    full_indices: np.ndarray,
    full_distances: np.ndarray,
    selected: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter *complete* nearest-neighbor lists without recomputing distances.

    Selected IDs are original row positions, in ascending order. Mapping them
    back to local row positions preserves deterministic distance/tie ordering.
    """
    indices = np.asarray(full_indices, dtype=np.int64)
    distances = np.asarray(full_distances, dtype=np.float64)
    keep = np.asarray(selected, dtype=np.int64)
    n = indices.shape[0]
    if indices.shape != (n, max(0, n - 1)) or distances.shape != indices.shape:
        raise ValueError("full neighbor arrays must have shape (n, n-1)")
    if keep.ndim != 1 or not np.array_equal(keep, np.unique(keep)):
        raise ValueError("selected indices must be sorted and unique")
    if np.any(keep < 0) or np.any(keep >= n):
        raise ValueError("selected indices are out of range")
    if not 1 <= k < len(keep):
        raise ValueError("k must be between 1 and selected_count - 1")

    mask = np.zeros(n, dtype=bool)
    mask[keep] = True
    original_to_local = np.full(n, -1, dtype=np.int64)
    original_to_local[keep] = np.arange(len(keep), dtype=np.int64)
    picked_indices = np.empty((len(keep), k), dtype=np.int64)
    picked_distances = np.empty((len(keep), k), dtype=np.float64)
    for row, original in enumerate(keep):
        valid = mask[indices[original]]
        nearest = indices[original][valid][:k]
        nearest_distances = distances[original][valid][:k]
        if nearest.size != k or np.any(nearest == original):
            raise ValueError("full neighbor lists are missing required non-self neighbors")
        picked_indices[row] = original_to_local[nearest]
        picked_distances[row] = nearest_distances
    return picked_indices, picked_distances


def sweep_level(
    *,
    level: int,
    full_indices: np.ndarray,
    full_distances: np.ndarray,
    weights: np.ndarray,
    significance: float,
    min_cluster_size: int,
    min_cluster_mass: float | None,
    k_values: Sequence[int] = K_GRID,
    original_labels: np.ndarray | None = None,
    subset_fraction: float = 0.8,
    subset_repeats: int = 20,
    seed: int = 1729,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    """Compare Wishart families over k, with paired 80%-type stability trials.

    ARI is a partition-consistency diagnostic, not proof of semantic quality.
    In particular, one-cluster partitions can achieve trivially high ARI.
    """
    indices = np.asarray(full_indices, dtype=np.int64)
    distances = np.asarray(full_distances, dtype=np.float64)
    n = indices.shape[0]
    if n < 2 or indices.shape != (n, n - 1) or distances.shape != indices.shape:
        raise ValueError("provide all n-1 neighbors for each of at least two types")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("neighbor indices out of range")
    if np.any(indices == np.arange(n)[:, None]):
        raise ValueError("self-neighbors are forbidden")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("distances must be finite and non-negative")
    if any(len(np.unique(row)) != n - 1 for row in indices):
        raise ValueError("each full neighbor list must contain every other type once")
    masses = np.asarray(weights, dtype=np.float64)
    if masses.shape != (n,) or np.any(masses <= 0) or not np.all(np.isfinite(masses)):
        raise ValueError("weights must be positive finite values of length n")
    if original_labels is not None:
        original_labels = np.asarray(original_labels, dtype=np.int64)
        if original_labels.shape != (n,):
            raise ValueError("original labels must match the selected type population")
    if not 0 < subset_fraction < 1 or subset_repeats < 0:
        raise ValueError("subset_fraction must be in (0,1); repeats non-negative")
    ks = tuple(int(k) for k in k_values)
    if not ks or any(k < 1 or k >= n for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("k values must be unique integers in [1,n-1]")

    sample_size = min(n - 1, int(math.floor(n * subset_fraction)))
    rng = np.random.default_rng(seed + 1009 * int(level))
    subsets = [
        np.sort(rng.choice(n, size=sample_size, replace=False))
        for _ in range(subset_repeats)
    ] if sample_size >= 2 else []
    rows: list[dict[str, object]] = []
    all_labels: dict[int, np.ndarray] = {}

    for k in ks:
        fitted = wishart_cluster(
            indices[:, :k], distances[:, :k],
            significance=significance,
            min_cluster_size=min_cluster_size,
            sample_weights=masses,
            min_cluster_mass=min_cluster_mass,
        )
        labels = fitted.labels.copy()
        all_labels[k] = labels
        ari_samples: list[float] = []
        cluster_samples: list[int] = []
        for selected in subsets:
            if len(selected) <= k:
                continue
            sub_indices, sub_distances = subset_neighbors(
                indices, distances, selected, k=k,
            )
            sub = wishart_cluster(
                sub_indices, sub_distances,
                significance=significance,
                min_cluster_size=min_cluster_size,
                sample_weights=masses[selected],
                min_cluster_mass=min_cluster_mass,
            )
            ari_samples.append(float(adjusted_rand_score(labels[selected], sub.labels)))
            cluster_samples.append(sub.cluster_count)

        largest_size = max(fitted.cluster_sizes.values(), default=0)
        largest_mass = max(fitted.cluster_masses.values(), default=0.0)
        rows.append({
            "level": int(level),
            "k": int(k),
            "type_count": int(n),
            "family_count": int(fitted.cluster_count),
            "noise_fraction": float(np.mean(labels < 0)),
            "largest_family_fraction": float(largest_size / n),
            "largest_family_mass_fraction": float(largest_mass / masses.sum()),
            "zero_kth_radius_fraction": float(np.mean(fitted.kth_radius <= 1e-8)),
            "stability_ari_median": (
                float(median(ari_samples)) if ari_samples else None
            ),
            "stability_ari_p10": (
                float(np.quantile(ari_samples, 0.10)) if ari_samples else None
            ),
            "subset_family_count_median": (
                float(median(cluster_samples)) if cluster_samples else None
            ),
            "stability_repeats_used": len(ari_samples),
            "baseline_ari": (
                float(adjusted_rand_score(original_labels, labels))
                if original_labels is not None and k == 12 else None
            ),
            "baseline_labels_exact": (
                bool(np.array_equal(original_labels, labels))
                if original_labels is not None and k == 12 else None
            ),
        })
    return rows, all_labels
