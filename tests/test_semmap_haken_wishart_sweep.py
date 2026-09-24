"""Offline Wishart k-sweep regression tests (no ConceptNet or Colab runtime)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from semmap_haken.wishart_cluster import wishart_cluster
from semmap_haken.wishart_knn_replay import K_GRID, subset_neighbors, sweep_level


def _complete_neighbors() -> tuple[np.ndarray, np.ndarray]:
    # Three geometrically separated groups, with deterministic tie order.
    coords = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 7.0, 7.1, 7.2])
    distances = abs(coords[:, None] - coords[None, :])
    np.fill_diagonal(distances, np.inf)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :-1]
    return indices, np.take_along_axis(distances, indices, axis=1)


def test_subset_neighbor_order_is_exact_after_excluding_types() -> None:
    indices, distances = _complete_neighbors()
    keep = np.array([0, 2, 3, 5, 7, 8])
    sub_indices, sub_distances = subset_neighbors(
        indices, distances, keep, k=3,
    )
    inverse = {int(old): new for new, old in enumerate(keep)}
    for row, old in enumerate(keep):
        allowed = [int(value) for value in indices[old] if value in inverse]
        expected = np.array([inverse[value] for value in allowed[:3]])
        np.testing.assert_array_equal(sub_indices[row], expected)
        source = distances[old][np.isin(indices[old], keep)][:3]
        np.testing.assert_array_equal(sub_distances[row], source)


def test_subset_nearest_neighbors_require_complete_input() -> None:
    indices, distances = _complete_neighbors()
    with pytest.raises(ValueError, match="shape"):
        subset_neighbors(indices[:, :4], distances[:, :4], np.array([0, 2, 3]), k=2)
    with pytest.raises(ValueError, match="sorted and unique"):
        subset_neighbors(indices, distances, np.array([2, 0, 3]), k=2)
    with pytest.raises(ValueError, match="selected_count"):
        subset_neighbors(indices, distances, np.array([0, 2, 3]), k=3)


def test_sweep_reuses_full_neighbors_and_has_reproducible_stability() -> None:
    indices, distances = _complete_neighbors()
    opts = {
        "level": 2,
        "full_indices": indices,
        "full_distances": distances,
        "weights": np.array([4., 5., 3., 4., 5., 3., 4., 5., 3.]),
        "significance": 0.7,
        "min_cluster_size": 2,
        "min_cluster_mass": 5.0,
        "k_values": (1, 2, 3),
        "subset_fraction": 0.8,
        "subset_repeats": 6,
        "seed": 1729,
    }
    first, labels_a = sweep_level(**opts)
    second, labels_b = sweep_level(**opts)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first) == 3
    for k in (1, 2, 3):
        np.testing.assert_array_equal(labels_a[k], labels_b[k])
    for row in first:
        assert row["stability_repeats_used"] == 6
        assert 0.0 <= row["noise_fraction"] <= 1.0
        assert 0.0 <= row["largest_family_fraction"] <= 1.0
        assert 0.0 <= row["largest_family_mass_fraction"] <= 1.0
        assert -1.0 <= row["stability_ari_median"] <= 1.0
        assert row["baseline_ari"] is None


def test_baseline_comparison_uses_ari_and_exact_label_check() -> None:
    indices, distances = _complete_neighbors()
    baseline = wishart_cluster(
        indices[:, :2], distances[:, :2],
        significance=0.7, min_cluster_size=2,
        sample_weights=np.ones(9), min_cluster_mass=None,
    )
    # Baseline metrics are present only for the original configuration k=12.
    coords = np.arange(15, dtype=float)
    distance = abs(coords[:, None] - coords[None, :])
    np.fill_diagonal(distance, np.inf)
    full_idx = np.argsort(distance, axis=1, kind="stable")[:, :-1]
    full_dst = np.take_along_axis(distance, full_idx, axis=1)
    original = wishart_cluster(
        full_idx[:, :12], full_dst[:, :12],
        significance=0.7, min_cluster_size=3,
        sample_weights=np.ones(15), min_cluster_mass=10.0,
    )
    rows, labels = sweep_level(
        level=0, full_indices=full_idx, full_distances=full_dst,
        weights=np.ones(15), significance=0.7,
        min_cluster_size=3, min_cluster_mass=10.0,
        k_values=(12,), original_labels=original.labels,
        subset_repeats=0,
    )
    assert rows[0]["baseline_ari"] == 1.0
    assert rows[0]["baseline_labels_exact"] is True
    np.testing.assert_array_equal(labels[12], original.labels)
    assert adjusted_rand_score(baseline.labels, baseline.labels) == 1.0


def test_grid_is_pre_registered_and_invalid_neighbors_fail() -> None:
    assert K_GRID == (1, 2, 3, 4, 6, 12)
    indices, distances = _complete_neighbors()
    bad = indices.copy()
    bad[0, 0] = 0
    with pytest.raises(ValueError, match="self"):
        sweep_level(
            level=0, full_indices=bad, full_distances=distances,
            weights=np.ones(9), significance=0.7,
            min_cluster_size=3, min_cluster_mass=10.0, k_values=(1,),
            subset_repeats=0,
        )


def test_colab_notebook_does_not_rerun_graph_compression() -> None:
    source = Path("notebooks/06_wishart_knn_replay_colab.ipynb")
    notebook = json.loads(source.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "load_latest_checkpoint" in code
    assert "sweep_level" in code
    assert "K_VALUES = (1, 2, 3, 4, 6, 12)" in code
    assert "GITHUB_TOKEN" in code
    assert "run_wishart_hierarchy(" not in code
    assert "semmap-wishart-colab" not in code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook_cell_{index}")
