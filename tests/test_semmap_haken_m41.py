from __future__ import annotations

import numpy as np

from semmap_haken.m41_metrics import M41TransitionMetrics, derive_fine_to_coarse, post_transient_relative_error, slow_order_parameter_error
from semmap_haken.m41_sweep import detect_plateau
from semmap_haken.metrics import trajectory_relative_error


def test_slow_metric_ignores_error_orthogonal_to_slow_basis() -> None:
    fine = np.array([[1.0, 2.0], [0.8, 1.5], [0.6, 1.0]])
    lifted = fine.copy()
    lifted[:, 1] += 10.0
    basis = np.array([[1.0], [0.0]])
    assert slow_order_parameter_error(fine, lifted, basis) < 1e-12
    assert trajectory_relative_error(fine, lifted) > 1.0


def test_post_transient_metric_discards_early_mismatch() -> None:
    fine = np.ones((4, 2))
    lifted = fine.copy()
    lifted[:2] = 0.0
    error, start_time, energy_fraction = post_transient_relative_error(
        fine, lifted, np.array([0.0, 1.0, 2.0, 3.0]), start_fraction=0.5
    )
    assert error == 0.0
    assert start_time == 2.0
    assert 0.0 < energy_fraction < 1.0


def test_derive_adjacent_membership_from_original_ancestry() -> None:
    fine = {0: ("a",), 1: ("b",), 2: ("c",), 3: ("d",)}
    coarse = {0: ("a", "b"), 1: ("c", "d")}
    np.testing.assert_array_equal(derive_fine_to_coarse(fine, coarse), np.array([0, 0, 1, 1]))


def _transition(level: int, *, slow_error: float = 0.1, full_error: float = 0.9, target_r: int = 3) -> M41TransitionMetrics:
    return M41TransitionMetrics(
        source_level=level, target_level=level + 1,
        fine_node_count=64 // (2**level), coarse_node_count=64 // (2 ** (level + 1)),
        source_r=3, target_r=target_r, achieved_reduction=0.5,
        subspace_projection_distance=0.1, slow_eigenvalue_max_abs_error=0.1,
        full_trajectory_error=full_error, slow_order_parameter_error=slow_error,
        post_transient_error=0.2, post_transient_start_fraction=0.5,
        post_transient_start_time=2.5, post_transient_energy_fraction=0.4,
    )


def test_detector_can_use_slow_metric_without_redefining_full_metric() -> None:
    metrics = (_transition(0), _transition(1))
    slow_detected, _ = detect_plateau(
        metrics, trajectory_metric="slow", r_tolerance=0, subspace_threshold=0.2,
        trajectory_threshold=0.25, eigenvalue_threshold=0.2, min_consecutive=2,
        min_achieved_reduction=0.1,
    )
    full_detected, _ = detect_plateau(
        metrics, trajectory_metric="full", r_tolerance=0, subspace_threshold=0.2,
        trajectory_threshold=0.25, eigenvalue_threshold=0.2, min_consecutive=2,
        min_achieved_reduction=0.1,
    )
    assert slow_detected is True
    assert full_detected is False
