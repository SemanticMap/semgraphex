from __future__ import annotations

from pathlib import Path

import numpy as np

from semmap_haken.hierarchy import HierarchyResult, LevelEvidence, TransitionEvidence
from semmap_haken.multiscale_config import HierarchyOptions, PlateauOptions, SyntheticOptions
from semmap_haken.plateau import detect_plateaus
from semmap_haken.synthetic import build_synthetic_controls


def _transition(level: int, source_r: int = 4, target_r: int = 4, good: bool = True) -> TransitionEvidence:
    return TransitionEvidence(
        source_level=level,
        target_level=level + 1,
        source_r=source_r,
        target_r=target_r,
        fine_node_count=128 // (2**level),
        coarse_node_count=128 // (2 ** (level + 1)),
        compression_ratio=2.0,
        achieved_reduction=0.5,
        subspace_projection_distance=0.1 if good else 0.9,
        slow_eigenvalue_max_abs_error=0.05 if good else 0.9,
        mean_trajectory_relative_error=0.2 if good else 0.9,
        shortfall_reason=None,
        method="connectivity_matching",
    )


def test_plateau_requires_consecutive_good_transitions_and_stable_r() -> None:
    levels = tuple(LevelEvidence(i, 128 // (2**i), 0, 4, 0.9, -0.1, 0.2, 0.1, 1 + i, 10 + i) for i in range(5))
    transitions = (_transition(0), _transition(1), _transition(2), _transition(3, good=False))
    hierarchy = HierarchyResult(levels, transitions, tuple(), tuple(), "max_levels", "connectivity_matching", 0.1)
    report = detect_plateaus(hierarchy, PlateauOptions(min_consecutive=3))
    assert len(report.candidates) == 1
    assert report.candidates[0].start_level == 0
    assert report.candidates[0].end_level == 3

    unstable = HierarchyResult(levels, (_transition(0), _transition(1, target_r=3), _transition(2)), tuple(), tuple(), "max_levels", "connectivity_matching", 0.1)
    assert detect_plateaus(unstable, PlateauOptions(min_consecutive=3)).candidates == ()


def test_synthetic_controls_are_sparse_symmetric_and_distinct() -> None:
    positive, negative = build_synthetic_controls(SyntheticOptions(macro_blocks=4, nodes_per_block=8, internal_weight=1.0, bridge_weight=0.05, seed=7))
    assert positive.adjacency.shape == negative.adjacency.shape == (32, 32)
    assert (positive.adjacency - positive.adjacency.T).nnz == 0
    assert (negative.adjacency - negative.adjacency.T).nnz == 0
    assert positive.ground_truth["condition"] == "planted_hierarchical_timescale_separation"
    assert negative.ground_truth["condition"] == "no_planted_block_hierarchy"
    assert not np.array_equal(positive.adjacency.toarray(), negative.adjacency.toarray())
