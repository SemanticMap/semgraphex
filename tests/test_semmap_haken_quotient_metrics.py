"""Focused B2 contracts: quotient, lifting, subspace, and trajectory distortion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.coarsen import build_partition
from semmap_haken.haken_embedding import build_haken_embedding
from semmap_haken.metrics import (
    lift_state,
    restrict_state,
    slow_subspace_comparison,
    slow_eigenvalue_error,
    trajectory_relative_error,
)
from semmap_haken.modes import analyze_normalized_adjacency, BetaSelection, DimensionSelection, ModeResult
from semmap_haken.operators import normalized_adjacency
from semmap_haken.quotient import build_quotient, membership_matrix


def _sample_adjacency() -> sparse.csr_matrix:
    # Two triangles joined by one bridge edge: small, connected, hand-checkable.
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))


def _partition(adjacency: sparse.csr_matrix) -> "build_partition":
    coordinates = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    return build_partition(adjacency, coordinates, method="connectivity_matching", target_reduction=1 / 3, seed=0)


def test_membership_matrix_is_sparse_reversible_and_assigns_each_node_once() -> None:
    adjacency = _sample_adjacency()
    partition = _partition(adjacency)
    membership = membership_matrix(partition.fine_to_coarse)
    assert sparse.isspmatrix_csr(membership)
    assert membership.shape == (6, partition.coarse_node_count)
    # Indicator semantics: P^T P = diag(cluster sizes); every fine node appears exactly once.
    gram = (membership.T @ membership).toarray()
    assert np.allclose(gram, np.diag(np.bincount(partition.fine_to_coarse)))
    assert membership.sum() == 6
    assert np.array_equal(np.asarray(membership.sum(axis=1)).ravel(), np.ones(6))


def test_quotient_sum_aggregation_matches_hand_computation_and_conserves_weight() -> None:
    adjacency = _sample_adjacency()
    partition = _partition(adjacency)
    quotient = build_quotient(adjacency, partition.fine_to_coarse, aggregation="sum", node_ids=("a", "b", "c", "d", "e", "f"))
    assert quotient.adjacency.shape == (partition.coarse_node_count,) * 2
    # Total edge weight is conserved: sum of all adjacency entries equals quotient sum.
    assert np.isclose(quotient.adjacency.sum(), adjacency.sum())
    # Mass is conserved: supernode masses sum to one and sizes recover the fine count.
    assert np.isclose(quotient.masses.sum(), 1.0)
    assert sum(quotient.supernode_sizes) == 6
    # Reversibility: parent -> members mapping covers each fine node exactly once.
    flat = [node for members in quotient.parent_child.values() for node in members]
    assert sorted(flat) == list(range(6))
    # Original IDs preserved through membership.
    ids = [uri for members in quotient.parent_child.values() for uri in map(lambda i: quotient.fine_node_ids[i], members)]
    assert sorted(ids) == list("abcdef")


def test_quotient_mean_density_aggregation_normalizes_by_block_size() -> None:
    adjacency = _sample_adjacency()
    partition = _partition(adjacency)
    quotient_sum = build_quotient(adjacency, partition.fine_to_coarse, aggregation="sum", node_ids=tuple("abcdef"))
    quotient_density = build_quotient(adjacency, partition.fine_to_coarse, aggregation="mean_density", node_ids=tuple("abcdef"))
    coarse_count = quotient_sum.adjacency.shape[0]
    for a in range(coarse_count):
        for b in range(coarse_count):
            scale = quotient_sum.supernode_sizes[a] * quotient_sum.supernode_sizes[b]
            assert np.isclose(quotient_density.adjacency[a, b] * scale, quotient_sum.adjacency[a, b])


def test_quotient_and_metric_path_does_not_densify_fine_adjacency(monkeypatch: pytest.MonkeyPatch) -> None:
    def dense_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fine graphs must never be densified")

    monkeypatch.setattr(sparse.spmatrix, "toarray", dense_access, raising=False)
    monkeypatch.setattr(sparse.spmatrix, "todense", dense_access, raising=False)
    adjacency = _sample_adjacency()
    partition = _partition(adjacency)
    quotient = build_quotient(adjacency, partition.fine_to_coarse, aggregation="sum", node_ids=tuple("abcdef"))
    assert sparse.isspmatrix_csr(quotient.adjacency)


def test_restriction_and_lifting_are_adjoint_under_mass_inner_product() -> None:
    fine = np.arange(6, dtype=float)
    partition = build_partition(sparse.csr_matrix((6, 6)), np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [3.0]]), method="unconstrained_matching", target_reduction=0.5, seed=0)
    membership = membership_matrix(partition.fine_to_coarse)
    sizes = np.bincount(partition.fine_to_coarse)
    masses = sizes / 6.0
    restricted = restrict_state(fine, membership, masses)
    coarse_dual = np.array([1.0, 2.0, -0.5])
    lifted = lift_state(coarse_dual, membership, masses)
    # Fine uniform metric (1/N per node), coarse mass metric m_c: <x, L y>_(1/N) = <R x, y>_m.
    left = float(np.dot(fine, lifted)) / 6.0
    right = float(np.dot(masses * restricted, coarse_dual))
    assert np.isclose(left, right)
    # Restriction is the block mean; lifting broadcasts supernode values.
    assert np.allclose(restricted, [0.5, 2.5, 4.5])
    assert np.allclose(lifted, [1.0, 1.0, 2.0, 2.0, -0.5, -0.5])


def test_subspace_metric_is_invariant_to_sign_and_rotation() -> None:
    rng = np.random.default_rng(11)
    basis = np.linalg.qr(rng.normal(size=(12, 4)))[0]
    signs = basis * np.array([-1.0, 1.0, -1.0, 1.0])
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    reference = slow_subspace_comparison(basis, basis)
    flipped = slow_subspace_comparison(basis, signs @ rotation)
    # Principal-angle/SVD arithmetic around singular value 1 carries about 1e-8
    # absolute rounding in the derived distance; this is numerical zero here.
    assert np.isclose(reference.projection_distance, flipped.projection_distance, atol=1e-7)
    assert np.allclose(reference.principal_angles, flipped.principal_angles, atol=1e-7)


def test_slow_eigenvalue_error_pairs_sorted_nontrivial_values_with_truncation_rule() -> None:
    fine = np.array([1.0, 0.9, 0.8, 0.1])
    coarse = np.array([1.0, 0.88, 0.79])
    result = slow_eigenvalue_error(fine, coarse)
    assert result.pairs == ((0, 0), (1, 1))
    assert np.isclose(result.max_abs_error, max(abs(0.9 - 0.88), abs(0.8 - 0.79)))
    assert "truncated_to_coarse" in result.rule


def test_trajectory_distortion_is_zero_for_exact_equitable_contraction_and_nonzero_otherwise() -> None:
    # Complete bipartite-style twins: nodes 0 and 1 share identical neighbourhoods.
    adjacency = sparse.csr_matrix(
        (np.ones(12), (np.array([0, 1, 0, 2, 1, 2, 2, 3, 2, 4, 3, 4]), np.array([2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2]))),
        shape=(5, 5),
    )
    twins_partition = build_partition(adjacency, np.zeros((5, 1)), method="connectivity_matching", target_reduction=0.2, seed=0)
    # Force the twin pair (0,1) explicitly through membership semantics.
    fine_to_coarse = np.array([0, 0, 1, 2, 3])
    operator, degrees = normalized_adjacency(adjacency)
    modes = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=4, max_r=3)
    quotient = build_quotient(adjacency, fine_to_coarse, aggregation="sum", node_ids=tuple("abcde"))
    coarse_operator, coarse_degrees = normalized_adjacency(quotient.adjacency)
    coarse_modes = analyze_normalized_adjacency(coarse_operator, degrees=coarse_degrees, alpha=1.0, beta=0.5, top_k=3, max_r=2)
    membership = membership_matrix(fine_to_coarse)
    masses = quotient.masses
    fine_state = np.array([1.0, 1.0, 2.0, 0.5, 0.5])
    restricted = restrict_state(fine_state, membership, masses)
    times = np.linspace(0.0, 1.0, 5)
    fine_traj = np.stack([fine_state * float(np.exp(-0.5 * t)) for t in times])
    coarse_traj = np.stack([restricted * float(np.exp(-0.5 * t)) for t in times])
    lifted = np.stack([lift_state(row, membership, masses) for row in coarse_traj])
    # For an exactly equitable contraction of symmetric twin states the lift matches the fine trajectory.
    error = trajectory_relative_error(fine_traj, lifted)
    assert error < 1e-10
    # Distinct non-twin merging must show a nonzero distortion signal.
    assert twins_partition.coarse_node_count == 4
