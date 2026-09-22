"""M4.1 trajectory metrics for Haken-coarsening validation.

M4 compared fine and lifted-coarse trajectories in the full node state. M4.1
keeps that control metric and adds two diagnostics:

* slow_order_parameter_error: error only in amplitudes of the selected slow
  spectral subspace. This asks whether candidate macroscopic variables are
  preserved even when intentionally discarded fast coordinates differ.
* post_transient_error: full-state error after the initial transient interval.
  A transient is the short-lived response immediately after a perturbation.

These metrics are diagnostics; they do not prove that the selected modes are
Haken order parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .dynamics import build_perturbations, run_linear_dynamics
from .metrics import lift_state, restrict_state, slow_eigenvalue_error, slow_subspace_comparison, trajectory_relative_error
from .modes import analyze_normalized_adjacency
from .operators import normalized_adjacency
from .quotient import membership_matrix


@dataclass(frozen=True)
class M41TransitionMetrics:
    source_level: int
    target_level: int
    fine_node_count: int
    coarse_node_count: int
    source_r: int
    target_r: int
    achieved_reduction: float
    subspace_projection_distance: float
    slow_eigenvalue_max_abs_error: float
    full_trajectory_error: float
    slow_order_parameter_error: float
    post_transient_error: float
    post_transient_start_fraction: float
    post_transient_start_time: float
    post_transient_energy_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have identical shapes")
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(approximation) == 0.0 else float("inf")
    return float(np.linalg.norm(reference - approximation) / denominator)


def slow_order_parameter_error(fine_trajectory: np.ndarray, lifted_coarse_trajectory: np.ndarray, fine_slow_basis: np.ndarray) -> float:
    """Relative error of amplitudes in the fine slow subspace.

    If Q contains orthonormal slow modes, a(t)=x(t)Q are their amplitudes. The
    same rotation/sign change of Q affects both trajectories and leaves this
    Frobenius relative error unchanged.
    """
    fine = np.asarray(fine_trajectory, dtype=np.float64)
    lifted = np.asarray(lifted_coarse_trajectory, dtype=np.float64)
    basis = np.asarray(fine_slow_basis, dtype=np.float64)
    if fine.shape != lifted.shape or fine.ndim != 2:
        raise ValueError("trajectories must have identical (time,node) shape")
    if basis.ndim != 2 or basis.shape[0] != fine.shape[1] or basis.shape[1] < 1:
        raise ValueError("fine_slow_basis must be a non-empty node-by-mode matrix")
    q, _ = np.linalg.qr(basis)
    return _relative_error(fine @ q, lifted @ q)


def post_transient_relative_error(fine_trajectory: np.ndarray, lifted_coarse_trajectory: np.ndarray, time_grid: np.ndarray, *, start_fraction: float) -> tuple[float, float, float]:
    """Return tail error, actual tail start time, and remaining-energy fraction."""
    fine = np.asarray(fine_trajectory, dtype=np.float64)
    lifted = np.asarray(lifted_coarse_trajectory, dtype=np.float64)
    times = np.asarray(time_grid, dtype=np.float64)
    if fine.shape != lifted.shape or fine.ndim != 2 or fine.shape[0] != times.size:
        raise ValueError("trajectory/time shapes disagree")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must be in [0,1)")
    requested = float(times[0] + start_fraction * (times[-1] - times[0]))
    start_index = min(int(np.searchsorted(times, requested, side="left")), len(times) - 1)
    tail_reference = fine[start_index:]
    tail_approximation = lifted[start_index:]
    full_norm = float(np.linalg.norm(fine))
    energy_fraction = float(np.linalg.norm(tail_reference) / full_norm) if full_norm else 0.0
    return _relative_error(tail_reference, tail_approximation), float(times[start_index]), energy_fraction


def derive_fine_to_coarse(fine_membership: Mapping[int, Sequence[str]], coarse_membership: Mapping[int, Sequence[str]]) -> np.ndarray:
    """Recover the adjacent-scale contraction map from original-node ancestry."""
    original_to_coarse: dict[str, int] = {}
    for coarse, members in coarse_membership.items():
        for member in members:
            key = str(member)
            if key in original_to_coarse:
                raise ValueError("coarse membership contains duplicate original node")
            original_to_coarse[key] = int(coarse)
    assignment = np.empty(len(fine_membership), dtype=np.int64)
    for fine in range(len(fine_membership)):
        targets = {original_to_coarse[str(member)] for member in fine_membership[fine]}
        if len(targets) != 1:
            raise ValueError("fine supernode was split across adjacent coarse nodes")
        assignment[fine] = targets.pop()
    return assignment


def evaluate_transition_m41(fine_adjacency: sparse.spmatrix, coarse_adjacency: sparse.spmatrix, fine_to_coarse: np.ndarray, *, config: object, source_level: int, post_transient_start_fraction: float, compute: object | None = None) -> M41TransitionMetrics:
    """Recompute one adjacent-scale comparison with all three trajectory metrics."""
    fine_operator, fine_degrees = normalized_adjacency(fine_adjacency, symmetry_tolerance=config.spectral.symmetry_tolerance)
    coarse_operator, coarse_degrees = normalized_adjacency(coarse_adjacency, symmetry_tolerance=config.spectral.symmetry_tolerance)
    fine_modes = analyze_normalized_adjacency(
        fine_operator, degrees=fine_degrees, alpha=config.dynamics.alpha, beta=config.dynamics.beta,
        margin=config.dynamics.auto_critical_margin, top_k=config.spectral.top_k, max_r=config.spectral.max_r,
        tolerance=config.spectral.tolerance, maxiter=config.spectral.maxiter, compute=compute,
    )
    shared_beta = float(fine_modes.beta_selection.beta)
    coarse_modes = analyze_normalized_adjacency(
        coarse_operator, degrees=coarse_degrees, alpha=config.dynamics.alpha, beta=shared_beta,
        margin=config.dynamics.auto_critical_margin, top_k=config.spectral.top_k, max_r=config.spectral.max_r,
        tolerance=config.spectral.tolerance, maxiter=config.spectral.maxiter, compute=compute,
    )
    seed = int(config.dynamics.perturbation_seed + 1009 * source_level)
    batch = build_perturbations(
        node_count=fine_operator.shape[0], degrees=fine_degrees, seed=seed,
        amplitude=config.dynamics.perturbation_amplitude, per_kind=config.dynamics.perturbations_per_kind,
        sparse_fraction=config.dynamics.random_sparse_fraction,
    )
    time_grid = np.linspace(config.dynamics.time_start, config.dynamics.time_stop, config.dynamics.time_steps)
    fine_run = run_linear_dynamics(
        fine_operator, fine_modes, initial_states=batch.initial_states, labels=batch.labels, time_grid=time_grid,
        storage_policy="all", max_storage_bytes=config.dynamics.max_storage_mb * 1024 * 1024, compute=compute,
    )
    membership = membership_matrix(np.asarray(fine_to_coarse, dtype=np.int64))
    coarse_initial = np.asarray([restrict_state(state, membership) for state in batch.initial_states], dtype=np.float64)
    coarse_run = run_linear_dynamics(
        coarse_operator, coarse_modes, initial_states=coarse_initial, labels=batch.labels, time_grid=time_grid,
        storage_policy="all", max_storage_bytes=config.dynamics.max_storage_mb * 1024 * 1024, compute=compute,
    )
    if fine_run.trajectories is None or coarse_run.trajectories is None:
        raise ValueError("M4.1 requires stored fine and coarse trajectories")

    fine_r = min(fine_modes.selection.selected_r, fine_modes.eigenvectors.shape[1] - 1)
    coarse_r = min(coarse_modes.selection.selected_r, coarse_modes.eigenvectors.shape[1] - 1)
    fine_basis = fine_modes.eigenvectors[:, 1 : 1 + fine_r]
    lifted_coarse_basis = np.asarray(membership @ coarse_modes.eigenvectors[:, 1 : 1 + coarse_r])
    subspace = slow_subspace_comparison(fine_basis, lifted_coarse_basis)
    eigen_error = slow_eigenvalue_error(fine_modes.eigenvalues, coarse_modes.eigenvalues)

    full_errors: list[float] = []
    slow_errors: list[float] = []
    tail_errors: list[float] = []
    tail_energies: list[float] = []
    tail_start_time = float(time_grid[-1])
    for index in range(fine_run.trajectories.shape[0]):
        fine = fine_run.trajectories[index]
        lifted = np.asarray([lift_state(row, membership) for row in coarse_run.trajectories[index]], dtype=np.float64)
        full_errors.append(trajectory_relative_error(fine, lifted))
        slow_errors.append(slow_order_parameter_error(fine, lifted, fine_basis))
        tail_error, tail_start_time, energy_fraction = post_transient_relative_error(
            fine, lifted, time_grid, start_fraction=post_transient_start_fraction
        )
        tail_errors.append(tail_error)
        tail_energies.append(energy_fraction)

    achieved = (fine_operator.shape[0] - coarse_operator.shape[0]) / fine_operator.shape[0]
    return M41TransitionMetrics(
        source_level=source_level,
        target_level=source_level + 1,
        fine_node_count=int(fine_operator.shape[0]),
        coarse_node_count=int(coarse_operator.shape[0]),
        source_r=int(fine_r),
        target_r=int(coarse_r),
        achieved_reduction=float(achieved),
        subspace_projection_distance=float(subspace.projection_distance),
        slow_eigenvalue_max_abs_error=float(eigen_error.max_abs_error),
        full_trajectory_error=float(np.mean(full_errors)),
        slow_order_parameter_error=float(np.mean(slow_errors)),
        post_transient_error=float(np.mean(tail_errors)),
        post_transient_start_fraction=float(post_transient_start_fraction),
        post_transient_start_time=float(tail_start_time),
        post_transient_energy_fraction=float(np.mean(tail_energies)),
    )
