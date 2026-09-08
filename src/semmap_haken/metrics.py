"""Restriction, lifting, and cross-scale distortion metrics for M2.

Documented semantics:

- Inner product: the fine space carries the uniform measure metric
  ``<x, y>_(1/N) = (1/N) sum_i x_i y_i``; the coarse space carries the mass
  metric ``<u, v>_m = sum_c m_c u_c v_c`` with supernode mass ``m_c = |C_c|/N``.
- Restriction: ``R(x)_c = sum_{i in C_c} x_i / |C_c|`` (mass-aware block mean).
- Lifting: ``L(y)_i = y_{c(i)}`` (piecewise-constant broadcast).  Under these
  metrics ``L`` is the exact adjoint of ``R``:
  ``<x, L(y)>_(1/N) = <R(x), y>_m``.
- Subspace comparison uses orthonormal bases and principal angles only; it is
  invariant to per-vector sign flips and rotations inside a subspace.  Raw
  eigenvectors are never compared directly.
- Slow-eigenvalue error drops the Perron entry (index 0) of each input
  spectrum, pairs the remaining sorted eigenvalues by rank (0-based into the
  nontrivial lists), and truncates to the shorter list, recording the pairing
  and truncation rule.
- Trajectory error is ``||x(t) - L y(t)||_F / ||x(t)||_F`` over the whole
  batch, which is distinct from the M1 rank-r reconstruction error that
  compares a fine trajectory with its own spectral projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import svd

from .quotient import membership_matrix


def restrict_state(fine_state: np.ndarray, membership: sparse.csr_matrix, masses: np.ndarray | None = None) -> np.ndarray:
    """Mass-aware block mean ``R(x)_c = sum_{i in C_c} x_i / |C_c|``."""
    state = np.asarray(fine_state, dtype=np.float64)
    if state.ndim != 1 or state.shape[0] != membership.shape[0]:
        raise ValueError("fine_state must be one value per fine node")
    sizes = np.asarray(membership.sum(axis=0)).ravel()
    totals = np.asarray(membership.T @ state).ravel()
    return totals / sizes


def lift_state(coarse_state: np.ndarray, membership: sparse.csr_matrix, masses: np.ndarray | None = None) -> np.ndarray:
    """Piecewise-constant broadcast ``L(y)_i = y_{c(i)}``."""
    state = np.asarray(coarse_state, dtype=np.float64)
    if state.ndim != 1 or state.shape[0] != membership.shape[1]:
        raise ValueError("coarse_state must be one value per supernode")
    return np.asarray(membership @ state).ravel()


@dataclass(frozen=True)
class SubspaceComparison:
    principal_angles: np.ndarray
    projection_distance: float
    compared_rank: int


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    q, _ = np.linalg.qr(matrix)
    return q


def slow_subspace_comparison(fine_basis: np.ndarray, coarse_basis: np.ndarray) -> SubspaceComparison:
    """Compare subspaces via principal angles and projection-matrix distance.

    Both inputs are orthonormalized first, so sign flips and rotations inside
    either subspace cannot change the result.
    """
    fine_q = _orthonormalize(fine_basis)
    coarse_q = _orthonormalize(coarse_basis)
    rank = min(fine_q.shape[1], coarse_q.shape[1])
    fine_q, coarse_q = fine_q[:, :rank], coarse_q[:, :rank]
    # Principal angles: clipped singular values of Q_f^T Q_c (sorted descending).
    products = np.clip(fine_q.T @ coarse_q, -1.0, 1.0)
    _, singular, _ = svd(products)
    angles = np.arccos(np.clip(singular, 0.0, 1.0))
    projection_fine = fine_q @ fine_q.T
    projection_coarse = coarse_q @ coarse_q.T
    distance = float(np.linalg.norm(projection_fine - projection_coarse, ord="fro") / np.sqrt(2))
    return SubspaceComparison(angles, distance, rank)


@dataclass(frozen=True)
class EigenvalueError:
    pairs: tuple[tuple[int, int], ...]
    errors: tuple[float, ...]
    max_abs_error: float
    rule: str


def slow_eigenvalue_error(fine_eigenvalues: np.ndarray, coarse_eigenvalues: np.ndarray) -> EigenvalueError:
    """Pair sorted non-Perron eigenvalues by rank; truncate to the shorter list."""
    fine = np.sort(np.asarray(fine_eigenvalues, dtype=np.float64))[::-1]
    coarse = np.sort(np.asarray(coarse_eigenvalues, dtype=np.float64))[::-1]
    fine_nontrivial = fine[1:]
    coarse_nontrivial = coarse[1:]
    count = min(fine_nontrivial.size, coarse_nontrivial.size)
    pairs = tuple((index, index) for index in range(count))
    errors = tuple(float(abs(fine_nontrivial[index] - coarse_nontrivial[index])) for index in range(count))
    return EigenvalueError(pairs, errors, max(errors, default=0.0), "sorted_nontrivial_rank_pairing_truncated_to_coarse")


def trajectory_relative_error(fine_trajectories: np.ndarray, lifted_coarse_trajectories: np.ndarray) -> float:
    """Relative Frobenius error ``||X - L Y||_F / ||X||_F`` over the batch."""
    fine = np.asarray(fine_trajectories, dtype=np.float64)
    lifted = np.asarray(lifted_coarse_trajectories, dtype=np.float64)
    if fine.shape != lifted.shape:
        raise ValueError("fine and lifted trajectories must have identical shapes")
    denominator = float(np.linalg.norm(fine))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(lifted) == 0.0 else float("inf")
    return float(np.linalg.norm(fine - lifted) / denominator)


@dataclass(frozen=True)
class OneStepResult:
    """Per-method typed M2 evidence record."""

    method: str
    fine_node_count: int
    coarse_node_count: int
    requested_reduction: float
    achieved_reduction: float
    requested_merges: int
    achieved_merges: int
    shortfall_reason: str | None
    subspace_projection_distance: float | None
    principal_angles: tuple[float, ...]
    slow_eigenvalue_max_abs_error: float | None
    eigenvalue_pairing_rule: str
    trajectory_relative_errors: tuple[float, ...]
    trajectory_labels: tuple[str, ...]
    mean_trajectory_relative_error: float | None
    coarse_spectral_method: str
    infeasible_metrics: tuple[str, ...]
    runtime_seconds: float
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "fine_node_count": self.fine_node_count,
            "coarse_node_count": self.coarse_node_count,
            "compression_ratio": self.fine_node_count / self.coarse_node_count if self.coarse_node_count else float("inf"),
            "requested_reduction": self.requested_reduction,
            "achieved_reduction": self.achieved_reduction,
            "requested_merges": self.requested_merges,
            "achieved_merges": self.achieved_merges,
            "shortfall_reason": self.shortfall_reason,
            "subspace_projection_distance": self.subspace_projection_distance,
            "principal_angles": list(self.principal_angles),
            "slow_eigenvalue_max_abs_error": self.slow_eigenvalue_max_abs_error,
            "eigenvalue_pairing_rule": self.eigenvalue_pairing_rule,
            "trajectory_relative_errors": list(self.trajectory_relative_errors),
            "trajectory_labels": list(self.trajectory_labels),
            "mean_trajectory_relative_error": self.mean_trajectory_relative_error,
            "coarse_spectral_method": self.coarse_spectral_method,
            "infeasible_metrics": list(self.infeasible_metrics),
            "runtime_seconds": self.runtime_seconds,
            "caveats": list(self.caveats),
        }


def _dense_tiny_spectrum(operator: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Documented tiny-graph dense reference for coarse graphs below ``eigsh`` scale.

    Only the already-quotient (small) matrix is densified; the fine graph never is.
    """
    values, vectors = np.linalg.eigh(operator.toarray().astype(np.float64))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def evaluate_one_step(
    fine_adjacency: sparse.spmatrix,
    fine_modes: "ModeResult",  # noqa: F821 - imported lazily to avoid a cycle
    partition: "PartitionResult",  # noqa: F821
    quotient: "QuotientResult",  # noqa: F821
    *,
    fine_trajectories: np.ndarray,
    trajectory_labels: tuple[str, ...],
    time_grid: np.ndarray,
    alpha: float,
    beta: float,
    max_r: int,
    symmetry_tolerance: float,
) -> OneStepResult:
    """Recompute coarse spectral diagnostics and lifted fine/coarse distortion."""
    import time as _time

    from .dynamics import run_linear_dynamics
    from .modes import SpectralConfigurationError, analyze_normalized_adjacency
    from .operators import normalized_adjacency

    started = _time.perf_counter()
    infeasible: list[str] = []
    caveats = [
        "Coordinates are candidate slow-mode coordinates, not proven order parameters.",
        "Distortion compares lifted coarse dynamics with the fine full-system reference; it is distinct from M1 rank-r reconstruction error.",
    ]
    membership = membership_matrix(partition.fine_to_coarse)
    coarse_operator, coarse_degrees = normalized_adjacency(quotient.adjacency, symmetry_tolerance=symmetry_tolerance)
    coarse_count = coarse_operator.shape[0]
    spectral_method = "scipy.sparse.linalg.eigsh"
    if coarse_count >= 3:
        try:
            coarse_modes = analyze_normalized_adjacency(
                coarse_operator, degrees=coarse_degrees, alpha=alpha, beta=beta, top_k=max(2, min(max_r + 1, coarse_count - 1)), max_r=max_r,
                compute=None,
            )
        except SpectralConfigurationError:
            values, vectors = _dense_tiny_spectrum(coarse_operator)
            coarse_modes = _synthetic_mode_result(values, vectors, alpha, beta, max_r)
            spectral_method = "dense_numpy_eigh_tiny_graph_reference"
            caveats.append("Coarse graph is below iterative-solver scale; a documented dense tiny-graph reference was used for the quotient only.")
    else:
        values, vectors = _dense_tiny_spectrum(coarse_operator)
        coarse_modes = _synthetic_mode_result(values, vectors, alpha, beta, max_r)
        spectral_method = "dense_numpy_eigh_tiny_graph_reference"
        caveats.append("Coarse graph has fewer than three nodes; subspace comparison truncates accordingly.")

    fine_r = min(fine_modes.selection.selected_r, fine_modes.eigenvectors.shape[1] - 1)
    coarse_r = min(coarse_modes.selection.selected_r, coarse_modes.eigenvectors.shape[1] - 1)
    fine_basis = fine_modes.eigenvectors[:, 1 : 1 + fine_r]
    lifted_coarse_basis = np.asarray(membership @ coarse_modes.eigenvectors[:, 1 : 1 + coarse_r])
    if fine_r >= 1 and coarse_r >= 1:
        comparison = slow_subspace_comparison(fine_basis, lifted_coarse_basis)
        subspace_distance: float | None = comparison.projection_distance
        angles = tuple(float(angle) for angle in comparison.principal_angles)
    else:
        subspace_distance = None
        angles = ()
        infeasible.append("slow_subspace_distance")
    eigen_errors = slow_eigenvalue_error(fine_modes.eigenvalues, coarse_modes.eigenvalues)
    eigenvalue_error: float | None = None if not eigen_errors.pairs else eigen_errors.max_abs_error

    per_trajectory: list[float] = []
    if fine_trajectories is not None and fine_trajectories.size and fine_trajectories.ndim == 3 and fine_trajectories.shape[2] == partition.fine_to_coarse.size:
        restricted = np.asarray([restrict_state(state, membership, quotient.masses) for state in fine_trajectories[:, 0, :]])
        # The coarse batch is tiny by construction, so full-array storage is safe here.
        coarse_run = run_linear_dynamics(
            coarse_operator, coarse_modes, initial_states=restricted, time_grid=time_grid, labels=trajectory_labels, storage_policy="all", compute=None,
        )
        coarse_trajs = coarse_run.trajectories
        for index in range(fine_trajectories.shape[0]):
            lifted = np.asarray([lift_state(row, membership, quotient.masses) for row in coarse_trajs[index]])
            per_trajectory.append(trajectory_relative_error(fine_trajectories[index], lifted))
    else:
        infeasible.append("trajectory_relative_error")
    mean_error = float(np.mean(per_trajectory)) if per_trajectory else None
    runtime = _time.perf_counter() - started
    return OneStepResult(
        method=partition.method,
        fine_node_count=partition.fine_to_coarse.size,
        coarse_node_count=coarse_count,
        requested_reduction=partition.requested_reduction,
        achieved_reduction=partition.achieved_reduction,
        requested_merges=partition.requested_merges,
        achieved_merges=partition.achieved_merges,
        shortfall_reason=partition.shortfall_reason,
        subspace_projection_distance=subspace_distance,
        principal_angles=angles,
        slow_eigenvalue_max_abs_error=eigenvalue_error,
        eigenvalue_pairing_rule=eigen_errors.rule,
        trajectory_relative_errors=tuple(per_trajectory),
        trajectory_labels=trajectory_labels,
        mean_trajectory_relative_error=mean_error,
        coarse_spectral_method=spectral_method,
        infeasible_metrics=tuple(infeasible),
        runtime_seconds=runtime,
        caveats=tuple(caveats),
    )


def _synthetic_mode_result(values: np.ndarray, vectors: np.ndarray, alpha: float, beta: float, max_r: int) -> "ModeResult":  # noqa: F821
    from .modes import BetaSelection, DimensionSelection, ModeResult, select_slow_mode_dimension

    rates = -alpha + beta * values
    times = np.where(rates < 0, -1.0 / rates, np.inf)
    selection = select_slow_mode_dimension(eigenvalues=values[1:], growth_rates=rates[1:], max_r=max_r)
    beta_selection = BetaSelection(alpha, float("nan"), float("nan"), beta, beta, -alpha + beta, float("nan"), False, ())
    return ModeResult(values, vectors, np.zeros_like(values), rates, times, np.zeros_like(values), np.zeros_like(values), np.zeros_like(values), np.zeros_like(values), np.abs(np.diff(values)), np.zeros(max(len(values) - 1, 0)), beta_selection, selection, {"method": "dense_numpy_eigh_tiny_graph_reference"})


def save_m2_result(
    result: OneStepResult,
    partition: "PartitionResult",  # noqa: F821
    quotient: "QuotientResult",  # noqa: F821
    embedding: "HakenEmbedding",  # noqa: F821
    directory: str | Path,
    *,
    resolved_config: dict,
    input_checksums: dict[str, str],
    execution_telemetry: dict,
) -> dict[str, Path]:
    """Atomically persist quotient CSR, reversible mapping, partition, and metrics."""
    import hashlib
    import json
    import os
    import shutil as _shutil
    import tempfile as _tempfile
    from dataclasses import asdict as _asdict

    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(_tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    quotient_path = staging / "quotient.npz"
    mapping_path = staging / "mapping.json"
    metrics_path = staging / "metrics.json"
    embedding_path = staging / "embedding_metadata.json"
    sparse.save_npz(quotient_path, quotient.adjacency)
    mapping_payload = {
        "fine_to_coarse": partition.fine_to_coarse.tolist(),
        "clusters": [list(cluster) for cluster in partition.clusters],
        "merges": [{"left": int(left), "right": int(right), "haken_distance": float(distance)} for left, right, distance in partition.merges],
        "parent_child": {str(parent): [quotient.fine_node_ids[node] for node in members] for parent, members in quotient.parent_child.items()},
        "supernode_sizes": list(quotient.supernode_sizes),
        "masses": quotient.masses.tolist(),
        "aggregation": quotient.aggregation,
        "provenance": quotient.provenance,
    }
    mapping_path.write_text(json.dumps(mapping_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    embedding_payload = {
        "schema_version": 1,
        "selected_mode_indices": list(embedding.selected_mode_indices),
        "weighting": embedding.weighting,
        "shape": list(embedding.coordinates.shape),
        "caveats": list(embedding.caveats),
    }
    embedding_path.write_text(json.dumps(embedding_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    metrics_payload = {
        "schema_version": 1,
        "result": result.to_dict(),
        "resolved_config": resolved_config,
        "input_checksums": input_checksums,
        "execution": execution_telemetry.get("execution", {}),
        "artifacts_checksums": {"quotient.npz": digest(quotient_path), "mapping.json": digest(mapping_path), "embedding_metadata.json": digest(embedding_path)},
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backup = destination.with_name(f".{destination.name}.previous")
    if destination.exists():
        _shutil.rmtree(backup, ignore_errors=True)
        os.replace(destination, backup)
    os.replace(staging, destination)
    _shutil.rmtree(backup, ignore_errors=True)
    return {
        "quotient": destination / quotient_path.name,
        "mapping": destination / mapping_path.name,
        "metrics": destination / metrics_path.name,
        "embedding": destination / embedding_path.name,
    }
