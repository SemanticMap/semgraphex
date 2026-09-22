"""Sparse-safe linear trajectory diagnostics for the M1 evidence increment.

The implementation evolves ``dx/dt = (-alpha I + beta S)x`` with
``scipy.sparse.linalg.expm_multiply``.  It deliberately never constructs a
dense matrix exponential or a dense representation of the sparse operator.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .modes import ModeResult
from .compute import ComputeContext

StoragePolicy = Literal["all", "summaries", "none"]


class DynamicsConfigurationError(ValueError):
    """Raised when a requested linear trajectory experiment is invalid."""


@dataclass(frozen=True)
class PerturbationBatch:
    """Deterministic initial states and their reproducible labels."""

    initial_states: np.ndarray
    labels: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class DynamicsResult:
    """Numerical outputs for an M1 linear perturbation experiment."""

    time_grid: np.ndarray
    labels: tuple[str, ...]
    initial_states: np.ndarray
    modal_amplitudes: np.ndarray
    reconstructed_trajectories: np.ndarray | None
    trajectories: np.ndarray | None
    relative_rmse: np.ndarray
    nrmse: np.ndarray
    selected_r: int
    storage_policy: StoragePolicy
    aggregate: dict[str, Any]


def build_perturbations(
    *,
    node_count: int,
    degrees: np.ndarray,
    seed: int,
    amplitude: float,
    per_kind: int,
    sparse_fraction: float,
) -> PerturbationBatch:
    """Build four seeded first-evidence perturbation families.

    Local-neighborhood and relation-group perturbations are intentionally
    deferred: the prepared CSR artifact does not expose a cheap semantic group
    index, and adding one would expand this increment's scope.
    """
    if node_count < 1 or degrees.shape != (node_count,):
        raise DynamicsConfigurationError("degrees must have one finite value per node")
    if not np.all(np.isfinite(degrees)) or np.any(degrees < 0):
        raise DynamicsConfigurationError("degrees must be finite and non-negative")
    if not np.isfinite(amplitude) or amplitude <= 0 or per_kind < 1 or not 0 < sparse_fraction <= 1:
        raise DynamicsConfigurationError("amplitude, per_kind, and sparse_fraction are invalid")
    generator = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    labels: list[str] = []
    hub = int(np.argmax(degrees))
    active_count = max(1, int(np.ceil(node_count * sparse_fraction)))
    for index in range(per_kind):
        state = np.zeros(node_count)
        state[index % node_count] = amplitude
        states.append(state)
        labels.append(f"single_node-{index}")
    for index in range(per_kind):
        states.append(generator.normal(0.0, amplitude, node_count))
        labels.append(f"gaussian-{index}")
    for index in range(per_kind):
        state = np.zeros(node_count)
        active = generator.choice(node_count, size=active_count, replace=False)
        state[active] = generator.normal(0.0, amplitude, active_count)
        states.append(state)
        labels.append(f"random_sparse-{index}")
    for index in range(per_kind):
        state = np.zeros(node_count)
        state[hub] = amplitude
        states.append(state)
        labels.append(f"hub_targeted-{index}")
    return PerturbationBatch(np.asarray(states), tuple(labels), seed)


def _validate_inputs(operator: sparse.spmatrix, modes: ModeResult, initial_states: np.ndarray, time_grid: np.ndarray, max_storage_bytes: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    matrix = operator.tocsr().astype(np.float64, copy=False)
    times = np.asarray(time_grid, dtype=np.float64)
    states = np.asarray(initial_states, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1] or modes.eigenvectors.shape[0] != matrix.shape[0]:
        raise DynamicsConfigurationError("operator and spectral modes must have matching square dimensions")
    if states.ndim != 2 or states.shape[1] != matrix.shape[0] or not np.all(np.isfinite(states)):
        raise DynamicsConfigurationError("initial states must be a finite two-dimensional node array")
    if times.ndim != 1 or len(times) < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0) or not np.all(np.isfinite(times)):
        raise DynamicsConfigurationError("time grid must be finite, start at zero, and strictly increase with at least two values")
    if max_storage_bytes <= 0:
        raise DynamicsConfigurationError("max_storage_bytes must be positive")
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(modes.growth_rates)):
        raise DynamicsConfigurationError("operator and mode rates must be finite")
    return matrix, states, times


def _cpu_trajectory_chunk(jacobian: sparse.csr_matrix, states: np.ndarray, times: np.ndarray, threads_per_worker: int) -> np.ndarray:
    """Propagate a stable input-order chunk using SciPy's reference method."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        propagated = np.asarray(expm_multiply(jacobian, states.T, start=float(times[0]), stop=float(times[-1]), num=times.size, endpoint=True))
    else:
        with threadpool_limits(limits=threads_per_worker):
            propagated = np.asarray(expm_multiply(jacobian, states.T, start=float(times[0]), stop=float(times[-1]), num=times.size, endpoint=True))
    return np.transpose(propagated, (2, 0, 1))


def _cuda_trajectory_chunk(jacobian: sparse.csr_matrix, states: np.ndarray, times: np.ndarray, compute: ComputeContext) -> np.ndarray:
    """Use device-resident, fixed-step RK4; this is an accuracy-controlled approximation.

    CuPy currently exposes sparse eigensolvers but no sparse ``expm_multiply``
    equivalent.  The step size is bounded from a sparse infinity-norm estimate
    and the returned summary records this non-exact CUDA propagation method.
    """
    import cupy as cp
    from cupyx.scipy import sparse as cupyx_sparse

    with cp.cuda.Device(compute.device):
        matrix = cupyx_sparse.csr_matrix(jacobian.astype(compute.dtype, copy=False))
        state = cp.asarray(states.T, dtype=compute.dtype)
        norm_bound = float(np.max(np.asarray(np.abs(jacobian).sum(axis=1)).ravel()))
        output: list[np.ndarray] = [cp.asnumpy(state.T)]
        for start, stop in zip(times[:-1], times[1:], strict=True):
            interval = float(stop - start)
            substeps = max(1, int(np.ceil(interval * max(norm_bound, 1.0) / 0.02)))
            step = interval / substeps
            for _ in range(substeps):
                k1 = matrix @ state
                k2 = matrix @ (state + 0.5 * step * k1)
                k3 = matrix @ (state + 0.5 * step * k2)
                k4 = matrix @ (state + step * k3)
                state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            output.append(cp.asnumpy(state.T))
    return np.stack(output, axis=1)


def run_linear_dynamics(
    operator: sparse.spmatrix,
    modes: ModeResult,
    *,
    initial_states: np.ndarray,
    time_grid: np.ndarray,
    labels: tuple[str, ...] | None = None,
    storage_policy: StoragePolicy = "summaries",
    max_storage_bytes: int = 256 * 1024 * 1024,
    compute: ComputeContext | None = None,
) -> DynamicsResult:
    """Evolve sparse linear dynamics and reconstruct trajectories from A1 candidates.

    Reconstruction uses the orthogonal projection onto the selected nontrivial
    eigenvectors.  Relative RMSE is ``||x - x_hat|| / ||x||`` over every time
    and node sample; NRMSE divides RMSE by the trajectory's max absolute value.
    """
    if storage_policy not in {"all", "summaries", "none"}:
        raise DynamicsConfigurationError("storage_policy must be all, summaries, or none")
    matrix, states, times = _validate_inputs(operator, modes, initial_states, time_grid, max_storage_bytes)
    compute = compute or ComputeContext.create(backend="cpu", device=0, dtype="float64", workers=1, reserved_cpu_cores=0, threads_per_worker=1, gpu_memory_fraction=0.8, batch_size="auto", deterministic=True, allow_auto_fallback=True)
    labels = labels or tuple(f"trajectory-{index}" for index in range(len(states)))
    if len(labels) != len(states):
        raise DynamicsConfigurationError("labels must have one value per initial state")
    selected_r = min(modes.selection.selected_r, modes.eigenvectors.shape[1] - 1)
    if selected_r < 1:
        raise DynamicsConfigurationError("spectral analysis did not provide a nontrivial slow-mode candidate")
    basis = modes.eigenvectors[:, 1 : 1 + selected_r]
    jacobian = (-modes.beta_selection.alpha * sparse.eye(matrix.shape[0], format="csr") + modes.beta_selection.beta * matrix).tocsr()
    estimated_bytes = states.shape[0] * times.size * matrix.shape[0] * np.dtype(np.float64).itemsize
    if storage_policy == "all" and 2 * estimated_bytes > max_storage_bytes:
        raise DynamicsConfigurationError("full trajectory storage exceeds configured memory limit")
    started = time.perf_counter()
    full: list[np.ndarray] = []
    amplitudes: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    relative: list[float] = []
    normalized: list[float] = []
    chunk_size = len(states) if compute.batch_size == "auto" else min(compute.batch_size, len(states))
    chunks = [(index, states[index : index + chunk_size]) for index in range(0, len(states), chunk_size)]
    if compute.backend == "cuda":
        trajectories = [_cuda_trajectory_chunk(jacobian, chunk, times, compute) for _, chunk in chunks]
        propagation_method = "cuda_rk4_sparse_accuracy_controlled"
    elif compute.workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=min(compute.workers, len(chunks))) as pool:
            completed = [(index, pool.submit(_cpu_trajectory_chunk, jacobian, chunk, times, compute.threads_per_worker)) for index, chunk in chunks]
            trajectories = [future.result() for _, future in sorted(completed, key=lambda item: item[0])]
        propagation_method = "scipy_expm_multiply_process_chunks"
    else:
        trajectories = [_cpu_trajectory_chunk(jacobian, chunk, times, compute.threads_per_worker) for _, chunk in chunks]
        propagation_method = "scipy_expm_multiply_vectorized"
    for trajectory in np.concatenate(trajectories, axis=0):
        if trajectory.shape != (times.size, matrix.shape[0]) or not np.all(np.isfinite(trajectory)):
            raise DynamicsConfigurationError("linear solver returned non-finite or malformed trajectory")
        modal = trajectory @ basis
        reconstruction = modal @ basis.T
        error = trajectory - reconstruction
        rmse = float(np.sqrt(np.mean(error**2)))
        scale = float(np.sqrt(np.mean(trajectory**2)))
        relative.append(rmse / scale if scale else 0.0)
        peak = float(np.max(np.abs(trajectory)))
        normalized.append(rmse / peak if peak else 0.0)
        amplitudes.append(modal)
        if storage_policy == "all":
            full.append(trajectory)
            reconstructions.append(reconstruction)
    relative_array, normalized_array = np.asarray(relative), np.asarray(normalized)
    aggregate: dict[str, Any] = {
        "trajectory_count": len(states),
        "time_count": int(times.size),
        "node_count": int(matrix.shape[0]),
        "selected_r": selected_r,
        "mean_relative_rmse": float(np.mean(relative_array)),
        "max_relative_rmse": float(np.max(relative_array)),
        "mean_nrmse": float(np.mean(normalized_array)),
        "spectral_abscissa": float(modes.beta_selection.spectral_abscissa),
        "propagation_method": propagation_method,
        "elapsed_seconds": time.perf_counter() - started,
        "throughput_trajectories_per_second": len(states) / max(time.perf_counter() - started, np.finfo(float).eps),
        "execution": compute.telemetry()["execution"],
        "batch_size": chunk_size,
    }
    return DynamicsResult(times, tuple(labels), states, np.asarray(amplitudes), np.asarray(reconstructions) if full else None, np.asarray(full) if full else None, relative_array, normalized_array, selected_r, storage_policy, aggregate)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_dynamics_result(result: DynamicsResult, directory: str | Path, *, resolved_config: dict[str, Any], input_artifacts: dict[str, str]) -> dict[str, Path]:
    """Atomically persist compressed arrays plus machine-readable summaries."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    numeric, summary, table = staging / "dynamics_arrays.npz", staging / "dynamics_summary.json", staging / "trajectory_reconstruction.csv"
    payload: dict[str, np.ndarray] = {"time_grid": result.time_grid, "initial_states": result.initial_states, "modal_amplitudes": result.modal_amplitudes, "relative_rmse": result.relative_rmse, "nrmse": result.nrmse}
    if result.trajectories is not None:
        payload["trajectories"] = result.trajectories
        payload["reconstructed_trajectories"] = result.reconstructed_trajectories
    np.savez_compressed(numeric, **payload)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "relative_rmse", "nrmse", "selected_r"])
        writer.writeheader()
        writer.writerows({"label": label, "relative_rmse": relative, "nrmse": nrmse, "selected_r": result.selected_r} for label, relative, nrmse in zip(result.labels, result.relative_rmse, result.nrmse, strict=True))
    summary.write_text(json.dumps({"schema_version": 1, "aggregate": result.aggregate, "labels": list(result.labels), "storage_policy": result.storage_policy, "resolved_config": resolved_config, "input_artifacts": input_artifacts, "numeric_checksum_sha256": _sha256(numeric)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backup = destination.with_name(f".{destination.name}.previous")
    if destination.exists():
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(destination, backup)
    os.replace(staging, destination)
    shutil.rmtree(backup, ignore_errors=True)
    return {"numeric": destination / numeric.name, "summary": destination / summary.name, "table": destination / table.name}


def save_m1_plots(directory: str | Path, modes: ModeResult, result: DynamicsResult) -> dict[str, Path]:
    """Write lightweight PNG evidence plots when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "spectrum": destination / "spectrum.png",
        "timescales": destination / "relaxation_times.png",
        "eigengaps": destination / "eigengaps.png",
        "reconstruction": destination / "trajectory_reconstruction.png",
    }
    fig, axis = plt.subplots()
    axis.plot(np.arange(len(modes.eigenvalues)), modes.eigenvalues, marker="o")
    axis.set(xlabel="mode rank", ylabel="eigenvalue", title="Normalized-adjacency spectrum")
    fig.savefig(paths["spectrum"], dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots()
    axis.plot(np.arange(len(modes.relaxation_times)), modes.relaxation_times, marker="o")
    axis.set_yscale("log"); axis.set(xlabel="mode rank", ylabel="relaxation time", title="Linear relaxation times")
    fig.savefig(paths["timescales"], dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots()
    axis.plot(np.arange(len(modes.eigengaps)), modes.eigengaps, marker="o")
    axis.set(xlabel="nontrivial mode rank", ylabel="eigenvalue gap", title="Nontrivial eigengaps")
    fig.savefig(paths["eigengaps"], dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots()
    axis.bar(np.arange(len(result.relative_rmse)), result.relative_rmse)
    axis.set(xlabel="trajectory", ylabel="relative RMSE", title="Slow-subspace reconstruction")
    fig.savefig(paths["reconstruction"], dpi=120, bbox_inches="tight"); plt.close(fig)
    return paths
