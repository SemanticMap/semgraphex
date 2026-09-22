"""Fixed-hierarchy comparison of linear and nonlinear semantic dynamics.

The hierarchy is treated as experimental data: graph contraction is not rerun
when the dynamics law changes.  This isolates the effect of the dynamics model
from path dependence in recursive coarsening.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .dynamics import build_perturbations
from .modes import analyze_normalized_adjacency
from .multiscale_config import DynamicsComparisonOptions, DynamicsModel
from .operators import normalized_adjacency


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(reference - candidate))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _restrict_batch(states: np.ndarray, fine_to_coarse: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    assignment = np.asarray(fine_to_coarse, dtype=np.int64)
    coarse_count = int(assignment.max()) + 1
    counts = np.bincount(assignment, minlength=coarse_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("fine_to_coarse contains an empty coarse cluster")
    result = np.empty((states.shape[0], coarse_count), dtype=np.float64)
    for index, state in enumerate(states):
        result[index] = np.bincount(assignment, weights=state, minlength=coarse_count) / counts
    return result


def _rhs(
    state: np.ndarray,
    operator: sparse.csr_matrix,
    *,
    model: DynamicsModel,
    alpha: float,
    beta: float,
    cubic_g: float,
) -> np.ndarray:
    if model == "cubic_haken":
        coupling = (operator @ state.T).T
        return -alpha * state + beta * coupling - cubic_g * state**3
    if model == "tanh":
        coupling = (operator @ np.tanh(state).T).T
        return -alpha * state + beta * coupling
    raise ValueError(f"nonlinear rhs does not support model={model!r}")


def propagate_model(
    operator: sparse.spmatrix,
    initial_states: np.ndarray,
    time_grid: np.ndarray,
    *,
    model: DynamicsModel,
    alpha: float,
    beta: float,
    cubic_g: float,
    nonlinear_max_step: float,
) -> np.ndarray:
    """Return trajectories with shape ``(batch, time, node)``.

    Linear dynamics use ``expm_multiply`` exactly for the sparse Jacobian.
    Nonlinear models use deterministic fixed-step RK4 with a configurable
    maximum step, preserving the requested output time grid exactly.
    """
    matrix = operator.tocsr().astype(np.float64, copy=False)
    states = np.asarray(initial_states, dtype=np.float64)
    times = np.asarray(time_grid, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != matrix.shape[0]:
        raise ValueError("initial_states must have shape (batch, node_count)")
    if times.ndim != 1 or times.size < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_grid must start at zero and be strictly increasing")
    if model == "linear":
        jacobian = (-alpha * sparse.eye(matrix.shape[0], format="csr") + beta * matrix).tocsr()
        propagated = np.asarray(
            expm_multiply(
                jacobian,
                states.T,
                start=float(times[0]),
                stop=float(times[-1]),
                num=times.size,
                endpoint=True,
            )
        )
        result = np.transpose(propagated, (2, 0, 1))
    else:
        if nonlinear_max_step <= 0 or cubic_g <= 0:
            raise ValueError("nonlinear integration parameters must be positive")
        state = states.copy()
        output = [state.copy()]
        for start, stop in zip(times[:-1], times[1:], strict=True):
            interval = float(stop - start)
            substeps = max(1, int(np.ceil(interval / nonlinear_max_step)))
            step = interval / substeps
            for _ in range(substeps):
                k1 = _rhs(state, matrix, model=model, alpha=alpha, beta=beta, cubic_g=cubic_g)
                k2 = _rhs(state + 0.5 * step * k1, matrix, model=model, alpha=alpha, beta=beta, cubic_g=cubic_g)
                k3 = _rhs(state + 0.5 * step * k2, matrix, model=model, alpha=alpha, beta=beta, cubic_g=cubic_g)
                k4 = _rhs(state + step * k3, matrix, model=model, alpha=alpha, beta=beta, cubic_g=cubic_g)
                state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                if not np.all(np.isfinite(state)):
                    raise FloatingPointError(f"{model} dynamics became non-finite")
            output.append(state.copy())
        result = np.stack(output, axis=1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{model} dynamics returned non-finite values")
    return result


def _load_membership(path: Path) -> dict[int, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): list(value) for key, value in raw.items()}


def reconstruct_fine_to_coarse(fine_membership_path: Path, coarse_membership_path: Path) -> np.ndarray:
    """Recover one adjacent hierarchy partition from cumulative memberships."""
    fine = _load_membership(fine_membership_path)
    coarse = _load_membership(coarse_membership_path)
    owner: dict[str, int] = {}
    for coarse_index, originals in coarse.items():
        for original in originals:
            owner[original] = coarse_index
    assignment = np.empty(len(fine), dtype=np.int64)
    for fine_index in range(len(fine)):
        originals = fine[fine_index]
        if not originals:
            raise ValueError("empty fine hierarchy cluster")
        coarse_index = owner[originals[0]]
        if any(owner[original] != coarse_index for original in originals):
            raise ValueError("saved hierarchy memberships are not nested")
        assignment[fine_index] = coarse_index
    return assignment


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _select_levels(levels: Sequence[dict[str, Any]], targets: Sequence[int]) -> tuple[list[tuple[int, list[int]]], list[int]]:
    usable = list(levels[:-1])
    if not usable:
        raise ValueError("hierarchy must contain at least one transition")
    final_nodes = int(levels[-1]["node_count"])
    by_level: dict[int, list[int]] = {}
    unavailable: list[int] = []
    for target in targets:
        if target < final_nodes:
            unavailable.append(int(target))
            continue
        selected = min(usable, key=lambda item: abs(int(item["node_count"]) - int(target)))
        by_level.setdefault(int(selected["level"]), []).append(int(target))
    return sorted(by_level.items()), unavailable


def materialize_dynamics_checkpoints(
    hierarchy_run: str | Path,
    options: DynamicsComparisonOptions,
) -> dict[str, Any]:
    """Create self-contained adjacent-scale checkpoints from a completed hierarchy."""
    run_dir = Path(hierarchy_run).expanduser().resolve()
    hierarchy_dir = run_dir / "hierarchy"
    summary_path = hierarchy_dir / "hierarchy.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing hierarchy summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    levels = summary["levels"]
    selected, unavailable = _select_levels(levels, options.node_targets)
    root = run_dir / "dynamics_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict[str, Any]] = []
    for level, requested_targets in selected:
        fine = levels[level]
        coarse = levels[level + 1]
        name = f"level_{level:03d}_N{int(fine['node_count'])}"
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        fine_adj_src = hierarchy_dir / f"level_{level:03d}_adjacency.npz"
        coarse_adj_src = hierarchy_dir / f"level_{level + 1:03d}_adjacency.npz"
        fine_mem = hierarchy_dir / f"level_{level:03d}_membership.json"
        coarse_mem = hierarchy_dir / f"level_{level + 1:03d}_membership.json"
        for path in (fine_adj_src, coarse_adj_src, fine_mem, coarse_mem):
            if not path.exists():
                raise FileNotFoundError(f"missing hierarchy artifact: {path}")
        _link_or_copy(fine_adj_src, directory / "fine_adjacency.npz")
        _link_or_copy(coarse_adj_src, directory / "coarse_adjacency.npz")
        mapping = reconstruct_fine_to_coarse(fine_mem, coarse_mem)
        np.save(directory / "fine_to_coarse.npy", mapping, allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "level": level,
            "requested_targets": requested_targets,
            "fine_node_count": int(fine["node_count"]),
            "coarse_node_count": int(coarse["node_count"]),
            "selected_r": int(fine["selected_r"]),
            "beta": float(fine["beta"]),
            "perturbation_seed": int(fine["perturbation_seed"]),
            "source_hierarchy_run": str(run_dir),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checkpoints.append({"name": name, "directory": str(directory), **metadata})
    index = {
        "schema_version": 1,
        "source_hierarchy_run": str(run_dir),
        "checkpoints": checkpoints,
        "unavailable_targets": unavailable,
    }
    (root / "checkpoint_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def compare_checkpoint_dynamics(
    hierarchy_run: str | Path,
    *,
    config: Any,
    options: DynamicsComparisonOptions,
) -> dict[str, Any]:
    """Compare all configured laws on the same saved adjacent graph scales."""
    run_dir = Path(hierarchy_run).expanduser().resolve()
    root = run_dir / "dynamics_checkpoints"
    index_path = root / "checkpoint_index.json"
    if not index_path.exists():
        materialize_dynamics_checkpoints(run_dir, options)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    checkpoints = index["checkpoints"]
    time_grid = np.linspace(config.dynamics.time_start, config.dynamics.time_stop, config.dynamics.time_steps)
    rows: list[dict[str, Any]] = []
    total = max(1, len(checkpoints) * len(options.models))
    completed = 0
    started_all = time.perf_counter()
    for checkpoint in checkpoints:
        directory = Path(checkpoint["directory"])
        fine_adjacency = sparse.load_npz(directory / "fine_adjacency.npz").tocsr()
        coarse_adjacency = sparse.load_npz(directory / "coarse_adjacency.npz").tocsr()
        mapping = np.load(directory / "fine_to_coarse.npy", allow_pickle=False)
        fine_operator, fine_degrees = normalized_adjacency(
            fine_adjacency, symmetry_tolerance=config.spectral.symmetry_tolerance
        )
        coarse_operator, _ = normalized_adjacency(
            coarse_adjacency, symmetry_tolerance=config.spectral.symmetry_tolerance
        )
        modes = analyze_normalized_adjacency(
            fine_operator,
            degrees=fine_degrees,
            alpha=config.dynamics.alpha,
            beta=config.dynamics.beta,
            margin=config.dynamics.auto_critical_margin,
            top_k=min(config.spectral.top_k, max(2, fine_operator.shape[0] - 1)),
            max_r=min(config.spectral.max_r, max(1, fine_operator.shape[0] - 2)),
            tolerance=config.spectral.tolerance,
            maxiter=config.spectral.maxiter,
            compute=None,
        )
        beta = float(modes.beta_selection.beta)
        alpha = float(modes.beta_selection.alpha)
        seed = int(checkpoint["perturbation_seed"])
        batch = build_perturbations(
            node_count=fine_operator.shape[0],
            degrees=fine_degrees,
            seed=seed,
            amplitude=config.dynamics.perturbation_amplitude,
            per_kind=config.dynamics.perturbations_per_kind,
            sparse_fraction=config.dynamics.random_sparse_fraction,
        )
        coarse_initial = _restrict_batch(batch.initial_states, mapping)
        selected_r = min(modes.selection.selected_r, modes.eigenvectors.shape[1] - 1)
        basis = modes.eigenvectors[:, 1 : 1 + selected_r]
        post_start = min(time_grid.size - 1, int(np.floor(options.post_transient_fraction * (time_grid.size - 1))))
        for model in options.models:
            started = time.perf_counter()
            fine_traj = propagate_model(
                fine_operator,
                batch.initial_states,
                time_grid,
                model=model,
                alpha=alpha,
                beta=beta,
                cubic_g=options.cubic_g,
                nonlinear_max_step=options.nonlinear_max_step,
            )
            coarse_traj = propagate_model(
                coarse_operator,
                coarse_initial,
                time_grid,
                model=model,
                alpha=alpha,
                beta=beta,
                cubic_g=options.cubic_g,
                nonlinear_max_step=options.nonlinear_max_step,
            )
            lifted = coarse_traj[:, :, mapping]
            full_error = _relative_error(fine_traj, lifted)
            post_error = _relative_error(fine_traj[:, post_start:, :], lifted[:, post_start:, :])
            fine_modal = np.tensordot(fine_traj, basis, axes=([2], [0]))
            lifted_modal = np.tensordot(lifted, basis, axes=([2], [0]))
            slow_error = _relative_error(fine_modal, lifted_modal)
            full_energy = float(np.linalg.norm(fine_traj) ** 2)
            modal_energy = float(np.linalg.norm(fine_modal) ** 2)
            rank_r_reconstruction_error = (
                float(np.sqrt(max(0.0, full_energy - modal_energy) / full_energy)) if full_energy else 0.0
            )
            rows.append(
                {
                    "level": int(checkpoint["level"]),
                    "fine_node_count": int(checkpoint["fine_node_count"]),
                    "coarse_node_count": int(checkpoint["coarse_node_count"]),
                    "requested_targets": list(checkpoint["requested_targets"]),
                    "model": model,
                    "selected_r": int(selected_r),
                    "alpha": alpha,
                    "beta": beta,
                    "cubic_g": options.cubic_g if model == "cubic_haken" else None,
                    "full_trajectory_relative_error": full_error,
                    "post_transient_relative_error": post_error,
                    "slow_coordinate_relative_error": slow_error,
                    "rank_r_reconstruction_error": rank_r_reconstruction_error,
                    "max_abs_state": float(np.max(np.abs(fine_traj))),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            completed += 1
            elapsed = time.perf_counter() - started_all
            print(
                f"[dynamics] {completed}/{total} | N={fine_operator.shape[0]:,} | "
                f"model={model} | full={full_error:.4f} | post={post_error:.4f} | slow={slow_error:.4f} | "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    output_dir = run_dir / "dynamics_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "source_hierarchy_run": str(run_dir),
        "models": list(options.models),
        "cubic_g": options.cubic_g,
        "nonlinear_max_step": options.nonlinear_max_step,
        "post_transient_fraction": options.post_transient_fraction,
        "rows": rows,
        "unavailable_targets": index.get("unavailable_targets", []),
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "level",
        "fine_node_count",
        "coarse_node_count",
        "model",
        "selected_r",
        "alpha",
        "beta",
        "cubic_g",
        "full_trajectory_relative_error",
        "post_transient_relative_error",
        "slow_coordinate_relative_error",
        "rank_r_reconstruction_error",
        "max_abs_state",
        "elapsed_seconds",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report
