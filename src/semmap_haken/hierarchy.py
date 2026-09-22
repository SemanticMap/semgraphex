"""M3 recursive Haken coarsening on sparse graphs with retained cross-scale evidence."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .coarsen import build_partition
from .dynamics import build_perturbations, run_linear_dynamics
from .haken_embedding import build_haken_embedding
from .metrics import evaluate_one_step
from .modes import analyze_normalized_adjacency
from .multiscale_config import HierarchyOptions
from .operators import normalized_adjacency
from .quotient import build_quotient


_PROGRESS_BAR_WIDTH = 24
_PROGRESS_WINDOW = 5
_REDUCTION_FLOAT_EPSILON = 1e-12


@dataclass(frozen=True)
class LevelEvidence:
    level: int
    node_count: int
    adjacency_nnz: int
    selected_r: int
    beta: float
    spectral_abscissa: float
    max_eigengap: float
    mean_rank_r_reconstruction_error: float
    perturbation_seed: int
    coarsening_seed: int


@dataclass(frozen=True)
class TransitionEvidence:
    source_level: int
    target_level: int
    source_r: int
    target_r: int | None
    fine_node_count: int
    coarse_node_count: int
    compression_ratio: float
    achieved_reduction: float
    subspace_projection_distance: float | None
    slow_eigenvalue_max_abs_error: float | None
    mean_trajectory_relative_error: float | None
    shortfall_reason: str | None
    method: str


@dataclass(frozen=True)
class HierarchyResult:
    levels: tuple[LevelEvidence, ...]
    transitions: tuple[TransitionEvidence, ...]
    adjacencies: tuple[sparse.csr_matrix, ...]
    memberships: tuple[dict[int, tuple[str, ...]], ...]
    stop_reason: str
    method: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": self.method,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "levels": [asdict(item) for item in self.levels],
            "transitions": [asdict(item) for item in self.transitions],
        }


def _compose_membership(
    current: dict[int, tuple[str, ...]],
    parent_child: dict[int, tuple[int, ...]],
) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for coarse, children in parent_child.items():
        flattened: list[str] = []
        for child in children:
            flattened.extend(current[int(child)])
        result[int(coarse)] = tuple(sorted(flattened))
    return result


def _estimate_planned_contractions(
    node_count: int,
    *,
    min_nodes: int,
    max_levels: int,
    target_reduction: float,
) -> int:
    """Estimate contractions using the same floor quantization as coarsening."""
    current = int(node_count)
    planned = 0
    while planned < max_levels and current > min_nodes and current >= 3:
        requested = min(
            current // 2,
            int(np.floor(current * target_reduction + _REDUCTION_FLOAT_EPSILON)),
        )
        if requested <= 0:
            break
        current -= requested
        planned += 1
    return planned


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _progress_bar(fraction: float, width: int = _PROGRESS_BAR_WIDTH) -> str:
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = min(width, int(round(width * fraction)))
    return "█" * filled + "░" * (width - filled)


def run_hierarchy(
    adjacency: sparse.spmatrix,
    node_ids: Sequence[str],
    *,
    config: Any,
    options: HierarchyOptions,
    compute: Any = None,
) -> HierarchyResult:
    """Repeatedly apply the tested M2 contraction and measure every adjacent scale.

    ``max_levels`` is the maximum number of contractions; the returned level
    sequence therefore contains at most ``max_levels + 1`` graph states.
    """
    started = time.perf_counter()
    current = adjacency.tocsr().astype(np.float64, copy=False)
    current_ids = tuple(str(item) for item in node_ids)
    if current.shape[0] != len(current_ids):
        raise ValueError("node_ids must match adjacency")
    current_members = {index: (node_id,) for index, node_id in enumerate(current_ids)}
    levels: list[LevelEvidence] = []
    transitions: list[TransitionEvidence] = []
    adjacencies: list[sparse.csr_matrix] = []
    memberships: list[dict[int, tuple[str, ...]]] = []
    stop_reason = "max_levels"
    time_grid = np.linspace(config.dynamics.time_start, config.dynamics.time_stop, config.dynamics.time_steps)

    planned_contractions = _estimate_planned_contractions(
        current.shape[0],
        min_nodes=options.min_nodes,
        max_levels=options.max_levels,
        target_reduction=config.coarsening.target_reduction,
    )
    progress_durations: list[float] = []
    progress_checkpoint = started
    print(
        "[haken] planned "
        f"{planned_contractions} contractions | "
        f"N={current.shape[0]:,} -> <= {options.min_nodes:,} | "
        f"target_reduction={config.coarsening.target_reduction:.4f}",
        file=sys.stderr,
        flush=True,
    )

    for level in range(options.max_levels + 1):
        if current.shape[0] < 3:
            stop_reason = "graph_below_spectral_minimum"
            break
        operator, degrees = normalized_adjacency(current, symmetry_tolerance=config.spectral.symmetry_tolerance)
        modes = analyze_normalized_adjacency(
            operator,
            degrees=degrees,
            alpha=config.dynamics.alpha,
            beta=config.dynamics.beta,
            margin=config.dynamics.auto_critical_margin,
            top_k=config.spectral.top_k,
            max_r=config.spectral.max_r,
            tolerance=config.spectral.tolerance,
            maxiter=config.spectral.maxiter,
            compute=compute,
        )
        perturbation_seed = int(config.dynamics.perturbation_seed + 1009 * level)
        coarsening_seed = int(config.coarsening.seed + 2003 * level)
        batch = build_perturbations(
            node_count=current.shape[0], degrees=degrees, seed=perturbation_seed,
            amplitude=config.dynamics.perturbation_amplitude,
            per_kind=config.dynamics.perturbations_per_kind,
            sparse_fraction=config.dynamics.random_sparse_fraction,
        )
        dynamics = run_linear_dynamics(
            operator, modes, initial_states=batch.initial_states, labels=batch.labels,
            time_grid=time_grid, storage_policy="all",
            max_storage_bytes=config.dynamics.max_storage_mb * 1024 * 1024,
            compute=compute,
        )
        max_gap = float(np.max(modes.eigengaps)) if modes.eigengaps.size else 0.0
        levels.append(LevelEvidence(
            level=level,
            node_count=int(current.shape[0]),
            adjacency_nnz=int(current.nnz),
            selected_r=int(modes.selection.selected_r),
            beta=float(modes.beta_selection.beta),
            spectral_abscissa=float(modes.beta_selection.spectral_abscissa),
            max_eigengap=max_gap,
            mean_rank_r_reconstruction_error=float(dynamics.aggregate["mean_relative_rmse"]),
            perturbation_seed=perturbation_seed,
            coarsening_seed=coarsening_seed,
        ))
        adjacencies.append(current.copy())
        memberships.append(dict(current_members))

        if level >= options.max_levels:
            stop_reason = "max_levels"
            break
        if current.shape[0] <= options.min_nodes:
            stop_reason = "min_nodes"
            break

        embedding = build_haken_embedding(modes, weighting=config.coarsening.embedding_weighting)
        partition = build_partition(
            current, embedding.coordinates, method=options.method,
            target_reduction=config.coarsening.target_reduction, seed=coarsening_seed,
            distance_threshold=config.coarsening.distance_threshold,
        )
        if partition.achieved_merges == 0:
            stop_reason = partition.shortfall_reason or "no_merges"
            break
        quotient = build_quotient(
            current, partition.fine_to_coarse, aggregation=config.coarsening.aggregation,
            node_ids=current_ids, level=level, parent_run_id=None,
        )
        one_step = evaluate_one_step(
            current, modes, partition, quotient,
            fine_trajectories=dynamics.trajectories,
            trajectory_labels=batch.labels,
            time_grid=time_grid,
            alpha=float(modes.beta_selection.alpha),
            beta=float(modes.beta_selection.beta),
            max_r=config.spectral.max_r,
            symmetry_tolerance=config.spectral.symmetry_tolerance,
        )
        transitions.append(TransitionEvidence(
            source_level=level,
            target_level=level + 1,
            source_r=int(modes.selection.selected_r),
            target_r=None,
            fine_node_count=int(one_step.fine_node_count),
            coarse_node_count=int(one_step.coarse_node_count),
            compression_ratio=float(one_step.fine_node_count / one_step.coarse_node_count),
            achieved_reduction=float(one_step.achieved_reduction),
            subspace_projection_distance=one_step.subspace_projection_distance,
            slow_eigenvalue_max_abs_error=one_step.slow_eigenvalue_max_abs_error,
            mean_trajectory_relative_error=one_step.mean_trajectory_relative_error,
            shortfall_reason=one_step.shortfall_reason,
            method=options.method,
        ))
        current_members = _compose_membership(current_members, quotient.parent_child)
        current = quotient.adjacency.tocsr()
        current_ids = tuple(f"level{level + 1}/c{index}" for index in range(current.shape[0]))

        now = time.perf_counter()
        progress_durations.append(now - progress_checkpoint)
        progress_checkpoint = now
        completed = level + 1
        denominator = max(1, planned_contractions)
        fraction = min(1.0, completed / denominator)
        recent_mean = float(np.mean(progress_durations[-_PROGRESS_WINDOW:]))
        remaining = max(0, planned_contractions - completed)
        eta_seconds = recent_mean * remaining
        elapsed_seconds = now - started
        print(
            f"[haken] [{_progress_bar(fraction)}] "
            f"{fraction * 100:6.2f}% | "
            f"level {completed}/{planned_contractions} | "
            f"N={current.shape[0]:,} | "
            f"step={_format_duration(progress_durations[-1])} | "
            f"elapsed={_format_duration(elapsed_seconds)} | "
            f"ETA={_format_duration(eta_seconds)}",
            file=sys.stderr,
            flush=True,
        )

    by_level = {item.level: item.selected_r for item in levels}
    transitions = [replace(item, target_r=by_level.get(item.target_level)) for item in transitions]
    return HierarchyResult(
        levels=tuple(levels), transitions=tuple(transitions), adjacencies=tuple(adjacencies),
        memberships=tuple(memberships), stop_reason=stop_reason, method=options.method,
        elapsed_seconds=time.perf_counter() - started,
    )


def save_hierarchy_result(result: HierarchyResult, directory: str | Path) -> dict[str, Path]:
    """Atomically persist the full sparse scale trajectory and original-node memberships."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    summary_path = staging / "hierarchy.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for level, (adjacency, membership) in enumerate(zip(result.adjacencies, result.memberships, strict=True)):
        sparse.save_npz(staging / f"level_{level:03d}_adjacency.npz", adjacency)
        (staging / f"level_{level:03d}_membership.json").write_text(
            json.dumps({str(key): list(value) for key, value in membership.items()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    backup = destination.with_name(f".{destination.name}.previous")
    if destination.exists():
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(destination, backup)
    os.replace(staging, destination)
    shutil.rmtree(backup, ignore_errors=True)
    paths = {"summary": destination / "hierarchy.json"}
    for level in range(len(result.adjacencies)):
        paths[f"level_{level:03d}_adjacency"] = destination / f"level_{level:03d}_adjacency.npz"
        paths[f"level_{level:03d}_membership"] = destination / f"level_{level:03d}_membership.json"
    return paths
