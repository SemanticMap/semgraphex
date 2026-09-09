"""M4 plateau detection over completed M3 hierarchy evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .hierarchy import HierarchyResult, TransitionEvidence
from .multiscale_config import PlateauOptions


@dataclass(frozen=True)
class PlateauCandidate:
    start_level: int
    end_level: int
    transition_count: int
    start_nodes: int
    end_nodes: int
    compression_ratio: float
    r_values: tuple[int, ...]
    max_subspace_distance: float
    max_trajectory_error: float
    max_eigenvalue_error: float


@dataclass(frozen=True)
class PlateauReport:
    candidates: tuple[PlateauCandidate, ...]
    qualifying_transitions: tuple[int, ...]
    rejected: tuple[dict[str, object], ...]
    options: PlateauOptions

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "options": asdict(self.options),
            "qualifying_transitions": list(self.qualifying_transitions),
            "candidates": [asdict(item) for item in self.candidates],
            "rejected": list(self.rejected),
        }


def _transition_check(item: TransitionEvidence, options: PlateauOptions) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if item.target_r is None or abs(item.source_r - item.target_r) > options.r_tolerance:
        reasons.append("r_instability")
    if item.achieved_reduction < options.min_achieved_reduction:
        reasons.append("insufficient_reduction")
    if item.subspace_projection_distance is None or item.subspace_projection_distance > options.max_subspace_distance:
        reasons.append("subspace_distortion")
    if item.mean_trajectory_relative_error is None or item.mean_trajectory_relative_error > options.max_trajectory_error:
        reasons.append("trajectory_distortion")
    if item.slow_eigenvalue_max_abs_error is None or item.slow_eigenvalue_max_abs_error > options.max_eigenvalue_error:
        reasons.append("eigenvalue_distortion")
    return not reasons, tuple(reasons)


def detect_plateaus(hierarchy: HierarchyResult, options: PlateauOptions) -> PlateauReport:
    """Return runs of consecutive transitions satisfying all configured gates."""
    qualifying: list[int] = []
    rejected: list[dict[str, object]] = []
    mask: list[bool] = []
    for index, transition in enumerate(hierarchy.transitions):
        passed, reasons = _transition_check(transition, options)
        mask.append(passed)
        if passed:
            qualifying.append(index)
        else:
            rejected.append({"transition_index": index, "source_level": transition.source_level, "target_level": transition.target_level, "reasons": list(reasons)})

    candidates: list[PlateauCandidate] = []
    start: int | None = None
    for index in range(len(mask) + 1):
        active = index < len(mask) and mask[index]
        if active and start is None:
            start = index
        if (not active) and start is not None:
            end = index - 1
            count = end - start + 1
            if count >= options.min_consecutive:
                transitions = hierarchy.transitions[start : end + 1]
                r_values = [transitions[0].source_r] + [item.target_r for item in transitions if item.target_r is not None]
                subspace = [float(item.subspace_projection_distance) for item in transitions if item.subspace_projection_distance is not None]
                trajectory = [float(item.mean_trajectory_relative_error) for item in transitions if item.mean_trajectory_relative_error is not None]
                eigenvalue = [float(item.slow_eigenvalue_max_abs_error) for item in transitions if item.slow_eigenvalue_max_abs_error is not None]
                candidates.append(PlateauCandidate(
                    start_level=transitions[0].source_level,
                    end_level=transitions[-1].target_level,
                    transition_count=count,
                    start_nodes=transitions[0].fine_node_count,
                    end_nodes=transitions[-1].coarse_node_count,
                    compression_ratio=float(transitions[0].fine_node_count / transitions[-1].coarse_node_count),
                    r_values=tuple(int(value) for value in r_values),
                    max_subspace_distance=max(subspace, default=float("nan")),
                    max_trajectory_error=max(trajectory, default=float("nan")),
                    max_eigenvalue_error=max(eigenvalue, default=float("nan")),
                ))
            start = None
    return PlateauReport(tuple(candidates), tuple(qualifying), tuple(rejected), options)


def save_plateau_report(report: PlateauReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
