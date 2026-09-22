"""Validated optional M3/M4/M5 research sections from one experiment YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


HierarchyMethod = Literal["connectivity_matching", "unconstrained_matching", "connectivity_agglomerative"]
DynamicsModel = Literal["linear", "cubic_haken", "tanh"]


@dataclass(frozen=True)
class HierarchyOptions:
    enabled: bool = False
    method: HierarchyMethod = "connectivity_matching"
    max_levels: int = 8
    min_nodes: int = 16


@dataclass(frozen=True)
class PlateauOptions:
    enabled: bool = True
    min_consecutive: int = 3
    r_tolerance: int = 0
    max_subspace_distance: float = 0.35
    max_trajectory_error: float = 0.50
    max_eigenvalue_error: float = 0.20
    min_achieved_reduction: float = 0.10


@dataclass(frozen=True)
class SyntheticOptions:
    macro_blocks: int = 4
    nodes_per_block: int = 16
    internal_weight: float = 1.0
    bridge_weight: float = 0.05
    seed: int = 1729


@dataclass(frozen=True)
class DynamicsComparisonOptions:
    """Compare one or more dynamics laws on one already-built graph hierarchy."""

    enabled: bool = False
    models: tuple[DynamicsModel, ...] = ("linear", "cubic_haken", "tanh")
    cubic_g: float = 1.0
    nonlinear_max_step: float = 0.05
    post_transient_fraction: float = 0.25
    all_levels: bool = False
    node_targets: tuple[int, ...] = (
        33000,
        30000,
        25000,
        24000,
        20000,
        15000,
        10000,
        7500,
        5000,
        3000,
        2000,
        1000,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        50,
    )


@dataclass(frozen=True)
class ResearchExtensions:
    hierarchy: HierarchyOptions
    plateau: PlateauOptions
    synthetic: SyntheticOptions
    dynamics_comparison: DynamicsComparisonOptions


def _mapping(document: dict, key: str) -> dict:
    value = document.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def load_research_extensions(path: str | Path) -> ResearchExtensions:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError("experiment YAML must be a mapping")
    h = _mapping(document, "hierarchy")
    p = _mapping(document, "plateau")
    s = _mapping(document, "synthetic")
    d = _mapping(document, "dynamics_comparison")
    method = h.get("method", "connectivity_matching")
    if method not in {"connectivity_matching", "unconstrained_matching", "connectivity_agglomerative"}:
        raise ValueError("hierarchy.method is unsupported")
    hierarchy = HierarchyOptions(
        enabled=bool(h.get("enabled", False)),
        method=method,
        max_levels=int(h.get("max_levels", 8)),
        min_nodes=int(h.get("min_nodes", 16)),
    )
    if hierarchy.max_levels < 1 or hierarchy.min_nodes < 3:
        raise ValueError("hierarchy.max_levels >= 1 and hierarchy.min_nodes >= 3 are required")
    plateau = PlateauOptions(
        enabled=bool(p.get("enabled", True)),
        min_consecutive=int(p.get("min_consecutive", 3)),
        r_tolerance=int(p.get("r_tolerance", 0)),
        max_subspace_distance=float(p.get("max_subspace_distance", 0.35)),
        max_trajectory_error=float(p.get("max_trajectory_error", 0.50)),
        max_eigenvalue_error=float(p.get("max_eigenvalue_error", 0.20)),
        min_achieved_reduction=float(p.get("min_achieved_reduction", 0.10)),
    )
    if plateau.min_consecutive < 1 or plateau.r_tolerance < 0:
        raise ValueError("plateau consecutive count and r tolerance are invalid")
    if min(plateau.max_subspace_distance, plateau.max_trajectory_error, plateau.max_eigenvalue_error) < 0:
        raise ValueError("plateau error thresholds must be non-negative")
    if not 0 < plateau.min_achieved_reduction <= 0.5:
        raise ValueError("plateau.min_achieved_reduction must be in (0, 0.5]")
    synthetic = SyntheticOptions(
        macro_blocks=int(s.get("macro_blocks", 4)),
        nodes_per_block=int(s.get("nodes_per_block", 16)),
        internal_weight=float(s.get("internal_weight", 1.0)),
        bridge_weight=float(s.get("bridge_weight", 0.05)),
        seed=int(s.get("seed", 1729)),
    )
    if synthetic.macro_blocks < 2 or synthetic.nodes_per_block < 4:
        raise ValueError("synthetic requires at least two blocks with four nodes each")
    if synthetic.internal_weight <= 0 or not 0 < synthetic.bridge_weight < synthetic.internal_weight:
        raise ValueError("synthetic weights must satisfy 0 < bridge_weight < internal_weight")

    models_raw = d.get("models", ["linear", "cubic_haken", "tanh"])
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("dynamics_comparison.models must be a non-empty list")
    allowed_models = {"linear", "cubic_haken", "tanh"}
    if any(model not in allowed_models for model in models_raw) or len(set(models_raw)) != len(models_raw):
        raise ValueError("dynamics_comparison.models must be unique values from linear, cubic_haken, tanh")
    all_levels = d.get("all_levels", False)
    if not isinstance(all_levels, bool):
        raise ValueError("dynamics_comparison.all_levels must be a boolean")
    targets_raw = d.get("node_targets", list(DynamicsComparisonOptions().node_targets))
    if not isinstance(targets_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in targets_raw
    ):
        raise ValueError("dynamics_comparison.node_targets must be a list of integers >= 2")
    if not all_levels and not targets_raw:
        raise ValueError("dynamics_comparison.node_targets must be non-empty unless all_levels=true")
    cubic_g = float(d.get("cubic_g", 1.0))
    nonlinear_max_step = float(d.get("nonlinear_max_step", 0.05))
    post_fraction = float(d.get("post_transient_fraction", 0.25))
    if cubic_g <= 0 or nonlinear_max_step <= 0 or not 0 <= post_fraction < 1:
        raise ValueError("invalid dynamics_comparison nonlinear parameters")
    dynamics_comparison = DynamicsComparisonOptions(
        enabled=bool(d.get("enabled", False)),
        models=tuple(models_raw),
        cubic_g=cubic_g,
        nonlinear_max_step=nonlinear_max_step,
        post_transient_fraction=post_fraction,
        all_levels=all_levels,
        node_targets=tuple(int(value) for value in targets_raw),
    )
    return ResearchExtensions(hierarchy, plateau, synthetic, dynamics_comparison)
