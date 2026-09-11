"""Validated optional M3/M4 sections from the same experiment YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


HierarchyMethod = Literal["connectivity_matching", "unconstrained_matching", "connectivity_agglomerative"]


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
class ResearchExtensions:
    hierarchy: HierarchyOptions
    plateau: PlateauOptions
    synthetic: SyntheticOptions


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
    return ResearchExtensions(hierarchy, plateau, synthetic)
