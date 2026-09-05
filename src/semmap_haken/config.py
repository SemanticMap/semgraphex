"""Typed, single-source YAML configuration contracts for active workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

WeightTransform = Literal["binary", "raw", "log1p", "capped", "relation_normalized"]
OperatorName = Literal["normalized_adjacency", "random_walk", "laplacian", "jacobian"]
ComponentPolicy = Literal["largest", "all"]


class ConfigurationError(ValueError):
    """Raised when a YAML configuration cannot satisfy the foundation contract."""


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _require(value: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in value:
        raise ConfigurationError(f"{section}.{key} is required")
    return value[key]


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class PathsConfig:
    workspace_root: Path
    data_root: Path
    cache_root: Path
    runs_root: Path


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    language: str
    relations: tuple[str, ...]
    min_weight: float
    max_nodes: int
    component: ComponentPolicy
    path: Path | None = None
    version: str | None = None
    source_url: str | None = None
    max_rows: int | None = None


@dataclass(frozen=True)
class GraphConfig:
    directed: bool
    weight_transform: WeightTransform
    operator: OperatorName
    self_loop_policy: Literal["exclude", "include"] = "exclude"


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str
    random_seed: int
    resource_profile: Path | None


@dataclass(frozen=True)
class DynamicsConfig:
    """Linear-Jacobian and A2 trajectory experiment parameters."""

    model: Literal["linear"]
    alpha: float
    beta: float | Literal["auto_critical"]
    auto_critical_margin: float
    time_start: float
    time_stop: float
    time_steps: int
    perturbations_per_kind: int
    perturbation_amplitude: float
    perturbation_seed: int
    random_sparse_fraction: float
    storage_policy: Literal["all", "summaries", "none"]
    max_storage_mb: int


@dataclass(frozen=True)
class SpectralConfig:
    prepared_graph_dir: Path | None
    top_k: int
    max_r: int
    tolerance: float
    maxiter: int | None
    symmetry_tolerance: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully validated config plus its JSON/YAML-safe resolved representation."""

    source_path: Path
    paths: PathsConfig
    dataset: DatasetConfig
    graph: GraphConfig
    runtime: RuntimeConfig
    dynamics: DynamicsConfig
    spectral: SpectralConfig
    resolved: dict[str, Any]


def _paths_from(raw: Mapping[str, Any], source_path: Path) -> PathsConfig:
    base = source_path.parent.resolve()
    workspace_root = _resolve_path(raw.get("workspace_root", "."), base)
    return PathsConfig(
        workspace_root=workspace_root,
        data_root=_resolve_path(raw.get("data_root", "data"), workspace_root),
        cache_root=_resolve_path(raw.get("cache_root", "cache"), workspace_root),
        runs_root=_resolve_path(raw.get("runs_root", "runs"), workspace_root),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, validate, and resolve an Iteration 1 YAML configuration exactly once."""
    source_path = Path(path).resolve()
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read config {source_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {source_path}: {error}") from error
    raw = _require_mapping(document, "config")
    paths = _paths_from(_require_mapping(raw.get("paths", {}), "paths"), source_path)

    dataset_raw = _require_mapping(_require(raw, "dataset", "config"), "dataset")
    relations = _require(dataset_raw, "relations", "dataset")
    if not isinstance(relations, list) or not all(isinstance(item, str) for item in relations):
        raise ConfigurationError("dataset.relations must be a list of strings")
    component = _require(dataset_raw, "component", "dataset")
    if component not in {"largest", "all"}:
        raise ConfigurationError("dataset.component must be 'largest' or 'all'")
    max_rows = dataset_raw.get("max_rows")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0):
        raise ConfigurationError("dataset.max_rows must be a positive integer or null")
    dataset = DatasetConfig(
        source=str(_require(dataset_raw, "source", "dataset")),
        language=str(_require(dataset_raw, "language", "dataset")),
        relations=tuple(relations),
        min_weight=float(_require(dataset_raw, "min_weight", "dataset")),
        max_nodes=int(_require(dataset_raw, "max_nodes", "dataset")),
        component=component,
        path=_resolve_path(dataset_raw["path"], source_path.parent.resolve()) if dataset_raw.get("path") else None,
        version=str(dataset_raw["version"]) if dataset_raw.get("version") else None,
        source_url=str(dataset_raw["source_url"]) if dataset_raw.get("source_url") else None,
        max_rows=max_rows,
    )
    if dataset.min_weight < 0 or dataset.max_nodes <= 0:
        raise ConfigurationError("dataset.min_weight must be non-negative and dataset.max_nodes must be positive")

    graph_raw = _require_mapping(_require(raw, "graph", "config"), "graph")
    weight_transform = _require(graph_raw, "weight_transform", "graph")
    if weight_transform not in {"binary", "raw", "log1p", "capped", "relation_normalized"}:
        raise ConfigurationError("graph.weight_transform is unsupported")
    operator = _require(graph_raw, "operator", "graph")
    if operator not in {"normalized_adjacency", "random_walk", "laplacian", "jacobian"}:
        raise ConfigurationError("graph.operator is unsupported")
    directed = _require(graph_raw, "directed", "graph")
    if not isinstance(directed, bool):
        raise ConfigurationError("graph.directed must be a boolean")
    loop_policy = graph_raw.get("self_loop_policy", "exclude")
    if loop_policy not in {"exclude", "include"}:
        raise ConfigurationError("graph.self_loop_policy must be 'exclude' or 'include'")
    graph = GraphConfig(directed=directed, weight_transform=weight_transform, operator=operator, self_loop_policy=loop_policy)

    runtime_raw = _require_mapping(raw.get("runtime", {}), "runtime")
    resource_profile_value = runtime_raw.get("resource_profile")
    runtime = RuntimeConfig(
        profile=str(runtime_raw.get("profile", "small")),
        random_seed=int(runtime_raw.get("random_seed", 0)),
        resource_profile=(
            _resolve_path(resource_profile_value, source_path.parent.resolve())
            if resource_profile_value is not None
            else None
        ),
    )
    dynamics_raw = _require_mapping(raw.get("dynamics", {}), "dynamics")
    beta_value = dynamics_raw.get("beta", "auto_critical")
    if beta_value != "auto_critical" and (isinstance(beta_value, bool) or not isinstance(beta_value, (int, float))):
        raise ConfigurationError("dynamics.beta must be a number or 'auto_critical'")
    dynamics = DynamicsConfig(
        model=dynamics_raw.get("model", "linear"),
        alpha=float(dynamics_raw.get("alpha", 1.0)),
        beta=beta_value if beta_value == "auto_critical" else float(beta_value),
        auto_critical_margin=float(dynamics_raw.get("auto_critical_margin", 0.05)),
        time_start=float(dynamics_raw.get("time_start", 0.0)),
        time_stop=float(dynamics_raw.get("time_stop", 10.0)),
        time_steps=int(dynamics_raw.get("time_steps", 101)),
        perturbations_per_kind=int(dynamics_raw.get("perturbations_per_kind", 1)),
        perturbation_amplitude=float(dynamics_raw.get("perturbation_amplitude", 1.0)),
        perturbation_seed=int(dynamics_raw.get("perturbation_seed", runtime.random_seed)),
        random_sparse_fraction=float(dynamics_raw.get("random_sparse_fraction", 0.05)),
        storage_policy=dynamics_raw.get("storage_policy", "summaries"),
        max_storage_mb=int(dynamics_raw.get("max_storage_mb", 256)),
    )
    if dynamics.model != "linear" or dynamics.alpha <= 0 or not 0 < dynamics.auto_critical_margin < dynamics.alpha:
        raise ConfigurationError("M1 requires linear dynamics with alpha > margin > 0")
    if dynamics.time_start != 0.0 or dynamics.time_stop <= dynamics.time_start or dynamics.time_steps < 2 or dynamics.perturbations_per_kind < 1 or dynamics.perturbation_amplitude <= 0 or not 0 < dynamics.random_sparse_fraction <= 1 or dynamics.storage_policy not in {"all", "summaries", "none"} or dynamics.max_storage_mb < 1:
        raise ConfigurationError("invalid M1 dynamics time grid, perturbation, or storage configuration")
    spectral_raw = _require_mapping(raw.get("spectral", {}), "spectral")
    top_k, max_r = int(spectral_raw.get("top_k", 64)), int(spectral_raw.get("max_r", 32))
    maxiter = spectral_raw.get("maxiter")
    if top_k < 2 or max_r < 1 or (maxiter is not None and (isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0)):
        raise ConfigurationError("spectral.top_k >= 2, max_r >= 1, and maxiter must be positive or null")
    spectral = SpectralConfig(
        prepared_graph_dir=_resolve_path(spectral_raw["prepared_graph_dir"], source_path.parent.resolve()) if spectral_raw.get("prepared_graph_dir") else None,
        top_k=top_k,
        max_r=max_r,
        tolerance=float(spectral_raw.get("tolerance", 1e-10)),
        maxiter=maxiter,
        symmetry_tolerance=float(spectral_raw.get("symmetry_tolerance", 1e-10)),
    )
    if spectral.tolerance <= 0 or spectral.symmetry_tolerance < 0:
        raise ConfigurationError("spectral tolerances must be positive (symmetry may be zero)")
    resolved = {
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "dataset": {**{key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(dataset).items()}, "relations": list(dataset.relations)},
        "graph": asdict(graph),
        "runtime": {
            "profile": runtime.profile,
            "random_seed": runtime.random_seed,
            "resource_profile": str(runtime.resource_profile) if runtime.resource_profile else None,
        },
        "dynamics": asdict(dynamics),
        "spectral": {**asdict(spectral), "prepared_graph_dir": str(spectral.prepared_graph_dir) if spectral.prepared_graph_dir else None},
    }
    return ExperimentConfig(source_path, paths, dataset, graph, runtime, dynamics, spectral, resolved)
