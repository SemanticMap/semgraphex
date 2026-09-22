"""Thin, offline-safe helpers for Colab and local Jupyter workflows.

The core detects Colab only from already-loaded modules.  Importing or mounting
``google.colab`` is deliberately isolated behind :func:`mount_drive`.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import yaml

from .config import PathsConfig, load_config

GIB = 1024**3


class NotebookPreflightError(RuntimeError):
    """Raised before a notebook attempts work beyond its resource budget."""


@dataclass(frozen=True)
class NotebookEnvironment:
    kind: str
    python_version: str
    platform: str


@dataclass(frozen=True)
class ResourcePreflight:
    profile_path: Path | None
    required_ram_bytes: int
    required_disk_bytes: int
    available_ram_bytes: int
    available_disk_bytes: int
    ok: bool


@dataclass(frozen=True)
class PersistenceRecord:
    source: Path
    destination: Path
    size_bytes: int


def detect_environment(*, modules: Mapping[str, ModuleType | object] | None = None) -> NotebookEnvironment:
    """Classify the current host without importing notebook- or Colab-only code."""
    loaded = sys.modules if modules is None else modules
    if "google.colab" in loaded:
        kind = "colab"
    elif "IPython" in loaded:
        kind = "local_jupyter"
    else:
        kind = "cli"
    return NotebookEnvironment(kind, platform.python_version(), platform.platform())


def resolve_notebook_paths(
    config_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> PathsConfig:
    """Resolve notebook storage roots from shared config with explicit overrides."""
    source = Path(config_path)
    if source.is_dir():
        base = source.resolve()
        defaults = PathsConfig(base, base / "data", base / "cache", base / "runs")
    else:
        defaults = load_config(source).paths
    selected_workspace = Path(workspace_root).expanduser().resolve() if workspace_root else defaults.workspace_root

    def resolve_override(value: str | Path | None, default: Path) -> Path:
        if value is None:
            return default
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (selected_workspace / path).resolve()

    return PathsConfig(
        workspace_root=selected_workspace,
        data_root=resolve_override(data_root, defaults.data_root),
        cache_root=resolve_override(cache_root, defaults.cache_root),
        runs_root=resolve_override(runs_root, defaults.runs_root),
    )


def available_memory_bytes() -> int:
    """Return usable host RAM where discoverable, otherwise zero (safe fail)."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return 0


def available_disk_bytes(path: str | Path) -> int:
    return shutil.disk_usage(Path(path)).free


def run_resource_preflight(profile_path: str | Path | None, storage_root: str | Path) -> ResourcePreflight:
    """Validate the configured RAM and disk recommendation before an expensive run."""
    raw: Mapping[str, Any] = {}
    profile = Path(profile_path).resolve() if profile_path else None
    if profile is not None:
        document = yaml.safe_load(profile.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise NotebookPreflightError(f"resource profile {profile} must be a mapping")
        raw = document
    required_ram = int(float(raw.get("recommended_ram_gib", 0)) * GIB)
    required_disk = int(float(raw.get("recommended_disk_gib", 0)) * GIB)
    free_ram = available_memory_bytes()
    root = Path(storage_root)
    root.mkdir(parents=True, exist_ok=True)
    free_disk = available_disk_bytes(root)
    result = ResourcePreflight(profile, required_ram, required_disk, free_ram, free_disk, free_ram >= required_ram and free_disk >= required_disk)
    if free_ram < required_ram:
        raise NotebookPreflightError(f"insufficient RAM: need {required_ram} bytes, found {free_ram}")
    if free_disk < required_disk:
        raise NotebookPreflightError(f"insufficient disk: need {required_disk} bytes, found {free_disk}")
    return result


def mount_drive(*, enabled: bool, mount_callback: Callable[[], Any] | None = None) -> Any | None:
    """Mount Drive only when a notebook explicitly provides a Colab-only callback."""
    if not enabled:
        return None
    if mount_callback is None:
        raise NotebookPreflightError("Drive use requires an explicit notebook mount callback")
    return mount_callback()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, temporary)
    try:
        shutil.copystat(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def persist_paths(paths: Sequence[str | Path], *, source_root: str | Path, destination_root: str | Path) -> list[PersistenceRecord]:
    """Atomically copy selected local artifacts to durable storage after execution."""
    source_base = Path(source_root).resolve()
    destination_base = Path(destination_root).resolve()
    records: list[PersistenceRecord] = []
    for item in paths:
        source = Path(item).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"cannot persist missing file {source}")
        try:
            relative = source.relative_to(source_base)
        except ValueError as error:
            raise ValueError(f"persistence source {source} is outside {source_base}") from error
        destination = destination_base / relative
        _atomic_copy(source, destination)
        records.append(PersistenceRecord(source, destination, destination.stat().st_size))
    return records


def environment_snapshot(environment: NotebookEnvironment | None = None) -> dict[str, Any]:
    detected = environment or detect_environment()
    return {"environment": asdict(detected), "executable": sys.executable, "cwd": str(Path.cwd())}


def make_execution_metadata(
    *,
    notebook_name: str,
    environment: NotebookEnvironment,
    paths: PathsConfig,
    drive_enabled: bool,
    persisted_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Return JSON-safe manifest metadata for notebook-executed package work."""
    return {
        "notebook_name": notebook_name,
        "environment_snapshot": environment_snapshot(environment),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "drive": {"enabled": drive_enabled, "mounting_is_explicit": True},
        "persistence": {"mode": "active_local_then_atomic_copy", "paths": [str(Path(path)) for path in persisted_paths]},
    }
