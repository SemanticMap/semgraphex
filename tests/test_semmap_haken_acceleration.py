from __future__ import annotations

import builtins
import importlib.util
import sys

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.compute import ComputeConfigurationError, ComputeContext
from semmap_haken.config import ConfigurationError, load_config
from semmap_haken.dynamics import run_linear_dynamics
from semmap_haken.modes import analyze_normalized_adjacency
from semmap_haken.operators import normalized_adjacency


def _path_graph(size: int) -> sparse.csr_matrix:
    rows = np.arange(size - 1)
    return sparse.csr_matrix(
        (np.ones(2 * (size - 1)), (np.concatenate((rows, rows + 1)), np.concatenate((rows + 1, rows)))),
        shape=(size, size),
    )


def _context(*, workers: int | str = 1) -> ComputeContext:
    return ComputeContext.create(
        backend="cpu", device=0, dtype="float64", workers=workers,
        reserved_cpu_cores=0, threads_per_worker=1, gpu_memory_fraction=0.8,
        batch_size=2, deterministic=True, allow_auto_fallback=True,
    )


def test_execution_config_defaults_resolve_and_invalid_values_fail(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
dataset: {source: fixture, language: en, relations: [], min_weight: 1.0, max_nodes: 10, component: largest}
graph: {directed: false, weight_transform: raw, operator: normalized_adjacency}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.execution.backend == "auto"
    assert config.execution.workers == "auto"
    assert config.execution.batch_size == "auto"
    assert config.execution.deterministic is True
    assert config.resolved["execution"]["dtype"] == "float64"

    config_path.write_text(config_path.read_text(encoding="utf-8") + "\nexecution: {backend: cuda, workers: 0}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="execution.workers"):
        load_config(config_path)


def test_cpu_context_never_imports_cupy_and_emits_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("cupy"):
            raise AssertionError("CPU context must not import CuPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    context = _context()
    telemetry = context.telemetry()
    assert context.backend == "cpu"
    assert telemetry["cpu"]["logical_count"] >= 1
    assert telemetry["execution"]["workers"] == 1
    assert "cupy" not in sys.modules


def test_strict_cuda_fails_without_usable_cuda() -> None:
    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("strict unavailable-CUDA behavior is only deterministic on a CPU-only environment")
    with pytest.raises(ComputeConfigurationError):
        ComputeContext.create(
            backend="cuda", device=0, dtype="float64", workers="auto",
            reserved_cpu_cores=0, threads_per_worker=1, gpu_memory_fraction=0.8,
            batch_size="auto", deterministic=True, allow_auto_fallback=True,
        )


def test_serial_and_multicore_cpu_dynamics_are_deterministically_equivalent() -> None:
    operator, degrees = normalized_adjacency(_path_graph(8))
    modes = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=5)
    states = np.eye(8)[:4]
    first = run_linear_dynamics(operator, modes, initial_states=states, time_grid=np.linspace(0.0, 1.0, 5), storage_policy="all", compute=_context(workers=1))
    second = run_linear_dynamics(operator, modes, initial_states=states, time_grid=np.linspace(0.0, 1.0, 5), storage_policy="all", compute=_context(workers=2))
    assert np.allclose(first.trajectories, second.trajectories, atol=1e-12)
    assert first.aggregate["execution"]["backend"] == "cpu"
    assert second.aggregate["execution"]["workers"] == 2


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CUDA parity requires CuPy")
def test_cuda_eigenspace_and_trajectory_parity_when_available() -> None:
    operator, degrees = normalized_adjacency(_path_graph(8))
    cpu = _context()
    try:
        cuda = ComputeContext.create(
            backend="cuda", device=0, dtype="float64", workers=1,
            reserved_cpu_cores=0, threads_per_worker=1, gpu_memory_fraction=0.8,
            batch_size=2, deterministic=True, allow_auto_fallback=False,
        )
    except ComputeConfigurationError as error:
        pytest.skip(f"CUDA is unavailable: {error}")
    reference = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=5, compute=cpu)
    accelerated = analyze_normalized_adjacency(operator, degrees=degrees, alpha=1.0, beta=0.5, top_k=5, compute=cuda)
    assert np.allclose(reference.eigenvalues, accelerated.eigenvalues, atol=1e-8)
    assert np.allclose(reference.eigenvectors @ reference.eigenvectors.T, accelerated.eigenvectors @ accelerated.eigenvectors.T, atol=1e-6)
