"""Lazy hardware selection and execution telemetry for sparse experiments.

The CPU path deliberately has no CuPy import.  CUDA imports occur only after a
CUDA backend is requested, so a base installation remains a CPU-only package.
"""

from __future__ import annotations

import importlib
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

Backend = Literal["auto", "cpu", "cuda"]
DTypeName = Literal["float64", "float32"]
AutoInt = int | Literal["auto"]


class ComputeConfigurationError(RuntimeError):
    """Raised when the requested execution backend cannot be used safely."""


def _physical_cpu_count() -> int | None:
    """Return a best-effort physical-core count without adding a dependency."""
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


@dataclass(frozen=True)
class ComputeContext:
    """Selected sparse execution backend plus reproducibility telemetry."""

    requested_backend: Backend
    backend: Literal["cpu", "cuda"]
    device: int
    dtype_name: DTypeName
    workers: int
    reserved_cpu_cores: int
    threads_per_worker: int
    gpu_memory_fraction: float
    batch_size: AutoInt
    deterministic: bool
    allow_auto_fallback: bool
    logical_cpu_count: int
    physical_cpu_count: int | None
    fallback_reason: str | None = None
    gpu: dict[str, Any] | None = None
    cupy_version: str | None = None
    cuda_version: str | None = None

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype_name)

    @classmethod
    def create(
        cls,
        *,
        backend: Backend,
        device: int,
        dtype: DTypeName,
        workers: AutoInt,
        reserved_cpu_cores: int,
        threads_per_worker: int,
        gpu_memory_fraction: float,
        batch_size: AutoInt,
        deterministic: bool,
        allow_auto_fallback: bool,
    ) -> "ComputeContext":
        if backend not in {"auto", "cpu", "cuda"}:
            raise ComputeConfigurationError("backend must be auto, cpu, or cuda")
        if dtype not in {"float64", "float32"}:
            raise ComputeConfigurationError("dtype must be float64 or float32")
        if device < 0 or reserved_cpu_cores < 0 or threads_per_worker < 1:
            raise ComputeConfigurationError("device/reserved_cpu_cores/threads_per_worker are invalid")
        if not 0 < gpu_memory_fraction <= 1:
            raise ComputeConfigurationError("gpu_memory_fraction must be in (0, 1]")
        if workers != "auto" and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
            raise ComputeConfigurationError("workers must be auto or a positive integer")
        if batch_size != "auto" and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ComputeConfigurationError("batch_size must be auto or a positive integer")
        logical = os.cpu_count() or 1
        physical = _physical_cpu_count()
        chosen_workers = max(1, logical - reserved_cpu_cores) if workers == "auto" else workers
        if backend == "cpu":
            return cls(backend, "cpu", device, dtype, chosen_workers, reserved_cpu_cores, threads_per_worker, gpu_memory_fraction, batch_size, deterministic, allow_auto_fallback, logical, physical)
        try:
            cupy = importlib.import_module("cupy")
            device_count = int(cupy.cuda.runtime.getDeviceCount())
            if device >= device_count:
                raise ComputeConfigurationError(f"CUDA device {device} is unavailable (visible devices: {device_count})")
            with cupy.cuda.Device(device):
                properties = cupy.cuda.runtime.getDeviceProperties(device)
                free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
            name = properties["name"].decode() if isinstance(properties["name"], bytes) else str(properties["name"])
            gpu = {
                "name": name,
                "compute_capability": f"{properties['major']}.{properties['minor']}",
                "total_vram_bytes": int(total_bytes),
                "free_vram_bytes": int(free_bytes),
                "memory_budget_bytes": int(free_bytes * gpu_memory_fraction),
            }
            cuda_runtime = cupy.cuda.runtime.runtimeGetVersion()
            return cls(backend, "cuda", device, dtype, chosen_workers, reserved_cpu_cores, threads_per_worker, gpu_memory_fraction, batch_size, deterministic, allow_auto_fallback, logical, physical, gpu=gpu, cupy_version=str(cupy.__version__), cuda_version=str(cuda_runtime))
        except (ImportError, ComputeConfigurationError, Exception) as error:
            if backend == "cuda":
                raise ComputeConfigurationError(f"strict CUDA backend is unavailable: {error}") from error
            if not allow_auto_fallback:
                raise ComputeConfigurationError(f"automatic CUDA selection failed and fallback is disabled: {error}") from error
            return cls(backend, "cpu", device, dtype, chosen_workers, reserved_cpu_cores, threads_per_worker, gpu_memory_fraction, batch_size, deterministic, allow_auto_fallback, logical, physical, fallback_reason=str(error))

    def telemetry(self) -> dict[str, Any]:
        """Return JSON-safe selection and hardware metadata for every artifact."""
        return {
            "execution": {
                "requested_backend": self.requested_backend,
                "backend": self.backend,
                "device": self.device,
                "dtype": self.dtype_name,
                "workers": self.workers,
                "threads_per_worker": self.threads_per_worker,
                "batch_size": self.batch_size,
                "deterministic": self.deterministic,
                "allow_auto_fallback": self.allow_auto_fallback,
                "fallback_reason": self.fallback_reason,
            },
            "cpu": {"logical_count": self.logical_cpu_count, "physical_count": self.physical_cpu_count, "reserved_cores": self.reserved_cpu_cores},
            "gpu": self.gpu,
            "software": {"cupy_version": self.cupy_version, "cuda_runtime_version": self.cuda_version, "python": platform.python_version()},
        }
