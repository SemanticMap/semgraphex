"""Iterative sparse diagnostics for A1 slow-mode candidates.

The results are diagnostics only: they do not establish graph order parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from .compute import ComputeContext


class SpectralConfigurationError(ValueError):
    """Raised when A1 spectral analysis cannot make a valid diagnostic."""


@dataclass(frozen=True)
class BetaSelection:
    alpha: float
    margin: float
    target_eigenvalue: float
    requested_beta: float
    beta: float
    spectral_abscissa: float
    target_growth_rate: float
    clipped_for_stability: bool
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class DimensionSelection:
    selected_r: int
    eigengap_r: int | None
    timescale_gap_r: int | None
    consensus: Literal["agreement", "fallback_eigengap", "fallback_timescale"]
    candidate_r: tuple[int, ...]
    bootstrap_deferred: bool = True


@dataclass(frozen=True)
class ModeResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residual_norms: np.ndarray
    growth_rates: np.ndarray
    relaxation_times: np.ndarray
    inverse_participation_ratios: np.ndarray
    participation_ratios: np.ndarray
    localization_scores: np.ndarray
    degree_correlations: np.ndarray
    eigengaps: np.ndarray
    timescale_gaps: np.ndarray
    beta_selection: BetaSelection
    selection: DimensionSelection
    solver: dict[str, Any]


def _eligible_nontrivial(eigenvalues: np.ndarray, *, tolerance: float = 1e-8) -> np.ndarray:
    values = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    return values[np.abs(values - 1.0) > tolerance]


def select_auto_critical_beta(eigenvalues: np.ndarray, *, alpha: float, margin: float) -> BetaSelection:
    """Select a deterministic stable beta for ``J=-alpha I+beta S``.

    The first non-Perron eigenvalue is the target.  In the full state space,
    exact targeting is often unstable because the Perron eigenvalue is one;
    beta is therefore clipped at ``alpha-margin``.  The artifact records this
    limitation instead of silently claiming a critical nontrivial mode.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise SpectralConfigurationError("alpha must be finite and positive")
    if not np.isfinite(margin) or not 0 < margin < alpha:
        raise SpectralConfigurationError("margin must be finite, positive, and smaller than alpha")
    eligible = _eligible_nontrivial(eigenvalues)
    positive = eligible[eligible > 1e-8]
    if positive.size == 0:
        raise SpectralConfigurationError("auto_critical requires an eligible nontrivial positive eigenvalue")
    target = float(positive[0])
    requested = (alpha - margin) / target
    # Normalized adjacency has spectral radius <= 1. Stability over all modes
    # requires beta <= alpha-margin when beta is non-negative.
    beta = min(requested, alpha - margin)
    spectral_abscissa = -alpha + beta
    clipped = beta < requested
    caveats = (
        "The Perron/stationary normalized-adjacency mode is excluded from slow-mode candidates.",
        "Full-state stability constrains beta; exact nontrivial margin targeting may be unavailable without a projected dynamics model.",
    ) if clipped else ("The Perron/stationary normalized-adjacency mode is excluded from slow-mode candidates.",)
    return BetaSelection(alpha, margin, target, requested, beta, spectral_abscissa, -alpha + beta * target, clipped, caveats)


def _largest_eigenpairs(operator: sparse.csr_matrix, top_k: int, tolerance: float, maxiter: int | None, compute: ComputeContext | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = operator.shape[0]
    if n < 3:
        raise SpectralConfigurationError("iterative eigsh diagnostics require a graph with at least three nodes")
    k = min(top_k, n - 1)
    if k < 2:
        raise SpectralConfigurationError("top_k must allow a trivial and an eligible nontrivial mode")
    started = time.perf_counter()
    backend = compute.backend if compute else "cpu"
    try:
        if backend == "cuda":
            import cupy as cp
            from cupyx.scipy import sparse as cupyx_sparse
            from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

            with cp.cuda.Device(compute.device):
                values_device, vectors_device = cupy_eigsh(cupyx_sparse.csr_matrix(operator.astype(compute.dtype, copy=False)), k=k, which="LA", tol=tolerance, maxiter=maxiter)
                values, vectors = cp.asnumpy(values_device), cp.asnumpy(vectors_device)
        else:
            values, vectors = eigsh(operator.astype(compute.dtype if compute else np.float64, copy=False), k=k, which="LA", tol=tolerance, maxiter=maxiter)
    except (ArpackNoConvergence, Exception) as error:
        prefix = "CUDA eigsh" if backend == "cuda" else "eigsh"
        raise SpectralConfigurationError(f"{prefix} did not converge: {error}") from error
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    residuals = np.linalg.norm(operator @ vectors - vectors * values, axis=0)
    return values, vectors, {"method": "cupyx.scipy.sparse.linalg.eigsh" if backend == "cuda" else "scipy.sparse.linalg.eigsh", "backend": backend, "which": "LA", "requested_k": top_k, "computed_k": k, "tolerance": tolerance, "maxiter": maxiter, "converged": bool(np.all(np.isfinite(residuals))), "elapsed_seconds": time.perf_counter() - started, "residual_norms": residuals.tolist(), "execution": compute.telemetry() if compute else None}


def _gap_choice(values: np.ndarray) -> int | None:
    if len(values) < 2:
        return None
    gaps = np.abs(np.diff(values))
    return int(np.argmax(gaps) + 1)


def select_slow_mode_dimension(*, eigenvalues: np.ndarray, growth_rates: np.ndarray, max_r: int) -> DimensionSelection:
    """Choose a diagnostic slow-mode candidate count from two independent gaps."""
    count = min(len(eigenvalues), len(growth_rates), max_r)
    if count < 2:
        return DimensionSelection(1, None, None, "fallback_eigengap", (1,))
    values = np.asarray(eigenvalues[:count], dtype=float)
    rates = np.asarray(growth_rates[:count], dtype=float)
    eigengap_r = _gap_choice(values)
    stable_times = np.where(rates < 0, -1.0 / rates, np.inf)
    finite_times = np.where(np.isfinite(stable_times), stable_times, np.finfo(float).max)
    timescale_gap_r = _gap_choice(np.log1p(finite_times))
    candidates = tuple(sorted({item for item in (eigengap_r, timescale_gap_r) if item is not None}))
    if eigengap_r == timescale_gap_r and eigengap_r is not None:
        return DimensionSelection(eigengap_r, eigengap_r, timescale_gap_r, "agreement", candidates)
    if eigengap_r is not None:
        return DimensionSelection(eigengap_r, eigengap_r, timescale_gap_r, "fallback_eigengap", candidates)
    return DimensionSelection(timescale_gap_r or 1, eigengap_r, timescale_gap_r, "fallback_timescale", candidates)


def analyze_normalized_adjacency(
    operator: sparse.spmatrix,
    *,
    degrees: np.ndarray,
    alpha: float,
    beta: float | Literal["auto_critical"],
    top_k: int,
    margin: float = 0.05,
    max_r: int = 32,
    tolerance: float = 1e-10,
    maxiter: int | None = None,
    compute: ComputeContext | None = None,
) -> ModeResult:
    """Compute sparse normalized-adjacency diagnostics for slow-mode candidates."""
    matrix = operator.tocsr()
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(degrees):
        raise SpectralConfigurationError("operator and degree vector shapes must agree")
    if not np.all(np.isfinite(matrix.data)):
        raise SpectralConfigurationError("operator values must be finite")
    if beta == "auto_critical":
        if np.any(np.asarray(degrees) <= 0):
            raise SpectralConfigurationError("auto_critical requires a connected graph without zero-degree nodes")
        component_count, _ = connected_components(matrix, directed=False, return_labels=True)
        if component_count != 1:
            raise SpectralConfigurationError("auto_critical requires a connected graph without zero-degree nodes")
    values, vectors, solver = _largest_eigenpairs(matrix, top_k, tolerance, maxiter, compute)
    beta_selection = select_auto_critical_beta(values, alpha=alpha, margin=margin) if beta == "auto_critical" else BetaSelection(alpha, margin, float("nan"), float(beta), float(beta), float(-alpha + float(beta)), float("nan"), False, ())
    rates = -alpha + beta_selection.beta * values
    times = np.where(rates < 0, -1.0 / rates, np.inf)
    ipr = np.sum(vectors**4, axis=0)
    participation = 1.0 / ipr
    localization = ipr * matrix.shape[0]
    correlations = np.array([
        np.corrcoef(np.abs(vectors[:, index]), degrees)[0, 1] if np.std(degrees) > 0 else np.nan
        for index in range(vectors.shape[1])
    ])
    eigengaps = np.abs(np.diff(values))
    finite_times = np.where(np.isfinite(times), times, np.finfo(float).max)
    timescale_gaps = np.abs(np.diff(np.log1p(finite_times)))
    # Index zero is the Perron mode for connected nonnegative input; its vector
    # is retained for solver provenance but excluded from the candidate count.
    selection = select_slow_mode_dimension(eigenvalues=values[1:], growth_rates=rates[1:], max_r=max_r)
    return ModeResult(values, vectors, np.asarray(solver["residual_norms"]), rates, times, ipr, participation, localization, correlations, eigengaps, timescale_gaps, beta_selection, selection, solver)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_mode_result(result: ModeResult, directory: str | Path, *, resolved_config: dict[str, Any], input_checksums: dict[str, str]) -> dict[str, Path]:
    """Atomically persist numeric A1 diagnostics and JSON provenance."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    numeric = staging / "spectral_modes.npz"
    diagnostics = staging / "spectral_diagnostics.json"
    np.savez_compressed(numeric, eigenvalues=result.eigenvalues, eigenvectors=result.eigenvectors, residual_norms=result.residual_norms, growth_rates=result.growth_rates, relaxation_times=result.relaxation_times, inverse_participation_ratios=result.inverse_participation_ratios, participation_ratios=result.participation_ratios, localization_scores=result.localization_scores, degree_correlations=result.degree_correlations, eigengaps=result.eigengaps, timescale_gaps=result.timescale_gaps)
    diagnostics.write_text(json.dumps({"schema_version": 1, "beta_selection": asdict(result.beta_selection), "selection": asdict(result.selection), "solver": result.solver, "resolved_config": resolved_config, "input_checksums": input_checksums, "numeric_checksum_sha256": _sha256(numeric)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backup = destination.with_name(f".{destination.name}.previous")
    if destination.exists():
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(destination, backup)
    os.replace(staging, destination)
    shutil.rmtree(backup, ignore_errors=True)
    return {"numeric": destination / numeric.name, "diagnostics": destination / diagnostics.name}


def load_mode_result(directory: str | Path) -> ModeResult:
    """Load and checksum-verify a persisted A1 spectral artifact."""
    source = Path(directory)
    diagnostics = json.loads((source / "spectral_diagnostics.json").read_text(encoding="utf-8"))
    numeric_path = source / "spectral_modes.npz"
    if _sha256(numeric_path) != diagnostics["numeric_checksum_sha256"]:
        raise ValueError("spectral numeric artifact checksum mismatch")
    arrays = np.load(numeric_path)
    return ModeResult(*(arrays[name] for name in ("eigenvalues", "eigenvectors", "residual_norms", "growth_rates", "relaxation_times", "inverse_participation_ratios", "participation_ratios", "localization_scores", "degree_correlations", "eigengaps", "timescale_gaps")), BetaSelection(**diagnostics["beta_selection"]), DimensionSelection(**diagnostics["selection"]), diagnostics["solver"])
