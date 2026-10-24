"""Sparse directed typed block intensities; optional batched CuPy acceleration."""
from __future__ import annotations

import numpy as np

from .graph import TypedGraph


def estimate_block_kernel(
    graph: TypedGraph, partition: np.ndarray, *, device: str = "auto",
    batch_size: int = 250_000, original_mass: np.ndarray | None = None,
) -> dict[str, object]:
    """Finite empirical step kernel, NOT a fitted W/S/I graphex limit.

    Only present block pairs are materialized. Missing pairs have zero observed
    counts. Intensities use original-node exposure, not uniform macro-node mass.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    labels = np.asarray(partition)
    if labels.shape != (graph.vertex_count,):
        raise ValueError("partition shape mismatch")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("partition requires nonnegative integer labels")
    _, block = np.unique(labels, return_inverse=True)
    b = int(block.max()) + 1 if block.size else 0
    mass = np.ones(graph.vertex_count) if original_mass is None else np.asarray(original_mass, dtype=float)
    if mass.shape != (graph.vertex_count,) or np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("original_mass requires positive finite values per vertex")
    block_mass = np.bincount(block, weights=mass, minlength=b)
    if len(graph.relation_names) * b * b >= np.iinfo(np.int64).max:
        raise ValueError("block-key range exceeds int64")

    cp = None
    fallback_reason = None
    if device != "cpu":
        try:
            import cupy as cp_module
            if cp_module.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("CUDA device is not available")
            cp = cp_module
            # Bound the per-batch workspace against available GPU memory.
            free, _ = cp.cuda.runtime.memGetInfo()
            batch_size = min(batch_size, max(1, int(free * 0.3 // 96)))
        except Exception as error:
            if device == "cuda":
                raise RuntimeError(f"CUDA backend unavailable: {error}") from error
            fallback_reason = str(error)
            cp = None
    xp = np if cp is None else cp
    keys: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for begin in range(0, graph.edge_count, batch_size):
        end = min(begin + batch_size, graph.edge_count)
        s = xp.asarray(block[graph.source[begin:end]], dtype=xp.int64)
        t = xp.asarray(block[graph.target[begin:end]], dtype=xp.int64)
        r = xp.asarray(graph.relation[begin:end], dtype=xp.int64)
        code = (r * b + s) * b + t
        unique, inverse, freq = xp.unique(code, return_inverse=True, return_counts=True)
        sums = xp.bincount(inverse, weights=xp.asarray(graph.weight[begin:end]), minlength=len(unique))
        convert = (lambda x: np.asarray(x)) if cp is None else cp.asnumpy
        keys.append(convert(unique))
        counts.append(convert(freq))
        weights.append(convert(sums))
    if keys:
        codes = np.concatenate(keys)
        uniq, inverse = np.unique(codes, return_inverse=True)
        edge_counts = np.bincount(inverse, weights=np.concatenate(counts))
        edge_weights = np.bincount(inverse, weights=np.concatenate(weights))
    else:
        uniq = np.array([], dtype=np.int64)
        edge_counts = edge_weights = np.array([], dtype=float)
    has_loops = bool(np.any(graph.source == graph.target))
    entries = []
    for code, count, total_weight in zip(uniq, edge_counts, edge_weights):
        dst = int(code % b)
        src = int((code // b) % b)
        relation = int(code // (b * b))
        possible = float(block_mass[src] * block_mass[dst])
        if not has_loops and src == dst:
            possible -= float(block_mass[src])
        entries.append({
            "relation": graph.relation_names[relation].rsplit("/", 1)[-1],
            "source_block": src, "target_block": dst,
            "edge_count": int(count), "weight_sum": float(total_weight),
            "exposure_pairs": possible,
            "edge_intensity": float(count / possible) if possible > 0 else None,
        })
    return {
        "scope": "finite directed typed step-kernel; W-only empirical projection",
        "backend": "cuda" if cp is not None else "cpu",
        "fallback_reason": fallback_reason,
        "block_mass": block_mass.tolist(),
        "entries": entries,
        "S": "not estimated", "I": "not estimated",
    }
