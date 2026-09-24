"""Optional, bounded CUDA nearest-neighbor backend for graph-type features.

WL/VF2 and CSR candidate extraction remain CPU work. Only dense cosine
search over the unique type embeddings is transferred to the GPU. The module
imports PyTorch lazily so CPU installations do not need a torch dependency.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse

DeviceChoice = Literal["auto", "cpu", "cuda"]


def resolve_device(
    requested: DeviceChoice,
    *,
    cuda_available: bool | None = None,
) -> Literal["cpu", "cuda"]:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cpu":
        return "cpu"
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but PyTorch/CUDA is not available")
    return "cuda" if cuda_available else "cpu"


def cuda_cosine_neighbors(
    features: sparse.spmatrix | np.ndarray,
    *,
    k: int,
    query_batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Exact full-database cosine top-k, batched by query on CUDA.

    The feature database is materialized once in VRAM. A CUDA OOM while
    allocating it is propagated to allow explicit auto->CPU fallback; an OOM
    in a query tile halves the tile size and retries. No CPU/GPU results are
    mixed inside one kNN graph.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if query_batch_size < 1 or k < 1:
        raise ValueError("k and query_batch_size must be positive")
    count = int(features.shape[0])
    if count <= 1:
        return (
            np.empty((count, 0), dtype=np.int64),
            np.empty((count, 0), dtype=np.float64),
            {"backend": "torch_cuda", "device": "cuda", "precision": "float32"},
        )
    k = min(k, count - 1)
    dense = (
        features.toarray().astype(np.float32, copy=False)
        if sparse.issparse(features)
        else np.asarray(features, dtype=np.float32)
    )
    # Torch is intentionally optional; this conversion is bounded by
    # N_types * feature_dim * sizeof(float32), not N_types squared.
    matrix = torch.as_tensor(dense, device="cuda")
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    indices = np.empty((count, k), dtype=np.int64)
    distances = np.empty((count, k), dtype=np.float64)
    start = 0
    batch = min(query_batch_size, count)
    while start < count:
        stop = min(start + batch, count)
        try:
            with torch.no_grad():
                similarity = matrix[start:stop] @ matrix.T
                block = (1.0 - similarity.clamp(-1.0, 1.0)).clamp_min_(0.0)
                local_rows = torch.arange(stop - start, device="cuda")
                global_rows = torch.arange(start, stop, device="cuda")
                block[local_rows, global_rows] = float("inf")
                # Stable sort also resolves exact-distance ties by type index.
                nearest = torch.argsort(block, dim=1, stable=True)[:, :k]
                selected = torch.gather(block, 1, nearest)
                indices[start:stop] = nearest.cpu().numpy().astype(np.int64)
                distances[start:stop] = selected.cpu().numpy().astype(np.float64)
            start = stop
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            # References are dead after the exception; do not use empty_cache
            # as a substitute for reducing the working batch.
    return indices, distances, {
        "backend": "torch_cuda",
        "device": "cuda",
        "precision": "float32",
        "query_batch_size": batch,
        "candidate_count": count,
    }
