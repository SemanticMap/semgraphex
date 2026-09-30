"""Integrity helpers for adjacency and relation-layer representations."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import sparse


def sum_relation_layers(
    relation_layers: Mapping[str, sparse.spmatrix],
    *,
    shape: tuple[int, int] | None = None,
) -> sparse.csr_matrix:
    if shape is None:
        first = next(iter(relation_layers.values()), None)
        shape = first.shape if first is not None else (0, 0)
    total = sparse.csr_matrix(shape, dtype=np.float64)
    for _, matrix in sorted(relation_layers.items()):
        if matrix.shape != shape:
            raise ValueError("relation layers have inconsistent shapes")
        total = total + matrix.tocsr().astype(np.float64, copy=False)
    total.sum_duplicates()
    total.sort_indices()
    total.eliminate_zeros()
    return total


def adjacency_relation_diagnostic(
    adjacency: sparse.spmatrix,
    relation_layers: Mapping[str, sparse.spmatrix],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, object]:
    source = adjacency.tocsr().astype(np.float64, copy=False)
    reconstructed = sum_relation_layers(
        relation_layers,
        shape=source.shape,
    )
    delta = (source - reconstructed).tocsr()
    max_abs = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    source_sum = float(source.sum())
    reconstructed_sum = float(reconstructed.sum())
    matches = bool(
        source.shape == reconstructed.shape
        and np.isclose(source_sum, reconstructed_sum, rtol=rtol, atol=atol)
        and max_abs <= atol + rtol * max(
            1.0,
            float(np.max(np.abs(source.data))) if source.nnz else 0.0,
        )
    )
    return {
        "matches": matches,
        "max_abs_error": max_abs,
        "adjacency_nnz": int(source.nnz),
        "relation_sum_nnz": int(reconstructed.nnz),
        "adjacency_weight_sum": source_sum,
        "relation_weight_sum": reconstructed_sum,
        "rtol": float(rtol),
        "atol": float(atol),
    }
