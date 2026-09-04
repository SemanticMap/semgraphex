"""Sparse symmetric operators for the undirected A1 spectral baseline."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class OperatorInputError(ValueError):
    """Raised when an adjacency matrix cannot define the A1 operator."""


def _validated_symmetric_adjacency(adjacency: sparse.spmatrix, *, symmetry_tolerance: float) -> sparse.csr_matrix:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise OperatorInputError("adjacency must be a square matrix")
    matrix = adjacency.astype(np.float64, copy=False).tocsr()
    if not np.all(np.isfinite(matrix.data)):
        raise OperatorInputError("adjacency values must be finite")
    asymmetry = matrix - matrix.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > symmetry_tolerance:
        raise OperatorInputError("undirected A1 adjacency must be symmetric within tolerance")
    return ((matrix + matrix.T) * 0.5).tocsr()


def normalized_adjacency(
    adjacency: sparse.spmatrix,
    *,
    symmetry_tolerance: float = 1e-10,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return ``D^{-1/2} A D^{-1/2}`` without dense materialization.

    Zero-degree rows use a zero inverse degree, so isolates remain zero rows.
    """
    if symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance must be non-negative")
    matrix = _validated_symmetric_adjacency(adjacency, symmetry_tolerance=symmetry_tolerance)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(degrees < -symmetry_tolerance):
        raise OperatorInputError("undirected A1 adjacency must have non-negative degrees")
    inverse_sqrt_degree = np.zeros_like(degrees)
    positive = degrees > 0
    inverse_sqrt_degree[positive] = 1.0 / np.sqrt(degrees[positive])
    operator = sparse.diags(inverse_sqrt_degree, format="csr") @ matrix @ sparse.diags(inverse_sqrt_degree, format="csr")
    operator.eliminate_zeros()
    return operator.tocsr(), degrees


def normalized_laplacian(adjacency: sparse.spmatrix, *, symmetry_tolerance: float = 1e-10) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return the sparse normalized Laplacian; isolates retain a zero row."""
    operator, degrees = normalized_adjacency(adjacency, symmetry_tolerance=symmetry_tolerance)
    diagonal = np.zeros(operator.shape[0], dtype=np.float64)
    diagonal[degrees > 0] = 1.0
    return (sparse.diags(diagonal, format="csr") - operator).tocsr(), degrees
