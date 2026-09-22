"""Sparse quotient construction through an explicit membership matrix.

Semantics (documented contract):

- ``membership_matrix``: CSR indicator ``P`` with ``P[i, c] = 1`` iff fine node
  ``i`` belongs to supernode ``c``. Clusters are disjoint, so ``P^T P`` is the
  diagonal matrix of supernode sizes.
- ``sum`` aggregation: ``A' = P^T A P``.  Each off-diagonal entry is the summed
  edge weight between two clusters; intra-cluster weight lands on the diagonal
  (self-loop convention).  Total edge weight is conserved.
- ``mean_density`` aggregation: ``A'`` divides each sum block by
  ``|C_a||C_b|`` (block mean density).
- Mass: ``m_c = |C_c| / N``; ``parent_child`` maps each supernode to the sorted
  fine node indices and original node URIs, so the contraction is reversible.
- The fine adjacency is never densified; only the small coarse block matrix is
  materialized via sparse products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class QuotientResult:
    adjacency: sparse.csr_matrix
    masses: np.ndarray
    supernode_sizes: tuple[int, ...]
    parent_child: dict[int, tuple[int, ...]]
    fine_node_ids: tuple[str, ...]
    aggregation: Literal["sum", "mean_density"]
    provenance: dict[str, object]


def membership_matrix(fine_to_coarse: np.ndarray) -> sparse.csr_matrix:
    """Return the sparse indicator matrix ``P`` of the fine-to-coarse map."""
    assignment = np.asarray(fine_to_coarse, dtype=np.int64)
    count = int(assignment.max()) + 1 if assignment.size else 0
    rows = np.arange(assignment.size)
    matrix = sparse.csr_matrix((np.ones(assignment.size, dtype=np.float64), (rows, assignment)), shape=(assignment.size, count))
    matrix.sum_duplicates()
    if matrix.nnz != assignment.size:
        raise ValueError("fine_to_coarse must assign each fine node exactly once")
    return matrix


def build_quotient(
    adjacency: sparse.spmatrix,
    fine_to_coarse: np.ndarray,
    *,
    aggregation: Literal["sum", "mean_density"],
    node_ids: Sequence[str] | None = None,
    level: int = 0,
    parent_run_id: str | None = None,
) -> QuotientResult:
    """Contract a sparse fine graph through its explicit membership matrix."""
    if aggregation not in {"sum", "mean_density"}:
        raise ValueError("aggregation must be sum or mean_density")
    fine = adjacency.tocsr()
    membership = membership_matrix(fine_to_coarse)
    if membership.shape[0] != fine.shape[0]:
        raise ValueError("fine_to_coarse must have one entry per fine node")
    sizes = np.asarray(membership.sum(axis=0)).ravel().astype(int)
    
    coarse = (membership.T @ fine @ membership).tocsr()

    # For an undirected fine graph P^T A P is mathematically symmetric.
    # Remove sparse floating-point accumulation asymmetry explicitly.
    coarse = ((coarse + coarse.T) * 0.5).tocsr()
    coarse.eliminate_zeros()
    
    if aggregation == "mean_density":
        coarse = _scale_blocks(coarse, np.outer(sizes, sizes))
    masses = sizes / float(fine.shape[0])
    parent_child = {int(coarse_index): tuple(np.flatnonzero(np.asarray(fine_to_coarse) == coarse_index).tolist()) for coarse_index in range(membership.shape[1])}
    ids = tuple(node_ids) if node_ids is not None else tuple(str(index) for index in range(fine.shape[0]))
    if len(ids) != fine.shape[0]:
        raise ValueError("node_ids must provide one identifier per fine node")
    provenance = {
        "level": level,
        "parent_run_id": parent_run_id,
        "fine_node_count": int(fine.shape[0]),
        "coarse_node_count": int(membership.shape[1]),
        "self_loop_convention": "intra_cluster_weight_on_diagonal",
        "fine_ids_preserved": True,
    }
    return QuotientResult(coarse, masses, tuple(int(size) for size in sizes), parent_child, ids, aggregation, provenance)


def _scale_blocks(matrix: sparse.csr_matrix, scale: np.ndarray) -> sparse.csr_matrix:
    """Divide each CSR entry ``(a, b)`` by ``scale[a, b]`` without densifying."""
    scaled = matrix.copy().astype(np.float64)
    rows = np.repeat(np.arange(scaled.shape[0]), np.diff(scaled.indptr))
    scaled.data = scaled.data / scale[rows, scaled.indices]
    return scaled
