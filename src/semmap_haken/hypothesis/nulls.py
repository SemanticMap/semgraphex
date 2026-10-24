"""Relation-specific directed degree-preserving nulls (multigraph semantics)."""
from __future__ import annotations

import numpy as np

from .graph import TypedGraph


def typed_degrees(graph: TypedGraph) -> tuple[np.ndarray, np.ndarray]:
    """Out/in degree by relation, including parallel edge multiplicities."""
    n, r = graph.vertex_count, len(graph.relation_names)
    out_degree = np.zeros((r, n), dtype=np.int64)
    in_degree = np.zeros((r, n), dtype=np.int64)
    for layer in range(r):
        indices = graph.relation == layer
        out_degree[layer] = np.bincount(graph.source[indices], minlength=n)
        in_degree[layer] = np.bincount(graph.target[indices], minlength=n)
    return out_degree, in_degree


def degree_preserving_null(
    graph: TypedGraph, *, seed: int, swaps_per_edge: int = 5,
) -> tuple[TypedGraph, dict[str, int]]:
    """Switch destinations of two edges *within the same relation*.

    Directed per-relation in/out degrees and relation counts are invariant.
    Existing loops remain untouched and new loops are forbidden; parallel
    edges are permitted, matching assertion-multigraph input semantics.
    Null edges are synthetic and therefore receive new provenance IDs.
    """
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be nonnegative")
    rng = np.random.default_rng(seed)
    targets = graph.target.copy()
    attempts = accepted = 0
    for layer in range(len(graph.relation_names)):
        idx = np.flatnonzero((graph.relation == layer) & (graph.source != graph.target))
        if idx.size < 2:
            continue
        for _ in range(swaps_per_edge * len(idx)):
            i, j = rng.choice(idx, size=2, replace=False)
            attempts += 1
            a, b = graph.source[i], graph.source[j]
            x, y = targets[i], targets[j]
            if a == b or x == y or a == y or b == x:
                continue
            targets[i], targets[j] = y, x
            accepted += 1
    shuffled = TypedGraph(
        graph.node_ids, graph.relation_names, graph.source.copy(), targets,
        graph.relation.copy(), graph.weight.copy(),
        tuple(f"null:{seed}:{i}" for i in range(graph.edge_count)),
    )
    return shuffled, {"attempted": attempts, "accepted": accepted}
