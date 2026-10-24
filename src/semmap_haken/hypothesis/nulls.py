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


def stub_matching_null(graph: TypedGraph, *, seed: int) -> tuple[TypedGraph, dict[str, object]]:
    """Exact independent typed directed configuration *stub* sample, O(E).

    Shuffles target stubs independently within each relation (including loops
    and multiedges). This is not a loop-free/simple-graph null; report the
    support explicitly. No MCMC burn-in or mixing assumptions are required.
    """
    rng = np.random.default_rng(seed)
    target = graph.target.copy()
    for r in range(len(graph.relation_names)):
        indices = np.flatnonzero(graph.relation == r)
        target[indices] = rng.permutation(target[indices])
    sampled = TypedGraph(graph.node_ids, graph.relation_names,
                         graph.source.copy(), target, graph.relation.copy(),
                         graph.weight.copy(),
                         tuple(f"null:stub:{seed}:{i}" for i in range(graph.edge_count)))
    before, after = typed_degrees(graph), typed_degrees(sampled)
    if any(not np.array_equal(a, b) for a,b in zip(before, after)):
        raise AssertionError("stub matching broke the typed degree sequence")
    diagnostic = {
        "ensemble": "directed_typed_stub_matching", "self_loops_allowed": True,
        "parallel_edges_allowed": True, "seed": int(seed),
        "loops": int(np.count_nonzero(sampled.source == sampled.target)),
        "changed_targets": int(np.count_nonzero(sampled.target != graph.target)),
        "edge_count": graph.edge_count,
    }
    return sampled, diagnostic


def flat_dc_sbm_null(
    graph: TypedGraph, partition: np.ndarray, *, seed: int,
) -> tuple[TypedGraph, dict[str, object]]:
    """Layerwise microcanonical DC-SBM, conditioned on fitted *flat* blocks.

    Preserves original directed typed degrees *and* inter-block counts per
    relation. Supports loops and parallel edges. Only valid as a conditional
    flat-model control, not as a full posterior predictive sample.
    """
    try:
        import graph_tool.all as gt
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError("flat DC-SBM controls require graph-tool and scipy") from exc
    labels = np.asarray(partition, dtype=np.int64)
    if labels.shape != (graph.vertex_count,) or np.any(labels < 0):
        raise ValueError("invalid flat block partition")
    _, block = np.unique(labels, return_inverse=True)
    blocks = int(np.max(block)) + 1
    out, inn = typed_degrees(graph)
    gt.seed_rng(int(seed))
    sources, targets, relations = [], [], []
    for r in range(len(graph.relation_names)):
        edge = graph.relation == r
        if not np.any(edge):
            continue
        from_block = block[graph.source[edge]]
        to_block = block[graph.target[edge]]
        pair_counts = sparse.coo_matrix(
            (np.ones(int(edge.sum()), dtype=np.int64),
             (from_block, to_block)), shape=(blocks, blocks)
        ).tocsr()
        sampled = gt.generate_sbm(
            block, pair_counts, out_degs=out[r], in_degs=inn[r],
            directed=True, micro_degs=True, micro_ers=True,
        )
        ends = np.asarray(sampled.get_edges(), dtype=np.int64)[:, :2]
        sources.append(ends[:, 0])
        targets.append(ends[:, 1])
        relations.append(np.full(len(ends), r, dtype=np.int32))
    u = np.concatenate(sources) if sources else np.array([], dtype=np.int64)
    v = np.concatenate(targets) if targets else np.array([], dtype=np.int64)
    r = np.concatenate(relations) if relations else np.array([], dtype=np.int32)
    obj = TypedGraph(graph.node_ids, graph.relation_names, u, v, r,
                     np.ones(len(u)), tuple(f"null:flat:{seed}:{i}" for i in range(len(u))))
    if any(not np.array_equal(a, b) for a,b in zip((out,inn),typed_degrees(obj))):
        raise AssertionError("flat DC-SBM generator changed relation-specific degrees")
    return obj, {"ensemble": "conditional_flat_dc_sbm", "seed": int(seed),
                 "flat_blocks": blocks, "self_loops_allowed": True,
                 "parallel_edges_allowed": True, "edge_count": obj.edge_count,
                 "loops": int(np.count_nonzero(u == v))}
