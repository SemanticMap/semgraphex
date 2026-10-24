"""Optional graph-tool inference of typed directed SBM and nested DC-SBM.

Graph-tool uses CPU/OpenMP, not CUDA. GPU acceleration is applied only to
vectorizable block projections, never claimed for Bayesian model fitting.
"""
from __future__ import annotations

import os

import numpy as np

from .graph import TypedGraph


def fit_models(graph: TypedGraph, *, seed: int, threads: int | None = None) -> dict[str, object]:
    """Fit three model classes to exactly the same directed typed multigraph.

    MDL values are in bits (graph-tool state entropy returns nats).
    """
    if graph.edge_count == 0:
        raise ValueError("cannot infer SBM on an edgeless graph")
    try:
        import graph_tool.all as gt
    except ImportError as exc:
        raise RuntimeError("graph-tool is required for nSBM; install it via a compatible system/conda package") from exc
    if threads is not None and threads < 1:
        raise ValueError("threads must be positive")
    gt.seed_rng(int(seed))
    gt.openmp_set_num_threads(int(threads or max(1, (os.cpu_count() or 1) - 1)))
    g = gt.Graph(directed=True)
    g.add_vertex(graph.vertex_count)
    layer = g.new_ep("int32_t")
    edges = np.column_stack((graph.source, graph.target, graph.relation)).astype(np.int64)
    g.add_edge_list(edges, eprops=[layer])
    base = gt.LayeredBlockState

    plain = gt.minimize_blockmodel_dl(g, state=base, state_args={"ec": layer, "deg_corr": False})
    dc = gt.minimize_blockmodel_dl(g, state=base, state_args={"ec": layer, "deg_corr": True})
    nested = gt.minimize_nested_blockmodel_dl(g, base_state=base,
                                              base_state_args={"ec": layer, "deg_corr": True})

    memberships = []
    current = None
    for level in nested.get_bs():
        vals = np.asarray(level.a if hasattr(level, "a") else level, dtype=np.int64)
        current = vals if current is None else vals[current]
        if current.shape != (graph.vertex_count,):
            raise RuntimeError("graph-tool returned an incompatible hierarchy")
        _, compact = np.unique(current, return_inverse=True)
        memberships.append(compact.tolist())
    bits = {"SBM": float(plain.entropy() / np.log(2)),
            "DC-SBM": float(dc.entropy() / np.log(2)),
            "nested-DC-SBM": float(nested.entropy() / np.log(2))}
    return {"mdl_bits": bits, "partitions": memberships,
            "blocks_by_level": [len(set(level)) for level in memberships],
            "seed": int(seed), "threads": int(threads or max(1, (os.cpu_count() or 1) - 1)),
            "model": "directed relation-layered SBM / DC-SBM / nested DC-SBM"}
