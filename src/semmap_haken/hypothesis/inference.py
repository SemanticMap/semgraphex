"""CPU/OpenMP graph-tool backend for directed relation-layered SBM inference.

No implicit CUDA claims: Bayesian SBM fits execute on CPU. Input assertion
weights are deliberately excluded from this H1 topology/multiplicity model.
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np

from .graph import TypedGraph

ModelKind = Literal["SBM", "DC-SBM", "nested-DC-SBM"]
MODEL_KINDS: tuple[ModelKind, ...] = ("SBM", "DC-SBM", "nested-DC-SBM")


def _graph_tool_graph(graph: TypedGraph, gt):
    g = gt.Graph(directed=True)
    g.add_vertex(graph.vertex_count)
    relation_prop = g.new_ep("int32_t")
    data = np.column_stack((graph.source, graph.target, graph.relation)).astype(np.int64)
    g.add_edge_list(data, eprops=[relation_prop])
    return g, relation_prop


def _compact_levels(raw_levels: list[object], vertex_count: int) -> list[list[int]]:
    """Compose nested parent mappings down to original level-zero vertices."""
    composed = None
    levels = []
    for entry in raw_levels:
        index = np.asarray(entry.a if hasattr(entry, "a") else entry, dtype=np.int64)
        composed = index if composed is None else index[composed]
        if composed.shape != (vertex_count,):
            raise RuntimeError("graph-tool hierarchy does not map to original vertices")
        _, compact = np.unique(composed, return_inverse=True)
        levels.append(compact.tolist())
    return levels


def fit_one(
    graph: TypedGraph, *, kind: ModelKind, seed: int, threads: int = 1,
    independent_layers: bool = True,
) -> dict[str, object]:
    """Fit one stochastic solution; all three classes use the same typed graph.

    Layer semantics are explicit. The default independent-layer model retains
    one global partition while modelling relation types as distinct layers.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind: {kind}")
    if threads < 1:
        raise ValueError("threads must be positive")
    if graph.edge_count == 0 or graph.vertex_count < 2:
        raise ValueError("SBM needs nonempty graph with at least two vertices")
    try:
        import graph_tool.all as gt
    except ImportError as exc:
        raise RuntimeError(
            "graph-tool is required for H1; install python3-graph-tool or a compatible conda package"
        ) from exc

    gt.seed_rng(int(seed))
    gt.openmp_set_num_threads(int(threads))
    g, rel = _graph_tool_graph(graph, gt)
    args = {"ec": rel, "independent": independent_layers,
            "deg_corr": kind != "SBM"}
    if kind == "nested-DC-SBM":
        state = gt.minimize_nested_blockmodel_dl(
            g, base_state=gt.LayeredBlockState, base_state_args=args)
        levels = _compact_levels(state.get_bs(), graph.vertex_count)
    else:
        state = gt.minimize_blockmodel_dl(
            g, state=gt.LayeredBlockState, state_args=args)
        levels = [np.asarray(state.get_blocks().a, dtype=np.int64).tolist()]
    score = float(state.entropy() / np.log(2.0))
    if not np.isfinite(score):
        raise RuntimeError("non-finite description length")
    return {
        "kind": kind, "seed": int(seed), "mdl_bits": score,
        "partitions": levels, "blocks_by_level": [len(set(x)) for x in levels],
        "graph_tool_version": str(getattr(gt, "__version__", "unknown")),
        "threads": threads, "independent_layers": independent_layers,
        "weights_modelled": False, "directed": True,
        "description": "degree-corrected" if kind != "SBM" else "uncorrected",
    }


def fit_models(graph: TypedGraph, *, seed: int, threads: int | None = None) -> dict[str, object]:
    """Backward-compatible three-model comparison for the earlier pilot."""
    n_threads = threads if threads is not None else max(1, (os.cpu_count() or 1) - 1)
    result = {kind: fit_one(graph, kind=kind, seed=seed, threads=n_threads)
              for kind in MODEL_KINDS}
    nested = result["nested-DC-SBM"]
    return {
        "mdl_bits": {kind: float(obj["mdl_bits"]) for kind, obj in result.items()},
        "partitions": nested["partitions"],
        "blocks_by_level": nested["blocks_by_level"],
        "seed": seed, "threads": n_threads,
        "model": "directed relation-layered SBM / DC-SBM / nested DC-SBM",
    }
