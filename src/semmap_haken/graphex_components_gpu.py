"""Optional CUDA acceleration for exact record-level W/S/I/R classification.

GPU only handles vectorized incidence and category masks. VF2, dictionary
canonicalization and byte-accurate ZIP/Huffman coding remain CPU workloads.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .graphex_components import AssignedEdge, EdgeRecord, classify_edges, validate_partition


def classify_edges_accelerated(
    vertex_count: int,
    edges: Iterable[EdgeRecord],
    figures: Iterable[Iterable[int]],
    *,
    device: str = "auto",
    batch_size: int = 500_000,
) -> tuple[AssignedEdge, ...]:
    """CUDA-first classification with bounded GPU batches and CPU fallback.

    Parameters are explicit to prevent silently claiming that a CPU run used
    the GPU. The returned assignments match classify_edges exactly.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    records = tuple(edges)
    groups = tuple(tuple(int(x) for x in group) for group in figures)
    if device == "cpu" or not records:
        return classify_edges(vertex_count, records, groups)
    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise RuntimeError("CUDA was explicitly requested but torch is unavailable")
        return classify_edges(vertex_count, records, groups)
    if not torch.cuda.is_available():
        if device == "cuda":
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return classify_edges(vertex_count, records, groups)
    if vertex_count < 0:
        raise ValueError("vertex_count must be nonnegative")
    if len({edge.edge_id for edge in records}) != len(records):
        raise ValueError("edge_id must be unique for each edge record")
    owners = np.full(vertex_count, -1, dtype=np.int32)
    for index, group in enumerate(groups):
        if not group or len(set(group)) != len(group):
            raise ValueError("figures must be nonempty sets of vertices")
        for node in group:
            if not 0 <= node < vertex_count:
                raise ValueError("figure vertex outside graph")
            if owners[node] != -1:
                raise ValueError("overlapping accepted figures")
            owners[node] = index
    source = np.fromiter((e.source for e in records), dtype=np.int64,
                         count=len(records))
    target = np.fromiter((e.target for e in records), dtype=np.int64,
                         count=len(records))
    if np.any(source < 0) or np.any(target < 0) or np.any(source >= vertex_count) or np.any(target >= vertex_count):
        raise ValueError("edge endpoint outside graph")
    d = torch.device("cuda")
    owner_gpu = torch.as_tensor(owners, device=d)
    degree_gpu = torch.zeros(vertex_count, dtype=torch.int64, device=d)
    labels = np.empty(len(records), dtype=np.uint8)
    for start in range(0, len(records), batch_size):
        stop = min(len(records), start + batch_size)
        u = torch.as_tensor(source[start:stop], device=d)
        v = torch.as_tensor(target[start:stop], device=d)
        degree_gpu.index_add_(0, u, torch.ones_like(u))
        degree_gpu.index_add_(0, v, torch.ones_like(v))
    for start in range(0, len(records), batch_size):
        stop = min(len(records), start + batch_size)
        u = torch.as_tensor(source[start:stop], device=d)
        v = torch.as_tensor(target[start:stop], device=d)
        fu, fv = owner_gpu[u], owner_gpu[v]
        different = u != v
        w = (fu >= 0) & (fu == fv)
        s = different & (((fu >= 0) & (fv < 0) & (degree_gpu[v] == 1)) |
                         ((fv >= 0) & (fu < 0) & (degree_gpu[u] == 1)))
        i = different & (fu < 0) & (fv < 0) & (degree_gpu[u] == 1) & (degree_gpu[v] == 1)
        local = torch.full_like(u, 3, dtype=torch.uint8)
        local[i] = 2
        local[s] = 1
        local[w] = 0
        labels[start:stop] = local.cpu().numpy()
    parts = ("W", "S", "I", "R")
    result = tuple(AssignedEdge(edge, parts[int(label)],
                                (int(owners[edge.source]) if owners[edge.source] >= 0
                                 else int(owners[edge.target]))
                                if label in (0, 1) else None)
                   for edge, label in zip(records, labels, strict=True))
    validate_partition(records, result)
    return result
