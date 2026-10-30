"""Leakage-free relation-stratified splits of directed assertion groups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import TypedGraph


@dataclass(frozen=True)
class GraphSplit:
    train: TypedGraph
    validation: TypedGraph
    test: TypedGraph
    group_counts: dict[str, int]


def split_graph(
    graph: TypedGraph, *, seed: int, train_fraction: float = .8,
    validation_fraction: float = .1,
) -> GraphSplit:
    """Keep parallel statements and reciprocals within one split.

    Relations with fewer than three distinct undirected endpoint pairs remain
    entirely in training; their prediction quality cannot be estimated.
    Vertex/relation dictionaries are kept identical across all three splits.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("need 0 < train_fraction < 1 and 0 <= validation_fraction < 1-train_fraction")
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, (u, v, r) in enumerate(zip(graph.source, graph.target, graph.relation)):
        x, y = sorted((int(u), int(v)))
        groups.setdefault((int(r), x, y), []).append(i)
    by_rel: dict[int, list[tuple[int, int, int]]] = {}
    for key in sorted(groups):
        by_rel.setdefault(key[0], []).append(key)
    mask = np.zeros(graph.edge_count, dtype=np.int8)
    n_groups = [0, 0, 0]
    for keys in by_rel.values():
        order = rng.permutation(len(keys))
        n = len(keys)
        if n < 3 or validation_fraction == 0:
            n_train = max(1, int(np.floor(n * train_fraction)))
            n_validation = 0
            if n < 2:
                n_train = n
        else:
            n_validation = max(1, int(round(n * validation_fraction)))
            n_train = max(1, int(round(n * train_fraction)))
            n_train = min(n_train, n - n_validation - 1)
        for rank, j in enumerate(order):
            bucket = 0 if rank < n_train else (1 if rank < n_train + n_validation else 2)
            n_groups[bucket] += 1
            mask[groups[keys[int(j)]]] = bucket
    return GraphSplit(
        train=graph.edge_subset(mask == 0),
        validation=graph.edge_subset(mask == 1),
        test=graph.edge_subset(mask == 2),
        group_counts={"train": n_groups[0], "validation": n_groups[1], "test": n_groups[2]},
    )
