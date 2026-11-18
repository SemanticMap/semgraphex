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
    group_keys: tuple[str, ...] | None = None,
) -> GraphSplit:
    """Keep parallel, reciprocal AND group-key-related edges in one split.

    Optional group_keys are external families (e.g. text ConceptNet assertion
    concept-pair group across many repeated word occurrences). They MUST be
    constructed without held-out labels. Splits are stratified by relation,
    preserve full node/relation dictionaries, and never split one family.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("need 0 < train_fraction < 1 and 0 <= validation_fraction < 1-train_fraction")
    if group_keys is not None and (
            len(group_keys) != graph.edge_count or
            any(not isinstance(x, str) or not x for x in group_keys)):
        raise ValueError("group_keys must contain one nonempty string per edge")
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = {}
    for i, (u, v, r) in enumerate(zip(graph.source, graph.target, graph.relation)):
        # Generic ConceptNet splits use the node pair. Text occurrence graphs
        # must group by original *concept* pair to stop source assertion reuse.
        x, y = sorted((int(u), int(v)))
        key = ((int(r), group_keys[i]) if group_keys is not None
               else (int(r), x, y))
        groups.setdefault(key, []).append(i)
    by_rel: dict[int, list[tuple]] = {}
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
