"""Directed typed edge records: no dense adjacency or implicit symmetrisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from semmap_haken.conceptnet import Assertion


@dataclass(frozen=True)
class TypedGraph:
    node_ids: tuple[str, ...]
    relation_names: tuple[str, ...]
    source: np.ndarray
    target: np.ndarray
    relation: np.ndarray
    weight: np.ndarray
    assertion_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n, m = len(self.node_ids), len(self.assertion_ids)
        if len(set(self.node_ids)) != n or len(set(self.relation_names)) != len(self.relation_names):
            raise ValueError("node and relation identifiers must be unique")
        for name in ("source", "target", "relation", "weight"):
            if np.asarray(getattr(self, name)).shape != (m,):
                raise ValueError(f"{name} length must match assertion_ids")
        if not (np.issubdtype(self.source.dtype, np.integer) and
                np.issubdtype(self.target.dtype, np.integer) and
                np.issubdtype(self.relation.dtype, np.integer)):
            raise ValueError("edge endpoints and relations must be integer arrays")
        if (np.any(self.source < 0) or np.any(self.source >= n) or
            np.any(self.target < 0) or np.any(self.target >= n) or
            np.any(self.relation < 0) or np.any(self.relation >= len(self.relation_names))):
            raise ValueError("edge endpoint or relation index out of range")
        if not np.all(np.isfinite(self.weight)):
            raise ValueError("edge weights must be finite")

    @property
    def vertex_count(self) -> int:
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        return len(self.assertion_ids)

    @classmethod
    def from_assertions(cls, assertions: Iterable[Assertion]) -> "TypedGraph":
        """Keep each original assertion separately, including parallel edges.

        Input is a filtered, finite dataset. The existing streaming parser remains
        responsible for source-level provenance and input checksum tracking.
        """
        records = list(assertions)
        nodes = tuple(sorted({v for a in records for v in (a.start_uri, a.end_uri)}))
        relations = tuple(sorted({a.relation_uri for a in records}))
        node_index = {v: i for i, v in enumerate(nodes)}
        relation_index = {r: i for i, r in enumerate(relations)}
        return cls(
            nodes, relations,
            np.fromiter((node_index[a.start_uri] for a in records), dtype=np.int64),
            np.fromiter((node_index[a.end_uri] for a in records), dtype=np.int64),
            np.fromiter((relation_index[a.relation_uri] for a in records), dtype=np.int32),
            np.fromiter((a.weight for a in records), dtype=np.float64),
            tuple(a.assertion_uri for a in records),
        )

    def edge_subset(self, keep: np.ndarray) -> "TypedGraph":
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (self.edge_count,):
            raise ValueError("edge mask shape mismatch")
        return TypedGraph(self.node_ids, self.relation_names,
                          self.source[mask], self.target[mask], self.relation[mask],
                          self.weight[mask], tuple(x for x, ok in zip(self.assertion_ids, mask) if ok))

    def vertex_subset(self, keep: np.ndarray, *, remove_isolates: bool = True) -> "TypedGraph":
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (self.vertex_count,):
            raise ValueError("vertex mask shape mismatch")
        edge_mask = mask[self.source] & mask[self.target]
        src, dst = self.source[edge_mask], self.target[edge_mask]
        if remove_isolates:
            mask = np.zeros(self.vertex_count, dtype=bool)
            mask[src] = True
            mask[dst] = True
        remap = np.full(self.vertex_count, -1, dtype=np.int64)
        remap[mask] = np.arange(np.count_nonzero(mask))
        return TypedGraph(tuple(n for n, ok in zip(self.node_ids, mask) if ok),
                          self.relation_names, remap[src], remap[dst],
                          self.relation[edge_mask], self.weight[edge_mask],
                          tuple(x for x, ok in zip(self.assertion_ids, edge_mask) if ok))


def p_sample(graph: TypedGraph, *, p: float, seed: int) -> TypedGraph:
    """Graphex p-sampling: induced vertex sample, then remove isolates."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = np.random.default_rng(seed)
    return graph.vertex_subset(rng.random(graph.vertex_count) < p)


def edge_dropout(graph: TypedGraph, *, keep_probability: float, seed: int) -> TypedGraph:
    """Perturb assertions without silently dropping surviving vertices."""
    if not 0.0 <= keep_probability <= 1.0:
        raise ValueError("keep_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    return graph.edge_subset(rng.random(graph.edge_count) < keep_probability)
