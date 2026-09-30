"""Indexes for incremental graph-grammar occurrence maintenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class IndexedOccurrence:
    occurrence_id: int
    type_id: str
    center: int
    nodes: tuple[int, ...]


class OccurrenceIndex:
    """Bidirectional index used to invalidate only locally affected motifs."""

    def __init__(self) -> None:
        self.by_id: dict[int, IndexedOccurrence] = {}
        self.by_type: dict[str, set[int]] = {}
        self.by_node: dict[int, set[int]] = {}
        self.by_center: dict[int, set[int]] = {}

    def add(self, occurrence: IndexedOccurrence) -> None:
        if occurrence.occurrence_id in self.by_id:
            raise ValueError("duplicate occurrence_id")
        if not occurrence.nodes:
            raise ValueError("occurrence must contain nodes")
        self.by_id[occurrence.occurrence_id] = occurrence
        self.by_type.setdefault(occurrence.type_id, set()).add(
            occurrence.occurrence_id
        )
        self.by_center.setdefault(int(occurrence.center), set()).add(
            occurrence.occurrence_id
        )
        for node in occurrence.nodes:
            self.by_node.setdefault(int(node), set()).add(
                occurrence.occurrence_id
            )

    def remove(self, occurrence_id: int) -> IndexedOccurrence:
        occurrence = self.by_id.pop(int(occurrence_id))
        self.by_type[occurrence.type_id].discard(occurrence.occurrence_id)
        if not self.by_type[occurrence.type_id]:
            del self.by_type[occurrence.type_id]
        self.by_center[occurrence.center].discard(occurrence.occurrence_id)
        if not self.by_center[occurrence.center]:
            del self.by_center[occurrence.center]
        for node in occurrence.nodes:
            ids = self.by_node[node]
            ids.discard(occurrence.occurrence_id)
            if not ids:
                del self.by_node[node]
        return occurrence

    def invalidate_nodes(
        self,
        nodes: Iterable[int],
    ) -> tuple[IndexedOccurrence, ...]:
        ids = {
            occurrence_id
            for node in nodes
            for occurrence_id in self.by_node.get(int(node), ())
        }
        return tuple(
            self.remove(occurrence_id)
            for occurrence_id in sorted(ids)
        )

    def occurrences_for_type(self, type_id: str) -> tuple[IndexedOccurrence, ...]:
        return tuple(
            self.by_id[occurrence_id]
            for occurrence_id in sorted(self.by_type.get(str(type_id), ()))
        )

    @staticmethod
    def affected_centers(
        adjacency: sparse.spmatrix,
        changed_nodes: Iterable[int],
        *,
        radius: int,
    ) -> tuple[int, ...]:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        graph = adjacency.tocsr()
        current = {int(node) for node in changed_nodes}
        if any(node < 0 or node >= graph.shape[0] for node in current):
            raise ValueError("changed node outside graph")
        seen = set(current)
        frontier = set(current)
        support = graph.maximum(graph.T).tocsr()
        for _ in range(radius):
            next_frontier: set[int] = set()
            for node in frontier:
                start, stop = support.indptr[node], support.indptr[node + 1]
                next_frontier.update(
                    int(value) for value in support.indices[start:stop]
                )
            next_frontier -= seen
            seen.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return tuple(sorted(seen))

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[IndexedOccurrence],
    ) -> "OccurrenceIndex":
        result = cls()
        for row in rows:
            result.add(row)
        return result
