"""Exclusive W/S/I/R labeling of directed, typed, weighted edge *records*.

This is a finite lossless-codec labeling, not an estimate of a limiting
exchangeable graphex. Parallel records remain distinct via their record ID.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

Part = Literal["W", "S", "I", "R"]


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: int
    source: int
    target: int
    relation: str
    weight: float


@dataclass(frozen=True)
class AssignedEdge:
    edge: EdgeRecord
    part: Part
    owner: int | None = None


def classify_edges(
    vertex_count: int,
    edges: Iterable[EdgeRecord],
    figures: Iterable[Iterable[int]],
) -> tuple[AssignedEdge, ...]:
    """Assign every directed record once; incidence ignores orientation.

    W: endpoints in the same accepted figure.
    S: one figure endpoint and a distinct degree-one vertex outside all
       figures, attached to exactly that one figure endpoint.
    I: an isolated two-vertex connected component with exactly one
       non-loop edge record and neither endpoint in a figure.
    R: all remaining records, including figure-to-figure boundaries.
    Self-loops never become S or I. Degree is counted by *edge records*,
    so parallel or reciprocal edges do not masquerade as isolated pairs.
    """
    if vertex_count < 0:
        raise ValueError("vertex_count must be nonnegative")
    records = tuple(edges)
    if len({e.edge_id for e in records}) != len(records):
        raise ValueError("edge_id must be unique for each edge record")
    owner: dict[int, int] = {}
    for number, group in enumerate(figures):
        nodes = tuple(int(n) for n in group)
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("figures must be nonempty sets of vertices")
        for node in nodes:
            if not 0 <= node < vertex_count:
                raise ValueError("figure vertex outside graph")
            if node in owner:
                raise ValueError("overlapping accepted figures")
            owner[node] = number
    degree: Counter[int] = Counter()
    for edge in records:
        if not (0 <= edge.source < vertex_count and 0 <= edge.target < vertex_count):
            raise ValueError("edge endpoint outside graph")
        # A loop is incident twice; it cannot count as a leaf.
        degree[edge.source] += 1
        degree[edge.target] += 1
    result: list[AssignedEdge] = []
    for edge in records:
        a, b = edge.source, edge.target
        fa, fb = owner.get(a), owner.get(b)
        part: Part = "R"
        assigned_owner: int | None = None
        if fa is not None and fa == fb:
            part, assigned_owner = "W", fa
        elif a != b and fa is not None and fb is None and degree[b] == 1:
            part, assigned_owner = "S", fa
        elif a != b and fb is not None and fa is None and degree[a] == 1:
            part, assigned_owner = "S", fb
        elif (a != b and fa is None and fb is None
              and degree[a] == degree[b] == 1):
            part = "I"
        result.append(AssignedEdge(edge, part, assigned_owner))
    if len(result) != len(records):
        raise AssertionError("edge ownership is not exhaustive")
    return tuple(result)


def validate_partition(original: Iterable[EdgeRecord],
                       assigned: Iterable[AssignedEdge]) -> None:
    """Check record-level conservation, not just aggregated adjacency."""
    source = {e.edge_id: e for e in original}
    partition = tuple(assigned)
    if len(source) != len(partition):
        raise ValueError("edge count differs or duplicate source IDs")
    if {item.edge.edge_id for item in partition} != set(source):
        raise ValueError("edge partition misses or duplicates record IDs")
    if any(source[item.edge.edge_id] != item.edge for item in partition):
        raise ValueError("edge record mutated during classification")
    if any(item.part not in ("W", "S", "I", "R") for item in partition):
        raise ValueError("unknown component")
