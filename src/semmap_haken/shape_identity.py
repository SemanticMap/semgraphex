"""Level-independent exact grammar symbols; configurable similarity is a view.

IDs are append-only within one persisted registry, not universal graph hashes.
Weights and external bindings belong to occurrences, not internal shape identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from typing import Literal
from uuid import uuid4

import networkx as nx

IsomorphismLevel = Literal["topology", "directed", "typed", "recursive"]
LEVELS = ("topology", "directed", "typed", "recursive")
CONTRACT = "directed-relation-multiset-recursive-v1"


@dataclass(frozen=True)
class Shape:
    node_types: tuple[str, ...]
    edges: tuple[tuple[int, int, str], ...]


def _graph(shape: Shape, level: str):
    if level not in LEVELS:
        raise ValueError(f"unknown isomorphism level: {level!r}")
    graph = nx.MultiGraph() if level == "topology" else nx.MultiDiGraph()
    for node, label in enumerate(shape.node_types):
        graph.add_node(node, kind=label if level == "recursive" else "ATOM")
    for u, v, relation in shape.edges:
        graph.add_edge(u, v, relation=relation if level in ("typed", "recursive") else "")
    return graph


def _bucket(graph):
    degrees = (sorted((graph.in_degree(n), graph.out_degree(n)) for n in graph)
               if graph.is_directed() else sorted(d for _, d in graph.degree()))
    return (len(graph), graph.number_of_edges(), tuple(degrees),
            tuple(sorted(Counter(d["kind"] for _, d in graph.nodes(data=True)).items())),
            tuple(sorted(Counter(d["relation"] for *_, d in graph.edges(data=True)).items())))


def _mapping(left, right):
    # A SET of labels would conflate [R,R,S] and [R,S,S] parallel edges.
    matcher = nx.isomorphism.MultiDiGraphMatcher if left.is_directed() else nx.isomorphism.MultiGraphMatcher
    match = matcher(left, right,
        node_match=lambda a, b: a["kind"] == b["kind"],
        edge_match=lambda a, b: Counter(d["relation"] for d in a.values()) ==
                                Counter(d["relation"] for d in b.values()))
    return next(match.isomorphisms_iter(), None)


class ShapeRegistry:
    """Single-writer exact dictionary. Similarity never changes symbol identity."""

    def __init__(self, namespace: str | None = None):
        self.namespace = namespace if namespace is not None else uuid4().hex
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("registry namespace must be a nonempty string")
        self._shapes: list[Shape] = []
        self._graphs: list[nx.MultiDiGraph] = []
        self._buckets = defaultdict(list)

    @property
    def shapes(self) -> tuple[Shape, ...]:
        return tuple(self._shapes)

    def intern(self, shape: Shape) -> tuple[int, tuple[int, ...]]:
        """Return (stable ID, prototype-node -> candidate-node permutation)."""
        n = len(shape.node_types)
        if not n:
            raise ValueError("a shape must have at least one vertex")
        for kind in shape.node_types:
            if kind == "ATOM":
                continue
            if not isinstance(kind, str) or not kind.startswith("S:"):
                raise ValueError("node type must be ATOM or a registered child S:<id>")
            child = kind[2:]
            if not child.isdecimal() or str(int(child)) != child or int(child) >= len(self._shapes):
                raise ValueError("unknown or forward child symbol; grammar must be acyclic")
        for u, v, relation in shape.edges:
            if type(u) is not int or type(v) is not int or not (0 <= u < n and 0 <= v < n):
                raise ValueError("shape edge endpoint is outside its vertex set")
            if not isinstance(relation, str):
                raise ValueError("relation identity must be a string, not a local integer")
        # Own immutable values; callers cannot mutate a published definition.
        shape = Shape(tuple(shape.node_types), tuple(sorted((u, v, r) for u, v, r in shape.edges)))
        graph = _graph(shape, "recursive")
        key = _bucket(graph)
        for symbol in self._buckets[key]:
            mapping = _mapping(self._graphs[symbol], graph)
            if mapping is not None:
                return symbol, tuple(mapping[i] for i in range(n))
        symbol = len(self._shapes)
        self._shapes.append(shape)
        self._graphs.append(graph)
        self._buckets[key].append(symbol)
        return symbol, tuple(range(n))

    def references(self, local_shapes: list[dict], relations: list[str]) -> list[dict]:
        """Intern codec-local shapes without changing binary stream slot order."""
        if len(set(relations)) != len(relations):
            raise ValueError("duplicate relation names")
        refs = []
        seen = set()
        for row in sorted(local_shapes, key=lambda s: s["shape_id"]):
            local_id = row["shape_id"]
            if type(local_id) is not int or local_id < 0 or local_id in seen:
                raise ValueError("invalid or duplicate local shape ID")
            seen.add(local_id)
            edges = []
            for u, v, rel in row["edges"]:
                if type(rel) is not int or not 0 <= rel < len(relations):
                    raise ValueError("invalid local relation ID")
                edges.append((u, v, relations[rel]))
            symbol, order = self.intern(Shape(tuple(row["node_types"]), tuple(edges)))
            ref = {"local": local_id, "symbol": symbol}
            if order != tuple(range(len(order))):
                ref["order"] = list(order)
            refs.append(ref)
        return refs

    def expand(self, refs: list[dict], relations: list[str]) -> list[dict]:
        """Recreate the existing codec's local shape table exactly."""
        relation_ids = {name: i for i, name in enumerate(relations)}
        if len(relation_ids) != len(relations):
            raise ValueError("duplicate relation names")
        rows, seen = [], set()
        for ref in refs:
            symbol, local = ref["symbol"], ref["local"]
            if type(symbol) is not int or not 0 <= symbol < len(self._shapes):
                raise ValueError("unknown global symbol")
            if type(local) is not int or local < 0 or local in seen:
                raise ValueError("invalid or duplicate local shape ID")
            seen.add(local)
            shape = self._shapes[symbol]
            n = len(shape.node_types)
            order = ref.get("order", list(range(n)))
            if any(type(x) is not int for x in order) or sorted(order) != list(range(n)):
                raise ValueError("invalid shape-node permutation")
            node_types = [""] * n
            for i, j in enumerate(order):
                node_types[j] = shape.node_types[i]
            if any(r not in relation_ids for _, _, r in shape.edges):
                raise ValueError("shared shape uses an unknown transition relation")
            edges = sorted([order[u], order[v], relation_ids[r]] for u, v, r in shape.edges)
            rows.append({"shape_id": local, "node_types": node_types, "edges": edges})
        return rows

    def classes(self, level: IsomorphismLevel = "recursive") -> dict[int, int]:
        """Projected-isomorphism classes, represented by earliest symbol IDs.

        These are not lossy grammar symbols or edit-distance thresholds.
        """
        if level not in LEVELS:
            raise ValueError(f"unknown isomorphism level: {level!r}")
        if level == "recursive":
            return {i: i for i in range(len(self._shapes))}
        representatives, buckets, result = {}, defaultdict(list), {}
        for symbol, shape in enumerate(self._shapes):
            graph = _graph(shape, level)
            key = _bucket(graph)
            found = next((other for other in buckets[key]
                          if _mapping(representatives[other], graph) is not None), None)
            if found is None:
                representatives[symbol] = graph
                buckets[key].append(symbol)
                found = symbol
            result[symbol] = found
        return result

    def dumps(self) -> str:
        return json.dumps({"contract": CONTRACT, "namespace": self.namespace,
                           "shapes": [{"node_types": s.node_types, "edges": s.edges}
                                      for s in self._shapes]},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def loads(cls, data: str | bytes) -> ShapeRegistry:
        payload = json.loads(data)
        if payload.get("contract") != CONTRACT:
            raise ValueError("unsupported shape identity contract")
        registry = cls(payload["namespace"])
        for expected, row in enumerate(payload["shapes"]):
            actual, _ = registry.intern(Shape(tuple(row["node_types"]),
                                             tuple(tuple(e) for e in row["edges"])))
            if actual != expected:
                raise ValueError("duplicate exact symbols in shared dictionary")
        return registry


def main(argv: list[str] | None = None) -> int:
    """Compare the same saved dictionary under an explicit projection."""
    import argparse
    import zipfile
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="v2 hierarchy ZIP or dictionary.json")
    parser.add_argument("--level", choices=LEVELS, default="recursive")
    args = parser.parse_args(argv)
    if zipfile.is_zipfile(args.archive):
        with zipfile.ZipFile(args.archive) as archive:
            registry = ShapeRegistry.loads(archive.read("dictionary.json"))
    else:
        registry = ShapeRegistry.loads(args.archive.read_bytes())
    print(json.dumps({"namespace": registry.namespace, "level": args.level,
                      "classes": registry.classes(args.level)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
