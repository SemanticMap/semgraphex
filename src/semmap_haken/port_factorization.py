"""Lossless factorization of stored graph-type prototypes (not the complete graph).

A shape contains typed, directed internal edges. Each exact type stores its
shape ID, VF2 node permutation and boundary-port residuals. External endpoint
identities are represented by the unchanged quotient/residual graph elsewhere.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict

import networkx as nx


def compact_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def normalize_prototype(type_id: str, prototype: dict) -> list:
    nodes = sorted(prototype["nodes"], key=lambda row: int(row["prototype_node"]))
    if [int(row["prototype_node"]) for row in nodes] != list(range(len(nodes))):
        raise ValueError(f"non-contiguous node IDs: {type_id}")
    node_rows = [
        [str(node["symbol_type"]), json.loads(node["boundary_ports"])]
        for node in nodes
    ]
    edges = sorted(
        [[int(edge["source"]), int(edge["target"]), str(edge["relation"])]
         for edge in prototype["edges"]],
        key=lambda edge: (edge[0], edge[1], edge[2]),
    )
    if any(u not in range(len(nodes)) or v not in range(len(nodes)) for u, v, _ in edges):
        raise ValueError(f"dangling edge: {type_id}")
    if len({(u, v) for u, v, _ in edges}) != len(edges):
        raise ValueError("parallel edges need explicit multigraph representation")
    return [type_id, node_rows, edges]


def denormalize_prototype(compact: list) -> dict:
    _, nodes, edges = compact
    return {
        "nodes": [
            {
                "prototype_node": position, "symbol_type": symbol,
                "boundary_ports": json.dumps(ports, ensure_ascii=False, separators=(",", ":")),
            }
            for position, (symbol, ports) in enumerate(nodes)
        ],
        "edges": [
            {"source": int(u), "target": int(v), "relation": str(rel)}
            for u, v, rel in edges
        ],
    }


def _graph(row: list) -> nx.DiGraph:
    _, nodes, edges = row
    graph = nx.DiGraph()
    graph.add_nodes_from(
        (i, {"symbol": str(node[0])}) for i, node in enumerate(nodes)
    )
    graph.add_edges_from(
        (int(u), int(v), {"relation": str(relation)}) for u, v, relation in edges
    )
    return graph


def _bucket(graph: nx.DiGraph) -> tuple:
    return (
        graph.number_of_nodes(), graph.number_of_edges(),
        tuple(sorted(
            (str(data["symbol"]), graph.in_degree(i), graph.out_degree(i))
            for i, data in graph.nodes(data=True)
        )),
        tuple(sorted(Counter(
            data["relation"] for _, _, data in graph.edges(data=True)
        ).items())),
    )


def factorize(prototypes: dict[str, dict]) -> tuple[dict, dict]:
    """Store each internal typed shape once; VF2 is the exact identity oracle."""
    baseline = {
        "format": "semmap-exact-prototypes-v1",
        "types": [normalize_prototype(tid, prototypes[tid]) for tid in sorted(prototypes)],
    }
    shapes: list[list] = []
    shape_graphs: list[nx.DiGraph] = []
    buckets: dict[tuple, list[int]] = defaultdict(list)
    variants: list[list] = []
    node_match = nx.algorithms.isomorphism.categorical_node_match("symbol", "ATOM")
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("relation", "__edge__")
    for row in baseline["types"]:
        type_id, nodes, edges = row
        candidate = _graph(row)
        shape_id: int | None = None
        mapping: dict[int, int] | None = None
        bucket = _bucket(candidate)
        for existing in buckets[bucket]:
            match = nx.algorithms.isomorphism.DiGraphMatcher(
                shape_graphs[existing], candidate,
                node_match=node_match, edge_match=edge_match,
            )
            if match.is_isomorphic():
                shape_id = existing
                mapping = {int(a): int(b) for a, b in match.mapping.items()}
                break
        if shape_id is None:
            shape_id = len(shapes)
            shapes.append([[str(node[0]) for node in nodes], edges])
            shape_graphs.append(candidate)
            buckets[bucket].append(shape_id)
            mapping = {i: i for i in range(len(nodes))}
        assert mapping is not None
        order = [mapping[i] for i in range(len(nodes))]
        if set(order) != set(range(len(nodes))):
            raise AssertionError(f"invalid VF2 permutation: {type_id}")
        ports = [[i, node[1]] for i, node in enumerate(nodes) if node[1]]
        variants.append([
            type_id, shape_id,
            None if order == list(range(len(nodes))) else order,
            ports,
        ])
    return baseline, {
        "format": "semmap-internal-shapes-plus-ports-v1",
        "shapes": shapes, "variants": variants,
    }


def restore(factored: dict) -> dict:
    result: list[list] = []
    for tid, shape_id, mapping, nonempty_ports in factored["variants"]:
        symbols, edges = factored["shapes"][int(shape_id)]
        n = len(symbols)
        order = list(range(n)) if mapping is None else [int(i) for i in mapping]
        if len(order) != n or set(order) != set(range(n)):
            raise ValueError(f"invalid node map: {tid}")
        ports = [[] for _ in range(n)]
        seen: set[int] = set()
        for position, payload in nonempty_ports:
            i = int(position)
            if i not in range(n) or i in seen or not payload:
                raise ValueError(f"invalid ports: {tid}")
            ports[i] = payload
            seen.add(i)
        nodes = [None] * n
        for i, j in enumerate(order):
            nodes[j] = [symbols[i], ports[j]]
        internal_edges = sorted(
            [[order[int(u)], order[int(v)], rel] for u, v, rel in edges],
            key=lambda edge: (edge[0], edge[1], edge[2]),
        )
        result.append([tid, nodes, internal_edges])
    return {"format": "semmap-exact-prototypes-v1", "types": result}


def snapshot(prototypes: dict[str, dict], frequencies: Counter[str]) -> dict:
    if not prototypes:
        raise ValueError("no prototypes")
    baseline, factored = factorize(prototypes)
    restored = restore(factored)
    if compact_json(restored) != compact_json(baseline):
        raise AssertionError("normalized lossless round-trip failed")
    for row in restored["types"]:
        if denormalize_prototype(row) != prototypes[row[0]]:
            raise AssertionError(f"source prototype round-trip failed: {row[0]}")
    raw_a, raw_b = compact_json(baseline), compact_json(factored)
    sizes = Counter(int(variant[1]) for variant in factored["variants"])
    shape_for = {str(tid): int(shape) for tid, shape, _, _ in factored["variants"]}
    return {
        "exact_types": len(prototypes),
        "internal_shapes": len(factored["shapes"]),
        "types_in_shared_shapes": sum(size for size in sizes.values() if size > 1),
        "occurrences_in_shared_shapes": sum(
            int(count) for tid, count in frequencies.items()
            if sizes[shape_for[tid]] > 1
        ),
        "baseline_json_bytes": len(raw_a),
        "factorized_json_bytes": len(raw_b),
        "json_saved_percent": round(100 * (1 - len(raw_b) / len(raw_a)), 3),
        "baseline_gzip_bytes": len(gzip.compress(raw_a, compresslevel=9, mtime=0)),
        "factorized_gzip_bytes": len(gzip.compress(raw_b, compresslevel=9, mtime=0)),
        "roundtrip_exact": True,
    }
