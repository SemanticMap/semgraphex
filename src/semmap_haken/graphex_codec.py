"""Offline, versioned lossless *CSR-record* graphex codec.

The byte count includes the complete ZIP container, dictionaries, Huffman
codebooks, node addresses, weights, relation strings and residuals. The
statistical (W,S,I) graphex is deliberately not inferred here.
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx

from .graph_mdl import build_canonical_huffman_codes
from .graphex_components import EdgeRecord, classify_edges, validate_partition

FORMAT = "graphex_exact_v1"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pack(symbols: Iterable[str], codes: Mapping[str, str]) -> tuple[bytes, int]:
    pending = 0
    bits = 0
    out = bytearray()
    count = 0
    for symbol in symbols:
        for digit in codes[symbol]:
            pending = (pending << 1) | (digit == "1")
            bits += 1
            count += 1
            if bits == 8:
                out.append(pending)
                pending = bits = 0
    if bits:
        out.append(pending << (8 - bits))
    return bytes(out), count


def _unpack(data: bytes, bit_count: int, count: int,
            codes: Mapping[str, str]) -> list[str]:
    inverse = {code: symbol for symbol, code in codes.items()}
    if len(inverse) != len(codes):
        raise ValueError("non-prefix-free Huffman codebook")
    result: list[str] = []
    prefix = ""
    for bit in range(bit_count):
        prefix += "1" if data[bit // 8] & (1 << (7 - bit % 8)) else "0"
        if prefix in inverse:
            result.append(inverse[prefix])
            prefix = ""
    if prefix or len(result) != count or bit_count > len(data) * 8:
        raise ValueError("corrupted Huffman stream")
    return result


def _shape_graph(nodes: tuple[int, ...], internal: list[EdgeRecord],
                 node_types: Mapping[int, str]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node, symbol_type=node_types.get(node, "ATOM"))
    for edge in internal:
        graph.add_edge(edge.source, edge.target, relation=edge.relation)
    return graph


def _canonical_shapes(
    figures: tuple[tuple[int, ...], ...],
    edges: tuple[EdgeRecord, ...],
    assigned: tuple,
    node_types: Mapping[int, str],
) -> tuple[list[dict], list[dict]]:
    """WL bucket + exact VF2; choose symmetry mapping by external port cost."""
    shapes: list[dict] = []
    graphs: list[nx.MultiDiGraph] = []
    buckets: dict[tuple, list[int]] = defaultdict(list)
    occurrences: list[dict] = []
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "symbol_type", "ATOM")
    edge_match = nx.algorithms.isomorphism.categorical_multiedge_match(
        "relation", "")
    for index, nodes in enumerate(figures):
        internal = [item.edge for item in assigned
                    if item.part == "W" and item.owner == index]
        graph = _shape_graph(nodes, internal, node_types)
        relation_counts = tuple(sorted(Counter(e.relation for e in internal).items()))
        # Degree+relation partition is a safe prefilter; VF2 decides equality.
        signature = (len(nodes), len(internal), relation_counts,
                     tuple(sorted((graph.in_degree(n), graph.out_degree(n),
                                   graph.nodes[n]["symbol_type"]) for n in nodes)))
        chosen = None
        mapping = None
        external = [item.edge for item in assigned if item.part == "S"
                    and item.owner == index]
        for shape_id in buckets[signature]:
            matcher = nx.algorithms.isomorphism.MultiDiGraphMatcher(
                graphs[shape_id], graph, node_match=node_match,
                edge_match=edge_match)
            # Port profile is the secondary canonicalization key, not
            # another graph-isomorphism constraint.
            for candidate in matcher.isomorphisms_iter():
                ordered = tuple(candidate[n] for n in shapes[shape_id]["node_order"])
                reverse = {node: position for position, node in enumerate(ordered)}
                port_profile = tuple(sorted(
                    (reverse[e.source] if e.source in reverse else reverse[e.target],
                     e.relation, e.source in reverse)
                    for e in external))
                criterion = (len(_json(port_profile)), port_profile, ordered)
                if mapping is None or criterion < mapping[0]:
                    chosen, mapping = shape_id, (criterion, ordered)
        if chosen is None:
            chosen = len(shapes)
            ordered = tuple(sorted(nodes))
            graph_copy = nx.relabel_nodes(graph, {v: i for i, v in enumerate(ordered)},
                                         copy=True)
            graphs.append(graph_copy)
            shapes.append({
                "shape_id": chosen,
                "node_order": list(range(len(ordered))),
                "node_types": [node_types.get(v, "ATOM") for v in ordered],
                "edges": sorted([[ordered.index(e.source), ordered.index(e.target),
                                  e.relation] for e in internal]),
            })
            buckets[signature].append(chosen)
            mapping = ((0, (), ordered), ordered)
        occurrences.append({"shape_id": chosen, "shape_to_fine_nodes": list(mapping[1]),
                            "figure_index": index})
    return shapes, occurrences


def encode_graph(
    vertex_count: int,
    edges: Iterable[EdgeRecord],
    figures: Iterable[Iterable[int]],
    output: str | Path,
    *,
    node_types: Mapping[int, str] | None = None,
    source_type_ids: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Write a complete decodable archive and return measured byte counts.

    The baseline is a separately serialized, DEFLATE-compressed exact record
    list. Neither compression ratio nor graphex convergence is presumed.
    """
    records = tuple(edges)
    groups = tuple(tuple(sorted(int(v) for v in group)) for group in figures)
    assigned = classify_edges(vertex_count, records, groups)
    validate_partition(records, assigned)
    shapes, occurrences = _canonical_shapes(groups, records, assigned,
                                            node_types or {})
    for item in occurrences:
        if source_type_ids and item["figure_index"] in source_type_ids:
            item["source_type_id"] = source_type_ids[item["figure_index"]]
    # Symbols are attached to actual edge records; their frequencies are
    # measured on this stream only, never summed across hierarchy levels.
    symbols: list[str] = []
    by_part: Counter[str] = Counter()
    conditional: Counter[tuple[int, str]] = Counter()
    for item in assigned:
        edge = item.edge
        by_part[item.part] += 1
        if item.part == "W":
            token = "W:" + str(occurrences[item.owner]["shape_id"])
        elif item.part == "S":
            local = set(groups[item.owner])
            direction = "out" if edge.source in local else "in"
            token = "S:" + _json([direction, edge.relation]).decode("utf-8")
            conditional[(occurrences[item.owner]["shape_id"], token)] += 1
        elif item.part == "I":
            token = "I:" + edge.relation
        else:
            token = "R"
        symbols.append(token)
    codes = build_canonical_huffman_codes(Counter(symbols))
    packed, bit_count = _pack(symbols, codes)
    data = {"format": FORMAT, "vertex_count": vertex_count,
            "nodes": [[i, (node_types or {}).get(i, "ATOM")]
                      for i in range(vertex_count)],
            "edge_count": len(records), "symbol_bits": bit_count,
            "codes": codes}
    # Record IDs and exact weight bit patterns are retained in the payload.
    # JSON float roundtrip preserves Python binary64 values; nonfinite
    # weights are rejected by allow_nan=False.
    payload = [[e.edge_id, e.source, e.target, e.relation, e.weight]
               for e in records]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        archive.writestr("manifest.json", _json(data))
        archive.writestr("w_shapes.json", _json(shapes))
        archive.writestr("occurrences.json", _json(occurrences))
        archive.writestr("records.json", _json(payload))
        archive.writestr("symbols.bin", packed)
    # The raw baseline has the same vertex metadata and exact edge payload.
    baseline_buffer = io.BytesIO()
    with zipfile.ZipFile(baseline_buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        archive.writestr("graph.json", _json({"vertex_count": vertex_count,
                                            "nodes": data["nodes"],
                                            "records": payload}))
    recovered_n, recovered = decode_graph(target)
    if recovered_n != vertex_count or recovered != records:
        target.unlink(missing_ok=True)
        raise AssertionError("graphex codec failed exact roundtrip")
    return {"format": FORMAT, "archive_bytes": target.stat().st_size,
            "baseline_bytes": len(baseline_buffer.getvalue()),
            "net_saved_bytes": len(baseline_buffer.getvalue()) - target.stat().st_size,
            "shapes": len(shapes), "occurrences": len(occurrences),
            "edge_records": len(records), "partition": dict(by_part),
            "conditional_S_given_W": [
                {"shape_id": k[0], "s_pattern": k[1], "frequency": v}
                for k, v in sorted(conditional.items())],
            "stream_bits": bit_count, "roundtrip_exact": True}


def decode_graph(archive_path: str | Path) -> tuple[int, tuple[EdgeRecord, ...]]:
    with zipfile.ZipFile(archive_path) as archive:
        meta = json.loads(archive.read("manifest.json"))
        if meta.get("format") != FORMAT:
            raise ValueError("unsupported archive format")
        raw = json.loads(archive.read("records.json"))
        tokens = _unpack(archive.read("symbols.bin"), meta["symbol_bits"],
                         meta["edge_count"], meta["codes"])
        shapes = json.loads(archive.read("w_shapes.json"))
        occurrences = json.loads(archive.read("occurrences.json"))
    if len(raw) != len(tokens):
        raise ValueError("truncated record payload")
    valid_shapes = {str(item["shape_id"]) for item in shapes}
    for token in tokens:
        if token.startswith("W:") and token[2:] not in valid_shapes:
            raise ValueError("unknown W shape")
        if not (token == "R" or token.startswith(("W:", "S:", "I:"))):
            raise ValueError("invalid component symbol")
    if len({item["figure_index"] for item in occurrences}) != len(occurrences):
        raise ValueError("duplicate figure")
    records = tuple(EdgeRecord(*record) for record in raw)
    if len({edge.edge_id for edge in records}) != len(records):
        raise ValueError("duplicate edge ID")
    return int(meta["vertex_count"]), records
