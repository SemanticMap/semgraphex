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
    """Degree/relation bucket + exact VF2; select cheapest port symmetry."""
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
    device: str = "cpu",
    gpu_batch_size: int = 500_000,
) -> dict[str, object]:
    """Write a complete decodable archive and return measured byte counts.

    The baseline is a separately serialized, DEFLATE-compressed exact record
    list. Neither compression ratio nor graphex convergence is presumed.
    """
    records = tuple(edges)
    groups = tuple(tuple(sorted(int(v) for v in group)) for group in figures)
    if device == "cpu":
        assigned = classify_edges(vertex_count, records, groups)
        actual_device = "cpu"
    else:
        from .graphex_components_gpu import classify_edges_accelerated
        assigned = classify_edges_accelerated(vertex_count, records, groups,
                                              device=device, batch_size=gpu_batch_size)
        import torch
        actual_device = "cuda" if torch.cuda.is_available() and records else "cpu"
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
    # W topology is reconstructed from the shared prototype and its
    # occurrence mapping. Only edge IDs and exact binary64 weights are stored
    # per W edge; S/I/R retain their exact directed payload.
    shape_lookup = {shape["shape_id"]: shape for shape in shapes}
    occurrences_by_owner = {occ["figure_index"]: occ for occ in occurrences}
    slots: dict[int, dict[tuple[int, int, str], list[int]]] = {}
    for occ in occurrences:
        shape = shape_lookup[occ["shape_id"]]
        fine = occ["shape_to_fine_nodes"]
        available: dict[tuple[int, int, str], list[int]] = defaultdict(list)
        for pos, (u, v, relation) in enumerate(shape["edges"]):
            available[(fine[u], fine[v], relation)].append(pos)
        slots[occ["figure_index"]] = available
    payloads: dict[str, list[list]] = {"W": [], "S": [], "I": [], "R": []}
    for item in assigned:
        e = item.edge
        if item.part == "W":
            matches = slots[item.owner].get((e.source, e.target, e.relation), [])
            if not matches:
                raise ValueError("shape prototype does not cover an internal edge")
            payloads["W"].append([e.edge_id, e.weight, item.owner, matches.pop(0)])
        else:
            payloads[item.part].append(
                [e.edge_id, e.source, e.target, e.relation, e.weight])
    if any(remaining for group in slots.values() for remaining in group.values()):
        raise ValueError("shape prototype has extra internal edges")
    # A sorted sequence of edge IDs is reconstructible from the payloads.
    # The old JSON array of >1M consecutive IDs cost >2 MB DEFLATE on CN100k.
    # Keep an explicit array only for nonmonotonic input order.
    ids = [e.edge_id for e in records]
    data["record_order"] = (
        "ascending_ids"
        if all(first < second for first, second in zip(ids, ids[1:]))
        else ids
    )
    payload = [[e.edge_id, e.source, e.target, e.relation, e.weight]
               for e in records]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        archive.writestr("manifest.json", _json(data))
        archive.writestr("w_shapes.json", _json(shapes))
        archive.writestr("occurrences.json", _json(occurrences))
        archive.writestr("payloads.json", _json(payloads))
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
    return {"format": FORMAT, "classification_device": actual_device,
            "archive_bytes": target.stat().st_size,
            "baseline_bytes": len(baseline_buffer.getvalue()),
            "net_saved_bytes": len(baseline_buffer.getvalue()) - target.stat().st_size,
            "shapes": len(shapes), "occurrences": len(occurrences),
            "edge_records": len(records), "partition": dict(by_part),
            "conditional_S_given_W": [
                {"shape_id": k[0], "s_pattern": k[1], "frequency": v}
                for k, v in sorted(conditional.items())],
            "stream_bits": bit_count, "roundtrip_exact": True}


def decode_graph(archive_path: str | Path) -> tuple[int, tuple[EdgeRecord, ...]]:
    """Reconstruct W from shape topology, all other records from payloads."""
    with zipfile.ZipFile(archive_path) as archive:
        meta = json.loads(archive.read("manifest.json"))
        if meta.get("format") != FORMAT:
            raise ValueError("unsupported archive format")
        payloads = json.loads(archive.read("payloads.json"))
        tokens = _unpack(archive.read("symbols.bin"), meta["symbol_bits"],
                         meta["edge_count"], meta["codes"])
        shapes = json.loads(archive.read("w_shapes.json"))
        occurrences = json.loads(archive.read("occurrences.json"))
    valid_shapes = {shape["shape_id"]: shape for shape in shapes}
    placements = {occ["figure_index"]: occ for occ in occurrences}
    if len(placements) != len(occurrences):
        raise ValueError("duplicate figure index")
    restored: dict[int, EdgeRecord] = {}
    derived: dict[int, str] = {}
    for edge_id, weight, owner, position in payloads["W"]:
        occurrence = placements[owner]
        shape = valid_shapes[occurrence["shape_id"]]
        fine = occurrence["shape_to_fine_nodes"]
        u, v, relation = shape["edges"][position]
        e = EdgeRecord(edge_id, fine[u], fine[v], relation, weight)
        if edge_id in restored:
            raise ValueError("duplicate edge ID")
        restored[edge_id], derived[edge_id] = e, "W:" + str(occurrence["shape_id"])
    for part in ("S", "I", "R"):
        for payload in payloads[part]:
            e = EdgeRecord(*payload)
            if e.edge_id in restored:
                raise ValueError("duplicate edge ID")
            restored[e.edge_id] = e
            if part == "S":
                owner = next((index for index, occ in placements.items()
                              if e.source in occ["shape_to_fine_nodes"]
                              or e.target in occ["shape_to_fine_nodes"]), None)
                if owner is None:
                    raise ValueError("S has no figure endpoint")
                nodes = set(placements[owner]["shape_to_fine_nodes"])
                direction = "out" if e.source in nodes else "in"
                derived[e.edge_id] = "S:" + _json([direction, e.relation]).decode("utf-8")
            elif part == "I":
                derived[e.edge_id] = "I:" + e.relation
            else:
                derived[e.edge_id] = "R"
    encoded_order = meta["record_order"]
    if encoded_order == "ascending_ids":
        order = sorted(restored)
    elif isinstance(encoded_order, list):
        order = encoded_order  # legacy archives and arbitrary input permutations
    else:
        raise ValueError("unsupported edge ordering")
    if len(order) != meta["edge_count"] or len(set(order)) != len(order):
        raise ValueError("invalid edge order")
    try:
        records = tuple(restored[edge_id] for edge_id in order)
        original_symbols = [derived[edge_id] for edge_id in order]
    except KeyError as exc:
        raise ValueError("missing edge from component payload") from exc
    if len(restored) != len(order) or original_symbols != tokens:
        raise ValueError("component payload and Huffman stream disagree")
    if any(not 0 <= e.source < meta["vertex_count"] or
           not 0 <= e.target < meta["vertex_count"] for e in records):
        raise ValueError("decoded edge endpoint outside graph")
    groups = [placements[i]["shape_to_fine_nodes"] for i in sorted(placements)]
    assigned = classify_edges(meta["vertex_count"], records, groups)
    if any(assigned[i].part != token[0] for i, token in enumerate(tokens)):
        raise ValueError("invalid W/S/I/R classification in archive")
    return int(meta["vertex_count"]), records
