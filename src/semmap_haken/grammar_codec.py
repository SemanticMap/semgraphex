"""Versioned recursive-ready lossless graph grammar codec.

The v2 archive is intentionally decoder-centric:

* reusable internal topology is stored once as InternalShape;
* boundary patterns are factorized as InterfaceVariant;
* each occurrence stores only a shape placement;
* exact internal weights/record IDs live in internal.bin;
* exact cross-boundary edges live in ports.bin;
* edges unrelated to accepted figures live in residual.bin.

Wishart/GloVe/graphex diagnostics are deliberately outside this archive.
"""
from __future__ import annotations

import io
import json
import math
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx

from .edge_table import EdgeTable
from .exact_edge_baseline import encode_edge_baseline_bytes
from .grammar_binary import (
    bits_to_float,
    float_to_bits,
    read_svarint,
    read_u64,
    read_uvarint,
    write_svarint,
    write_u64,
    write_uvarint,
)
from .grammar_streams import (
    STREAM_LAYOUT as COMPACT_STREAM_LAYOUT,
    read_internal as _read_internal_compact,
    read_occurrences as _read_occurrences_compact,
    read_ports as _read_ports_compact,
    read_residual as _read_residual_compact,
    write_internal as _write_internal_compact,
    write_occurrences as _write_occurrences_compact,
    write_ports as _write_ports_compact,
    write_residual as _write_residual_compact,
)
from .grammar_types import (
    GrammarRule,
    InterfaceVariant,
    InternalEdgePayload,
    InternalShape,
    Occurrence,
    PortBinding,
    PortSpec,
    ResidualEdge,
    ShapeEdge,
)
from .graphex_components import EdgeRecord

FORMAT = "semmap_grammar_exact_v2"


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_records(
    vertex_count: int,
    records: tuple[EdgeRecord, ...],
) -> None:
    if vertex_count < 0:
        raise ValueError("vertex_count must be non-negative")
    if len({int(row.edge_id) for row in records}) != len(records):
        raise ValueError("edge_id must be unique")
    for row in records:
        if not 0 <= int(row.source) < vertex_count:
            raise ValueError("edge source outside graph")
        if not 0 <= int(row.target) < vertex_count:
            raise ValueError("edge target outside graph")
        if not math.isfinite(float(row.weight)):
            raise ValueError("non-finite edge weight")


def _validate_figures(
    vertex_count: int,
    figures: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    used: set[int] = set()
    for figure in figures:
        nodes = tuple(sorted(int(node) for node in figure))
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("figures must be non-empty sets of vertices")
        if any(node < 0 or node >= vertex_count for node in nodes):
            raise ValueError("figure vertex outside graph")
        if used.intersection(nodes):
            raise ValueError("accepted figures must not overlap")
        used.update(nodes)
        groups.append(nodes)
    return tuple(groups)


def _shape_graph(
    nodes: tuple[int, ...],
    internal: list[EdgeRecord],
    node_types: Mapping[int, str],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node, symbol_type=str(node_types.get(node, "ATOM")))
    for edge in internal:
        graph.add_edge(
            int(edge.source),
            int(edge.target),
            relation=str(edge.relation),
        )
    return graph


def _shape_signature(
    graph: nx.MultiDiGraph,
    internal: list[EdgeRecord],
) -> tuple[object, ...]:
    relations = tuple(sorted(Counter(str(row.relation) for row in internal).items()))
    degrees = tuple(sorted(
        (
            int(graph.in_degree(node)),
            int(graph.out_degree(node)),
            str(graph.nodes[node]["symbol_type"]),
        )
        for node in graph.nodes
    ))
    return (
        int(graph.number_of_nodes()),
        int(graph.number_of_edges()),
        relations,
        degrees,
    )


def _external_for_figure(
    figure_index: int,
    owner: Mapping[int, int],
    records: tuple[EdgeRecord, ...],
) -> list[EdgeRecord]:
    result: list[EdgeRecord] = []
    for edge in records:
        source_owner = owner.get(int(edge.source))
        target_owner = owner.get(int(edge.target))
        if source_owner == figure_index and target_owner == figure_index:
            continue
        if source_owner == figure_index or target_owner == figure_index:
            result.append(edge)
    return result


def _port_profile(
    ordered_fine_nodes: tuple[int, ...],
    external: list[EdgeRecord],
    relation_to_id: Mapping[str, int],
) -> tuple[PortSpec, ...]:
    local = {fine: index for index, fine in enumerate(ordered_fine_nodes)}
    # InterfaceVariant is a reusable *port schema*, not an occurrence census.
    # Edge multiplicity is already preserved exactly by PortBinding rows.  A
    # set prevents otherwise identical shapes from fragmenting into different
    # variants merely because one occurrence has more external edges through
    # the same typed/directed port.
    ports: set[PortSpec] = set()
    for edge in external:
        if int(edge.source) in local:
            ports.add(
                PortSpec(
                    local_node=local[int(edge.source)],
                    relation_id=relation_to_id[str(edge.relation)],
                    direction="out",
                )
            )
        if int(edge.target) in local and int(edge.target) != int(edge.source):
            ports.add(
                PortSpec(
                    local_node=local[int(edge.target)],
                    relation_id=relation_to_id[str(edge.relation)],
                    direction="in",
                )
            )
        elif int(edge.target) in local and int(edge.target) == int(edge.source):
            raise AssertionError("unexpected cross-boundary self-loop")
    return tuple(sorted(ports))


def _shape_to_json(shape: InternalShape) -> dict[str, object]:
    return {
        "shape_id": shape.shape_id,
        "node_types": list(shape.node_types),
        "edges": [
            [edge.source, edge.target, edge.relation_id]
            for edge in shape.edges
        ],
    }


def _variant_to_json(variant: InterfaceVariant) -> dict[str, object]:
    return {
        "variant_id": variant.variant_id,
        "shape_id": variant.shape_id,
        "ports": [
            [port.local_node, port.relation_id, port.direction]
            for port in variant.ports
        ],
    }


def _occurrence_to_json(occurrence: Occurrence) -> dict[str, object]:
    return {
        "occurrence_id": occurrence.occurrence_id,
        "symbol_id": occurrence.symbol_id,
        "variant_id": occurrence.variant_id,
        "shape_to_fine_nodes": list(occurrence.shape_to_fine_nodes),
    }


def _rule_to_json(rule: GrammarRule) -> dict[str, object]:
    return {
        "symbol_id": rule.symbol_id,
        "shape_id": rule.shape_id,
        "child_symbols": list(rule.child_symbols),
    }


def _write_occurrences(rows: tuple[Occurrence, ...]) -> bytes:
    out = io.BytesIO()
    write_uvarint(out, len(rows))
    for row in rows:
        write_uvarint(out, row.occurrence_id)
        write_uvarint(out, row.symbol_id)
        write_uvarint(out, row.variant_id)
        write_uvarint(out, len(row.shape_to_fine_nodes))
        if not row.shape_to_fine_nodes:
            continue
        first = int(row.shape_to_fine_nodes[0])
        write_uvarint(out, first)
        previous = first
        for node in row.shape_to_fine_nodes[1:]:
            current = int(node)
            write_svarint(out, current - previous)
            previous = current
    return out.getvalue()


def _read_occurrences(data: bytes) -> tuple[Occurrence, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows: list[Occurrence] = []
    for _ in range(count):
        occurrence_id = read_uvarint(stream)
        symbol_id = read_uvarint(stream)
        variant_id = read_uvarint(stream)
        node_count = read_uvarint(stream)
        nodes: list[int] = []
        if node_count:
            first = read_uvarint(stream)
            nodes.append(first)
            previous = first
            for _ in range(node_count - 1):
                previous = previous + read_svarint(stream)
                if previous < 0:
                    raise ValueError("negative decoded fine node")
                nodes.append(previous)
        rows.append(
            Occurrence(
                occurrence_id=occurrence_id,
                symbol_id=symbol_id,
                variant_id=variant_id,
                shape_to_fine_nodes=tuple(nodes),
            )
        )
    if stream.read(1):
        raise ValueError("trailing occurrence payload bytes")
    return tuple(rows)


def _write_internal(rows: list[InternalEdgePayload]) -> bytes:
    out = io.BytesIO()
    write_uvarint(out, len(rows))
    for row in rows:
        write_uvarint(out, row.occurrence_id)
        write_uvarint(out, row.shape_edge_index)
        write_uvarint(out, row.edge_id)
        write_u64(out, row.weight_bits)
    return out.getvalue()


def _read_internal(data: bytes) -> tuple[InternalEdgePayload, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows = []
    for _ in range(count):
        rows.append(
            InternalEdgePayload(
                occurrence_id=read_uvarint(stream),
                shape_edge_index=read_uvarint(stream),
                edge_id=read_uvarint(stream),
                weight_bits=read_u64(stream),
            )
        )
    if stream.read(1):
        raise ValueError("trailing internal payload bytes")
    return tuple(rows)


def _write_ports(rows: list[PortBinding]) -> bytes:
    out = io.BytesIO()
    write_uvarint(out, len(rows))
    for row in rows:
        write_uvarint(out, row.occurrence_id)
        write_uvarint(out, row.local_port)
        write_uvarint(out, row.external_endpoint)
        write_uvarint(out, row.relation_id)
        write_uvarint(out, 1 if row.direction == "out" else 0)
        write_uvarint(out, row.edge_id)
        write_u64(out, row.weight_bits)
    return out.getvalue()


def _read_ports(data: bytes) -> tuple[PortBinding, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows = []
    for _ in range(count):
        occurrence_id = read_uvarint(stream)
        local_port = read_uvarint(stream)
        external_endpoint = read_uvarint(stream)
        relation_id = read_uvarint(stream)
        direction_code = read_uvarint(stream)
        if direction_code not in (0, 1):
            raise ValueError("invalid port direction")
        rows.append(
            PortBinding(
                occurrence_id=occurrence_id,
                local_port=local_port,
                external_endpoint=external_endpoint,
                relation_id=relation_id,
                direction="out" if direction_code else "in",
                edge_id=read_uvarint(stream),
                weight_bits=read_u64(stream),
            )
        )
    if stream.read(1):
        raise ValueError("trailing port payload bytes")
    return tuple(rows)


def _write_residual(rows: list[ResidualEdge]) -> bytes:
    out = io.BytesIO()
    write_uvarint(out, len(rows))
    for row in rows:
        write_uvarint(out, row.edge_id)
        write_uvarint(out, row.source)
        write_uvarint(out, row.target)
        write_uvarint(out, row.relation_id)
        write_u64(out, row.weight_bits)
    return out.getvalue()


def _read_residual(data: bytes) -> tuple[ResidualEdge, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows = []
    for _ in range(count):
        rows.append(
            ResidualEdge(
                edge_id=read_uvarint(stream),
                source=read_uvarint(stream),
                target=read_uvarint(stream),
                relation_id=read_uvarint(stream),
                weight_bits=read_u64(stream),
            )
        )
    if stream.read(1):
        raise ValueError("trailing residual payload bytes")
    return tuple(rows)


def _canonicalize_shapes(
    records: tuple[EdgeRecord, ...],
    groups: tuple[tuple[int, ...], ...],
    owner: Mapping[int, int],
    relation_to_id: Mapping[str, int],
    node_types: Mapping[int, str],
    grammar_relations: frozenset[str] | None = None,
) -> tuple[
    tuple[InternalShape, ...],
    tuple[InterfaceVariant, ...],
    tuple[GrammarRule, ...],
    tuple[Occurrence, ...],
    list[InternalEdgePayload],
]:
    shapes: list[InternalShape] = []
    shape_graphs: list[nx.MultiDiGraph] = []
    shape_buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
    variants: list[InterfaceVariant] = []
    variant_ids: dict[tuple[int, tuple[PortSpec, ...]], int] = {}
    occurrences: list[Occurrence] = []
    internal_payload: list[InternalEdgePayload] = []

    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "symbol_type", "ATOM"
    )
    edge_match = nx.algorithms.isomorphism.categorical_multiedge_match(
        "relation", ""
    )

    for occurrence_id, nodes in enumerate(groups):
        internal = [
            edge for edge in records
            if owner.get(int(edge.source)) == occurrence_id
            and owner.get(int(edge.target)) == occurrence_id
            and (
                grammar_relations is None
                or str(edge.relation) in grammar_relations
            )
        ]
        external = [
            edge
            for edge in _external_for_figure(occurrence_id, owner, records)
            if (
                grammar_relations is None
                or str(edge.relation) in grammar_relations
            )
        ]
        graph = _shape_graph(nodes, internal, node_types)
        signature = _shape_signature(graph, internal)

        selected_shape: int | None = None
        selected_order: tuple[int, ...] | None = None
        selected_criterion: tuple[object, ...] | None = None

        for shape_id in shape_buckets[signature]:
            matcher = nx.algorithms.isomorphism.MultiDiGraphMatcher(
                shape_graphs[shape_id],
                graph,
                node_match=node_match,
                edge_match=edge_match,
            )
            for mapping in matcher.isomorphisms_iter():
                ordered = tuple(
                    int(mapping[index])
                    for index in range(len(shapes[shape_id].node_types))
                )
                ports = _port_profile(ordered, external, relation_to_id)
                criterion = (
                    tuple(
                        (p.local_node, p.relation_id, p.direction)
                        for p in ports
                    ),
                    ordered,
                )
                if selected_criterion is None or criterion < selected_criterion:
                    selected_shape = shape_id
                    selected_order = ordered
                    selected_criterion = criterion

        if selected_shape is None:
            selected_shape = len(shapes)
            selected_order = tuple(sorted(nodes))
            fine_to_local = {
                fine: local for local, fine in enumerate(selected_order)
            }
            shape_edges = tuple(sorted(
                ShapeEdge(
                    fine_to_local[int(edge.source)],
                    fine_to_local[int(edge.target)],
                    relation_to_id[str(edge.relation)],
                )
                for edge in internal
            ))
            node_type_tuple = tuple(
                str(node_types.get(fine, "ATOM"))
                for fine in selected_order
            )
            shape = InternalShape(
                shape_id=selected_shape,
                node_types=node_type_tuple,
                edges=shape_edges,
            )
            shapes.append(shape)

            canonical_graph = nx.MultiDiGraph()
            for local, node_type in enumerate(node_type_tuple):
                canonical_graph.add_node(local, symbol_type=node_type)
            relations = tuple(sorted(relation_to_id, key=relation_to_id.get))
            for edge in shape_edges:
                canonical_graph.add_edge(
                    edge.source,
                    edge.target,
                    relation=relations[edge.relation_id],
                )
            shape_graphs.append(canonical_graph)
            shape_buckets[signature].append(selected_shape)

        assert selected_order is not None
        ports = _port_profile(selected_order, external, relation_to_id)
        variant_key = (selected_shape, ports)
        variant_id = variant_ids.get(variant_key)
        if variant_id is None:
            variant_id = len(variants)
            variant_ids[variant_key] = variant_id
            variants.append(
                InterfaceVariant(
                    variant_id=variant_id,
                    shape_id=selected_shape,
                    ports=ports,
                )
            )

        occurrence = Occurrence(
            occurrence_id=occurrence_id,
            symbol_id=selected_shape,
            variant_id=variant_id,
            shape_to_fine_nodes=selected_order,
        )
        occurrences.append(occurrence)

        shape = shapes[selected_shape]
        available: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, edge in enumerate(shape.edges):
            available[
                (
                    selected_order[edge.source],
                    selected_order[edge.target],
                    edge.relation_id,
                )
            ].append(index)
        for positions in available.values():
            positions.sort(reverse=True)

        for edge in sorted(internal, key=lambda item: int(item.edge_id)):
            key = (
                int(edge.source),
                int(edge.target),
                relation_to_id[str(edge.relation)],
            )
            if not available[key]:
                raise AssertionError("shape does not cover internal edge")
            slot = available[key].pop()
            internal_payload.append(
                InternalEdgePayload(
                    occurrence_id=occurrence_id,
                    shape_edge_index=slot,
                    edge_id=int(edge.edge_id),
                    weight_bits=float_to_bits(float(edge.weight)),
                )
            )
        if any(positions for positions in available.values()):
            raise AssertionError("shape has unmatched internal edges")

    rules = tuple(
        GrammarRule(
            symbol_id=shape.shape_id,
            shape_id=shape.shape_id,
            child_symbols=tuple(
                node_type if node_type != "ATOM" else None
                for node_type in shape.node_types
            ),
        )
        for shape in shapes
    )
    return (
        tuple(shapes),
        tuple(variants),
        rules,
        tuple(occurrences),
        internal_payload,
    )


def encode_grammar(
    vertex_count: int,
    edges: Iterable[EdgeRecord],
    figures: Iterable[Iterable[int]],
    output: str | Path,
    *,
    node_types: Mapping[int, str] | None = None,
    grammar_relations: Iterable[str] | None = None,
) -> dict[str, object]:
    """Encode a graph exactly using reusable shapes, ports, and residuals."""
    codec_started = time.perf_counter()
    records = tuple(edges)
    _validate_records(vertex_count, records)
    groups = _validate_figures(vertex_count, figures)
    node_types = dict(node_types or {})
    grammar_relation_set = (
        None
        if grammar_relations is None
        else frozenset(str(item) for item in grammar_relations)
    )
    if grammar_relation_set is not None and not grammar_relation_set:
        raise ValueError("grammar_relations must be non-empty when specified")

    table = EdgeTable.from_records(records)
    table.validate(vertex_count=vertex_count)
    relations = table.relations
    relation_to_id = {relation: index for index, relation in enumerate(relations)}
    if grammar_relation_set is not None:
        unknown_relations = grammar_relation_set.difference(relations)
        if unknown_relations:
            raise ValueError(
                "grammar_relations not present in source graph: "
                + ", ".join(sorted(unknown_relations))
            )

    owner: dict[int, int] = {}
    for occurrence_id, nodes in enumerate(groups):
        for node in nodes:
            owner[node] = occurrence_id

    canonicalization_started = time.perf_counter()
    (
        shapes,
        variants,
        rules,
        occurrences,
        internal_payload,
    ) = _canonicalize_shapes(
        records,
        groups,
        owner,
        relation_to_id,
        node_types,
        grammar_relation_set,
    )
    canonicalization_seconds = time.perf_counter() - canonicalization_started

    payload_started = time.perf_counter()
    port_payload: list[PortBinding] = []
    residual_payload: list[ResidualEdge] = []
    internal_ids = {item.edge_id for item in internal_payload}

    for edge in records:
        if int(edge.edge_id) in internal_ids:
            continue
        if (
            grammar_relation_set is not None
            and str(edge.relation) not in grammar_relation_set
        ):
            residual_payload.append(
                ResidualEdge(
                    edge_id=int(edge.edge_id),
                    source=int(edge.source),
                    target=int(edge.target),
                    relation_id=relation_to_id[str(edge.relation)],
                    weight_bits=float_to_bits(float(edge.weight)),
                )
            )
            continue
        source_owner = owner.get(int(edge.source))
        target_owner = owner.get(int(edge.target))
        if source_owner is None and target_owner is None:
            residual_payload.append(
                ResidualEdge(
                    edge_id=int(edge.edge_id),
                    source=int(edge.source),
                    target=int(edge.target),
                    relation_id=relation_to_id[str(edge.relation)],
                    weight_bits=float_to_bits(float(edge.weight)),
                )
            )
            continue

        if source_owner is not None and target_owner is not None:
            binding_owner = min(source_owner, target_owner)
        elif source_owner is not None:
            binding_owner = source_owner
        else:
            assert target_owner is not None
            binding_owner = target_owner

        occurrence = occurrences[binding_owner]
        fine_to_local = {
            fine: local
            for local, fine in enumerate(occurrence.shape_to_fine_nodes)
        }
        if int(edge.source) in fine_to_local:
            local_node = fine_to_local[int(edge.source)]
            external_endpoint = int(edge.target)
            direction = "out"
        elif int(edge.target) in fine_to_local:
            local_node = fine_to_local[int(edge.target)]
            external_endpoint = int(edge.source)
            direction = "in"
        else:
            raise AssertionError("selected port owner does not touch edge")

        port_payload.append(
            PortBinding(
                occurrence_id=binding_owner,
                local_port=local_node,
                external_endpoint=external_endpoint,
                relation_id=relation_to_id[str(edge.relation)],
                direction=direction,
                edge_id=int(edge.edge_id),
                weight_bits=float_to_bits(float(edge.weight)),
            )
        )

    payload_build_seconds = time.perf_counter() - payload_started
    ids = [int(row.edge_id) for row in records]
    record_order: str | list[int] = (
        "ascending_ids"
        if all(first < second for first, second in zip(ids, ids[1:]))
        else ids
    )
    manifest = {
        "format": FORMAT,
        "vertex_count": int(vertex_count),
        "edge_count": len(records),
        "relations": list(relations),
        "grammar_relations": (
            list(sorted(grammar_relation_set))
            if grammar_relation_set is not None
            else list(relations)
        ),
        "non_grammar_relations_as_residual": grammar_relation_set is not None,
        "record_order": record_order,
        "stream_layout": COMPACT_STREAM_LAYOUT,
        "shape_count": len(shapes),
        "variant_count": len(variants),
        "occurrence_count": len(occurrences),
        "internal_edge_records": len(internal_payload),
        "port_edge_records": len(port_payload),
        "residual_edge_records": len(residual_payload),
        "node_types": [
            [int(node), str(node_type)]
            for node, node_type in sorted(node_types.items())
            if str(node_type) != "ATOM"
        ],
        "exactness_scope": (
            "directed typed weighted EdgeRecord stream; binary64 weights "
            "preserved by exact IEEE bit pattern"
        ),
    }

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_write_started = time.perf_counter()
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("manifest.json", _json(manifest))
        archive.writestr(
            "shapes.json",
            _json([_shape_to_json(shape) for shape in shapes]),
        )
        archive.writestr(
            "variants.json",
            _json([_variant_to_json(variant) for variant in variants]),
        )
        archive.writestr(
            "rules.json",
            _json([_rule_to_json(rule) for rule in rules]),
        )
        shape_map = {shape.shape_id: shape for shape in shapes}
        variant_map = {variant.variant_id: variant for variant in variants}
        archive.writestr(
            "occurrences.bin",
            _write_occurrences_compact(occurrences, shape_map),
        )
        archive.writestr(
            "internal.bin",
            _write_internal_compact(internal_payload, occurrences, shape_map),
        )
        archive.writestr(
            "ports.bin",
            _write_ports_compact(port_payload, occurrences, variant_map),
        )
        archive.writestr(
            "residual.bin",
            _write_residual_compact(residual_payload),
        )
    archive_write_seconds = time.perf_counter() - archive_write_started

    # Legacy JSON+DEFLATE baseline remains for continuity with earlier
    # experiments, but the compact binary baseline is the authoritative
    # non-grammar storage comparator for new compression claims.
    baseline_started = time.perf_counter()
    baseline_buffer = io.BytesIO()
    with zipfile.ZipFile(
        baseline_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            "graph.json",
            _json({
                "vertex_count": vertex_count,
                "records": [
                    [
                        int(row.edge_id),
                        int(row.source),
                        int(row.target),
                        str(row.relation),
                        float(row.weight),
                    ]
                    for row in records
                ],
            }),
        )
    binary_baseline = encode_edge_baseline_bytes(vertex_count, records)
    baseline_seconds = time.perf_counter() - baseline_started

    verify_started = time.perf_counter()
    decoded_n, decoded_edges = decode_grammar(target)
    if decoded_n != vertex_count or decoded_edges != records:
        target.unlink(missing_ok=True)
        raise AssertionError("grammar_exact_v2 roundtrip failed")
    verify_decode_seconds = time.perf_counter() - verify_started

    with zipfile.ZipFile(target) as archive:
        entry_bytes = {
            info.filename: int(info.compress_size)
            for info in archive.infolist()
        }

    grammar_bytes = sum(
        entry_bytes.get(name, 0)
        for name in (
            "shapes.json",
            "variants.json",
            "rules.json",
            "occurrences.bin",
        )
    )
    payload_bytes = sum(
        entry_bytes.get(name, 0)
        for name in ("internal.bin", "ports.bin", "residual.bin")
    )
    baseline_json_bytes = len(baseline_buffer.getvalue())
    baseline_binary_bytes = len(binary_baseline)
    archive_bytes = target.stat().st_size
    return {
        "format": FORMAT,
        "archive_bytes": archive_bytes,
        # Backward-compatible alias for historical reports.
        "baseline_bytes": baseline_json_bytes,
        "baseline_json_bytes": baseline_json_bytes,
        "baseline_binary_bytes": baseline_binary_bytes,
        "net_saved_bytes": baseline_json_bytes - archive_bytes,
        "net_saved_vs_binary_bytes": baseline_binary_bytes - archive_bytes,
        "compression_ratio": (
            archive_bytes / baseline_json_bytes
            if baseline_json_bytes else None
        ),
        "compression_ratio_binary_baseline": (
            archive_bytes / baseline_binary_bytes
            if baseline_binary_bytes else None
        ),
        "shapes": len(shapes),
        "interface_variants": len(variants),
        "occurrences": len(occurrences),
        "internal_edge_records": len(internal_payload),
        "port_edge_records": len(port_payload),
        "residual_edge_records": len(residual_payload),
        "grammar_compressed_entry_bytes": grammar_bytes,
        "payload_compressed_entry_bytes": payload_bytes,
        "entry_compressed_bytes": entry_bytes,
        "grammar_relations": (
            list(sorted(grammar_relation_set))
            if grammar_relation_set is not None
            else list(relations)
        ),
        "grammar_relation_edge_records": sum(
            1
            for row in records
            if (
                grammar_relation_set is None
                or str(row.relation) in grammar_relation_set
            )
        ),
        "non_grammar_relation_edge_records": sum(
            1
            for row in records
            if (
                grammar_relation_set is not None
                and str(row.relation) not in grammar_relation_set
            )
        ),
        "timing_seconds": {
            "canonicalization": canonicalization_seconds,
            "payload_build": payload_build_seconds,
            "archive_write": archive_write_seconds,
            "baseline_build": baseline_seconds,
            "verify_decode": verify_decode_seconds,
            "total": time.perf_counter() - codec_started,
        },
        "roundtrip_exact": True,
    }


def decode_grammar(
    archive_path: str | Path,
) -> tuple[int, tuple[EdgeRecord, ...]]:
    """Decode an exact v2 grammar archive to the original EdgeRecord stream."""
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported grammar archive format")
        shapes_raw = json.loads(archive.read("shapes.json"))
        variants_raw = json.loads(archive.read("variants.json"))
        occurrences_data = archive.read("occurrences.bin")
        internal_data = archive.read("internal.bin")
        ports_data = archive.read("ports.bin")
        residual_data = archive.read("residual.bin")

    relations = tuple(str(item) for item in manifest["relations"])
    shapes = {
        int(item["shape_id"]): InternalShape(
            shape_id=int(item["shape_id"]),
            node_types=tuple(str(x) for x in item["node_types"]),
            edges=tuple(
                ShapeEdge(int(edge[0]), int(edge[1]), int(edge[2]))
                for edge in item["edges"]
            ),
        )
        for item in shapes_raw
    }
    variants = {
        int(item["variant_id"]): InterfaceVariant(
            variant_id=int(item["variant_id"]),
            shape_id=int(item["shape_id"]),
            ports=tuple(
                PortSpec(int(port[0]), int(port[1]), str(port[2]))
                for port in item["ports"]
            ),
        )
        for item in variants_raw
    }
    stream_layout = manifest.get("stream_layout")
    if stream_layout == COMPACT_STREAM_LAYOUT:
        occurrence_rows = _read_occurrences_compact(
            occurrences_data,
            shapes,
        )
        internal = _read_internal_compact(
            internal_data,
            occurrence_rows,
            shapes,
        )
        ports = _read_ports_compact(
            ports_data,
            occurrence_rows,
            variants,
        )
        residuals = _read_residual_compact(residual_data)
    elif stream_layout is None:
        # Backward compatibility for v2 archives written before stream layout
        # versioning was introduced on this development branch.
        occurrence_rows = _read_occurrences(occurrences_data)
        internal = _read_internal(internal_data)
        ports = _read_ports(ports_data)
        residuals = _read_residual(residual_data)
    else:
        raise ValueError(f"unsupported grammar stream layout: {stream_layout}")
    occurrences = {
        int(item.occurrence_id): item
        for item in occurrence_rows
    }

    restored: dict[int, EdgeRecord] = {}

    for row in internal:
        occurrence = occurrences.get(row.occurrence_id)
        if occurrence is None:
            raise ValueError("internal payload references missing occurrence")
        shape = shapes.get(occurrence.symbol_id)
        if shape is None:
            raise ValueError("occurrence references missing shape")
        if not 0 <= row.shape_edge_index < len(shape.edges):
            raise ValueError("internal payload references missing shape edge")
        shape_edge = shape.edges[row.shape_edge_index]
        source = occurrence.shape_to_fine_nodes[shape_edge.source]
        target = occurrence.shape_to_fine_nodes[shape_edge.target]
        if row.edge_id in restored:
            raise ValueError("duplicate edge ID")
        restored[row.edge_id] = EdgeRecord(
            row.edge_id,
            source,
            target,
            relations[shape_edge.relation_id],
            bits_to_float(row.weight_bits),
        )

    for row in ports:
        occurrence = occurrences.get(row.occurrence_id)
        if occurrence is None:
            raise ValueError("port payload references missing occurrence")
        variant = variants.get(occurrence.variant_id)
        if variant is None:
            raise ValueError("occurrence references missing interface variant")
        if not 0 <= row.local_port < len(occurrence.shape_to_fine_nodes):
            raise ValueError("port references missing local node")
        expected = PortSpec(
            row.local_port,
            row.relation_id,
            row.direction,
        )
        if expected not in variant.ports:
            raise ValueError("port binding not allowed by interface variant")
        local_node = occurrence.shape_to_fine_nodes[row.local_port]
        if row.direction == "out":
            source, target = local_node, row.external_endpoint
        else:
            source, target = row.external_endpoint, local_node
        if row.edge_id in restored:
            raise ValueError("duplicate edge ID")
        restored[row.edge_id] = EdgeRecord(
            row.edge_id,
            source,
            target,
            relations[row.relation_id],
            bits_to_float(row.weight_bits),
        )

    for row in residuals:
        if row.edge_id in restored:
            raise ValueError("duplicate edge ID")
        restored[row.edge_id] = EdgeRecord(
            row.edge_id,
            row.source,
            row.target,
            relations[row.relation_id],
            bits_to_float(row.weight_bits),
        )

    if len(restored) != int(manifest["edge_count"]):
        raise ValueError("decoded edge count differs")

    encoded_order = manifest["record_order"]
    if encoded_order == "ascending_ids":
        order = sorted(restored)
    elif isinstance(encoded_order, list):
        order = [int(item) for item in encoded_order]
    else:
        raise ValueError("unsupported record order")

    if len(order) != len(restored) or len(set(order)) != len(order):
        raise ValueError("invalid record order")
    try:
        records = tuple(restored[edge_id] for edge_id in order)
    except KeyError as error:
        raise ValueError("record order references missing edge") from error

    vertex_count = int(manifest["vertex_count"])
    if any(
        row.source < 0
        or row.target < 0
        or row.source >= vertex_count
        or row.target >= vertex_count
        for row in records
    ):
        raise ValueError("decoded endpoint outside graph")
    return vertex_count, records


def inspect_grammar_archive(archive_path: str | Path) -> dict[str, object]:
    """Return decoder-relevant archive metadata without expanding the graph."""
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported grammar archive format")
        return {
            **manifest,
            "entry_compressed_bytes": {
                info.filename: int(info.compress_size)
                for info in archive.infolist()
            },
        }
