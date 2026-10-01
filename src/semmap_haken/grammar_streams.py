"""Compact deterministic streams for semmap_grammar_exact_v2.

The archive grammar already determines several fields that older v2 streams
serialized for every row.  This layout removes those derivable fields while
keeping exact record IDs, endpoints and IEEE-754 weight bits.

Layout name: grouped_delta_v1
"""
from __future__ import annotations

import io
from collections import defaultdict
from typing import Mapping, Sequence

from .grammar_binary import (
    read_svarint,
    read_u64,
    read_uvarint,
    write_svarint,
    write_u64,
    write_uvarint,
)
from .grammar_types import (
    InterfaceVariant,
    InternalEdgePayload,
    InternalShape,
    Occurrence,
    PortBinding,
    PortSpec,
    ResidualEdge,
)

STREAM_LAYOUT = "grouped_delta_v1"


def write_occurrences(
    rows: Sequence[Occurrence],
    shapes: Mapping[int, InternalShape],
) -> bytes:
    """Encode occurrence IDs and node counts implicitly from stream order/rule."""
    out = io.BytesIO()
    write_uvarint(out, len(rows))
    previous_anchor = 0
    previous_symbol = 0
    previous_variant = 0
    for expected_id, row in enumerate(rows):
        if int(row.occurrence_id) != expected_id:
            raise ValueError("compact occurrences require contiguous occurrence IDs")
        shape = shapes.get(int(row.symbol_id))
        if shape is None:
            raise ValueError("occurrence references missing shape")
        if len(row.shape_to_fine_nodes) != len(shape.node_types):
            raise ValueError("occurrence node mapping differs from shape arity")
        symbol_id = int(row.symbol_id)
        variant_id = int(row.variant_id)
        write_svarint(out, symbol_id - previous_symbol)
        write_svarint(out, variant_id - previous_variant)
        previous_symbol = symbol_id
        previous_variant = variant_id
        if not row.shape_to_fine_nodes:
            continue
        first = int(row.shape_to_fine_nodes[0])
        write_svarint(out, first - previous_anchor)
        previous_anchor = first
        previous = first
        for node in row.shape_to_fine_nodes[1:]:
            current = int(node)
            write_svarint(out, current - previous)
            previous = current
    return out.getvalue()


def read_occurrences(
    data: bytes,
    shapes: Mapping[int, InternalShape],
) -> tuple[Occurrence, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows: list[Occurrence] = []
    previous_anchor = 0
    previous_symbol = 0
    previous_variant = 0
    for occurrence_id in range(count):
        symbol_id = previous_symbol + read_svarint(stream)
        variant_id = previous_variant + read_svarint(stream)
        if symbol_id < 0 or variant_id < 0:
            raise ValueError("negative decoded symbol or variant ID")
        previous_symbol = symbol_id
        previous_variant = variant_id
        shape = shapes.get(symbol_id)
        if shape is None:
            raise ValueError("occurrence references missing shape")
        node_count = len(shape.node_types)
        nodes: list[int] = []
        if node_count:
            first = previous_anchor + read_svarint(stream)
            if first < 0:
                raise ValueError("negative decoded fine node")
            previous_anchor = first
            nodes.append(first)
            previous = first
            for _ in range(node_count - 1):
                previous += read_svarint(stream)
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
        raise ValueError("trailing compact occurrence payload bytes")
    return tuple(rows)


def write_internal(
    rows: Sequence[InternalEdgePayload],
    occurrences: Sequence[Occurrence],
    shapes: Mapping[int, InternalShape],
) -> bytes:
    """Encode occurrence/shape-edge coordinates implicitly.

    The deterministic sequence is occurrence order followed by shape-edge
    index. Only record-ID deltas and exact weight bits remain in the stream.
    """
    by_slot: dict[tuple[int, int], InternalEdgePayload] = {}
    for row in rows:
        key = (int(row.occurrence_id), int(row.shape_edge_index))
        if key in by_slot:
            raise ValueError("duplicate internal payload slot")
        by_slot[key] = row

    out = io.BytesIO()
    previous_edge_id = 0
    expected_rows = 0
    for occurrence in occurrences:
        shape = shapes.get(int(occurrence.symbol_id))
        if shape is None:
            raise ValueError("occurrence references missing shape")
        for shape_edge_index in range(len(shape.edges)):
            expected_rows += 1
            key = (int(occurrence.occurrence_id), shape_edge_index)
            row = by_slot.get(key)
            if row is None:
                raise ValueError("missing internal payload slot")
            edge_id = int(row.edge_id)
            write_svarint(out, edge_id - previous_edge_id)
            previous_edge_id = edge_id
            write_u64(out, int(row.weight_bits))
    if len(by_slot) != expected_rows:
        raise ValueError("internal payload has unexpected slots")
    return out.getvalue()


def read_internal(
    data: bytes,
    occurrences: Sequence[Occurrence],
    shapes: Mapping[int, InternalShape],
) -> tuple[InternalEdgePayload, ...]:
    stream = io.BytesIO(data)
    rows: list[InternalEdgePayload] = []
    previous_edge_id = 0
    for occurrence in occurrences:
        shape = shapes.get(int(occurrence.symbol_id))
        if shape is None:
            raise ValueError("occurrence references missing shape")
        for shape_edge_index in range(len(shape.edges)):
            edge_id = previous_edge_id + read_svarint(stream)
            if edge_id < 0:
                raise ValueError("negative decoded edge ID")
            previous_edge_id = edge_id
            rows.append(
                InternalEdgePayload(
                    occurrence_id=int(occurrence.occurrence_id),
                    shape_edge_index=shape_edge_index,
                    edge_id=edge_id,
                    weight_bits=read_u64(stream),
                )
            )
    if stream.read(1):
        raise ValueError("trailing compact internal payload bytes")
    return tuple(rows)


def write_ports(
    rows: Sequence[PortBinding],
    occurrences: Sequence[Occurrence],
    variants: Mapping[int, InterfaceVariant],
) -> bytes:
    """Group port bindings by occurrence and reference the variant port schema."""
    by_occurrence: dict[int, list[PortBinding]] = defaultdict(list)
    for row in rows:
        by_occurrence[int(row.occurrence_id)].append(row)

    out = io.BytesIO()
    previous_edge_id = 0
    for occurrence in occurrences:
        occurrence_id = int(occurrence.occurrence_id)
        variant = variants.get(int(occurrence.variant_id))
        if variant is None:
            raise ValueError("occurrence references missing interface variant")
        schema_index = {port: index for index, port in enumerate(variant.ports)}
        group = sorted(
            by_occurrence.get(occurrence_id, ()),
            key=lambda row: int(row.edge_id),
        )
        write_uvarint(out, len(group))
        previous_endpoint = 0
        for row in group:
            spec = PortSpec(
                int(row.local_port),
                int(row.relation_id),
                row.direction,
            )
            try:
                port_index = schema_index[spec]
            except KeyError as error:
                raise ValueError("port binding absent from interface schema") from error
            write_uvarint(out, port_index)
            endpoint = int(row.external_endpoint)
            write_svarint(out, endpoint - previous_endpoint)
            previous_endpoint = endpoint
            edge_id = int(row.edge_id)
            write_svarint(out, edge_id - previous_edge_id)
            previous_edge_id = edge_id
            write_u64(out, int(row.weight_bits))
    unexpected = set(by_occurrence) - {
        int(row.occurrence_id) for row in occurrences
    }
    if unexpected:
        raise ValueError("port payload references missing occurrence")
    return out.getvalue()


def read_ports(
    data: bytes,
    occurrences: Sequence[Occurrence],
    variants: Mapping[int, InterfaceVariant],
) -> tuple[PortBinding, ...]:
    stream = io.BytesIO(data)
    rows: list[PortBinding] = []
    previous_edge_id = 0
    for occurrence in occurrences:
        variant = variants.get(int(occurrence.variant_id))
        if variant is None:
            raise ValueError("occurrence references missing interface variant")
        count = read_uvarint(stream)
        previous_endpoint = 0
        for _ in range(count):
            port_index = read_uvarint(stream)
            if port_index >= len(variant.ports):
                raise ValueError("port schema index outside interface variant")
            spec = variant.ports[port_index]
            endpoint = previous_endpoint + read_svarint(stream)
            if endpoint < 0:
                raise ValueError("negative decoded external endpoint")
            previous_endpoint = endpoint
            edge_id = previous_edge_id + read_svarint(stream)
            if edge_id < 0:
                raise ValueError("negative decoded edge ID")
            previous_edge_id = edge_id
            rows.append(
                PortBinding(
                    occurrence_id=int(occurrence.occurrence_id),
                    local_port=int(spec.local_node),
                    external_endpoint=endpoint,
                    relation_id=int(spec.relation_id),
                    direction=spec.direction,
                    edge_id=edge_id,
                    weight_bits=read_u64(stream),
                )
            )
    if stream.read(1):
        raise ValueError("trailing compact port payload bytes")
    return tuple(rows)


def write_residual(rows: Sequence[ResidualEdge]) -> bytes:
    """Delta-code residual record IDs and endpoints in record-ID order."""
    ordered = sorted(rows, key=lambda row: int(row.edge_id))
    out = io.BytesIO()
    write_uvarint(out, len(ordered))
    previous_edge_id = 0
    previous_source = 0
    for row in ordered:
        edge_id = int(row.edge_id)
        write_uvarint(out, edge_id - previous_edge_id)
        previous_edge_id = edge_id
        source = int(row.source)
        target = int(row.target)
        write_svarint(out, source - previous_source)
        previous_source = source
        write_svarint(out, target - source)
        write_uvarint(out, int(row.relation_id))
        write_u64(out, int(row.weight_bits))
    return out.getvalue()


def read_residual(data: bytes) -> tuple[ResidualEdge, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows: list[ResidualEdge] = []
    previous_edge_id = 0
    previous_source = 0
    for _ in range(count):
        edge_id = previous_edge_id + read_uvarint(stream)
        previous_edge_id = edge_id
        source = previous_source + read_svarint(stream)
        if source < 0:
            raise ValueError("negative decoded residual source")
        previous_source = source
        target = source + read_svarint(stream)
        if target < 0:
            raise ValueError("negative decoded residual target")
        relation_id = read_uvarint(stream)
        rows.append(
            ResidualEdge(
                edge_id=edge_id,
                source=source,
                target=target,
                relation_id=relation_id,
                weight_bits=read_u64(stream),
            )
        )
    if stream.read(1):
        raise ValueError("trailing compact residual payload bytes")
    return tuple(rows)
