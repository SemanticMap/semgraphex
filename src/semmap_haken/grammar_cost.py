"""Logical bit-cost model aligned with grammar_exact_v2 record layout.

This is not a prediction of DEFLATE container bytes. It is an exact accounting
of the codec's logical fields under the current varint/binary64 layout, with
conservative maxima for values unavailable during candidate scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .grammar_binary import encoded_uvarint_size
from .wishart_metrics import EgoCandidate


@dataclass(frozen=True)
class GrammarCodeCost:
    raw_bits: float
    encoded_bits: float
    rule_bits: float
    occurrence_bits: float
    internal_payload_bits: float
    port_payload_bits: float

    @property
    def gain_bits(self) -> float:
        return float(self.raw_bits - self.encoded_bits)


def _varint_bits(value: int) -> int:
    return 8 * encoded_uvarint_size(max(0, int(value)))


def _signed_delta_bits(value: int) -> int:
    value = int(value)
    zigzag = 2 * value if value >= 0 else -2 * value - 1
    return _varint_bits(zigzag)


def estimate_grammar_occurrence_cost(
    candidate: EgoCandidate,
    *,
    occurrence_nodes: Sequence[int],
    support: int,
    graph_node_count: int,
    relation_count: int,
    edge_record_count: int,
    type_code_bits: float,
) -> GrammarCodeCost:
    """Estimate logical v2 fields for one exact occurrence.

    Boundary signature counts are used for port records. External endpoint IDs
    are not known at this stage, so their varint cost is conservatively priced
    at the largest graph-node ID. Edge IDs are similarly priced at the largest
    source-level record ID.
    """
    support = max(1, int(support))
    relation_count = max(1, int(relation_count))
    edge_record_count = max(1, int(edge_record_count))
    nodes = tuple(int(node) for node in occurrence_nodes)

    internal_records = int(
        sum(int(layer.nnz) for layer in candidate.relation_layers.values())
    )
    boundary_records = int(
        sum(int(count) for _, _, _, count in candidate.boundary_signature)
    )

    max_node = max(0, int(graph_node_count) - 1)
    max_edge = max(0, edge_record_count - 1)
    relation_max = max(0, relation_count - 1)
    node_bits = _varint_bits(max_node)
    edge_id_bits = _varint_bits(max_edge)
    relation_bits = _varint_bits(relation_max)

    # A plain exact edge stream needs IDs/endpoints/relation/binary64 weight.
    raw_edge_bits = (
        edge_id_bits + 2 * node_bits + relation_bits + 64
    )
    raw_bits = float(
        (internal_records + boundary_records) * raw_edge_bits
    )

    # Rule definition: local endpoints + relation IDs, plus one byte-equivalent
    # child/node-type token per local node. The one-time rule is amortized over
    # all full-graph occurrences of the type.
    local_node_max = max(0, int(candidate.adjacency.shape[0]) - 1)
    local_bits = _varint_bits(local_node_max)
    rule_bits_total = (
        internal_records * (2 * local_bits + relation_bits)
        + int(candidate.adjacency.shape[0]) * 8
    )
    rule_bits = float(rule_bits_total / support)

    # Occurrence stream: symbol code, variant ID, node count and a delta-coded
    # permutation of fine nodes. Occurrence/variant identifiers are priced as
    # one varint each at this stage.
    occurrence_bits = float(type_code_bits + 8 + _varint_bits(len(nodes)))
    if nodes:
        occurrence_bits += _varint_bits(nodes[0])
        previous = nodes[0]
        for node in nodes[1:]:
            occurrence_bits += _signed_delta_bits(node - previous)
            previous = node

    # internal.bin fields:
    # occurrence_id, shape_edge_index, edge_id, weight_bits.
    internal_payload_bits = float(
        internal_records
        * (
            8
            + _varint_bits(max(0, internal_records - 1))
            + edge_id_bits
            + 64
        )
    )

    # ports.bin fields:
    # occurrence_id, local_node, external_endpoint, relation_id, direction,
    # edge_id and exact weight bits.
    port_payload_bits = float(
        boundary_records
        * (
            8
            + local_bits
            + node_bits
            + relation_bits
            + 8
            + edge_id_bits
            + 64
        )
    )

    encoded_bits = (
        rule_bits
        + occurrence_bits
        + internal_payload_bits
        + port_payload_bits
    )
    return GrammarCodeCost(
        raw_bits=float(raw_bits),
        encoded_bits=float(encoded_bits),
        rule_bits=float(rule_bits),
        occurrence_bits=float(occurrence_bits),
        internal_payload_bits=float(internal_payload_bits),
        port_payload_bits=float(port_payload_bits),
    )
