"""Core data model for the recursive lossless graph grammar codec.

The codec distinguishes reusable internal topology from occurrence-specific
interface bindings. All relation labels are represented by stable integer IDs
inside the binary codec; human-readable names live in the archive relation
dictionary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PortDirection = Literal["in", "out"]


@dataclass(frozen=True, order=True)
class ShapeEdge:
    source: int
    target: int
    relation_id: int


@dataclass(frozen=True)
class InternalShape:
    shape_id: int
    node_types: tuple[str, ...]
    edges: tuple[ShapeEdge, ...]


@dataclass(frozen=True, order=True)
class PortSpec:
    local_node: int
    relation_id: int
    direction: PortDirection


@dataclass(frozen=True)
class InterfaceVariant:
    variant_id: int
    shape_id: int
    ports: tuple[PortSpec, ...]


@dataclass(frozen=True)
class GrammarRule:
    symbol_id: int
    shape_id: int
    child_symbols: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class Occurrence:
    occurrence_id: int
    symbol_id: int
    variant_id: int
    shape_to_fine_nodes: tuple[int, ...]


@dataclass(frozen=True)
class InternalEdgePayload:
    occurrence_id: int
    shape_edge_index: int
    edge_id: int
    weight_bits: int


@dataclass(frozen=True)
class PortBinding:
    occurrence_id: int
    local_port: int
    external_endpoint: int
    relation_id: int
    direction: PortDirection
    edge_id: int
    weight_bits: int


@dataclass(frozen=True)
class ResidualEdge:
    edge_id: int
    source: int
    target: int
    relation_id: int
    weight_bits: int
