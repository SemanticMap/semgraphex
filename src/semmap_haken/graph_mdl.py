"""Structural MDL proxy and canonical Huffman codes for graph symbols."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .wishart_metrics import EgoCandidate


@dataclass(frozen=True)
class MdlOccurrence:
    dictionary_type_id: str
    candidate_index: int
    nodes: tuple[int, ...]
    raw_bits: float
    encoded_bits: float

    @property
    def mdl_gain(self) -> float:
        return float(self.raw_bits - self.encoded_bits)


def _huffman_lengths(counts: Mapping[str, int]) -> dict[str, int]:
    positive = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    if not positive:
        return {}
    if len(positive) == 1:
        return {next(iter(positive)): 1}

    order = 0
    heap: list[tuple[int, int, object]] = []
    for symbol in sorted(positive):
        heap.append((positive[symbol], order, symbol))
        order += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        weight_a, _, left = heapq.heappop(heap)
        weight_b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (weight_a + weight_b, order, (left, right)))
        order += 1

    lengths: dict[str, int] = {}

    def walk(node: object, depth: int) -> None:
        if isinstance(node, str):
            lengths[node] = max(1, depth)
            return
        left, right = node
        walk(left, depth + 1)
        walk(right, depth + 1)

    walk(heap[0][2], 0)
    return lengths


def build_canonical_huffman_codes(counts: Mapping[str, int]) -> dict[str, str]:
    """Return deterministic canonical Huffman codes."""
    lengths = _huffman_lengths(counts)
    if not lengths:
        return {}
    code = 0
    previous_length = 0
    result: dict[str, str] = {}
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous_length
        result[symbol] = format(code, f"0{length}b")
        code += 1
        previous_length = length
    return result


def estimated_occurrence_cost(
    candidate: EgoCandidate,
    *,
    graph_node_count: int,
    relation_count: int,
    type_code_bits: float,
    dictionary_amortized_bits: float = 0.0,
) -> tuple[float, float]:
    """Estimate raw vs dictionary-coded bits for one exact motif occurrence.

    This is a transparent structural MDL proxy, not a serialized file size.
    Boundary edges are excluded from both sides because they remain represented
    by the quotient/residual graph.
    """
    global_node_bits = max(1.0, math.log2(max(2, int(graph_node_count))))
    relation_bits = max(1.0, math.log2(max(2, int(relation_count) + 1)))
    node_count = int(candidate.adjacency.shape[0])
    edge_count = int(candidate.adjacency.nnz)
    raw_bits = edge_count * (2.0 * global_node_bits + relation_bits)
    encoded_bits = (
        node_count * global_node_bits
        + float(type_code_bits)
        + max(0.0, float(dictionary_amortized_bits))
    )
    return float(raw_bits), float(encoded_bits)


def estimate_dictionary_prototype_bits(
    candidate: EgoCandidate,
    *,
    relation_count: int,
) -> float:
    """Estimate one-time bits required to store a local typed prototype."""
    node_count = max(1, int(candidate.adjacency.shape[0]))
    local_node_bits = max(1.0, math.log2(max(2, node_count)))
    relation_bits = max(1.0, math.log2(max(2, int(relation_count) + 1)))
    edge_count = int(candidate.adjacency.nnz)
    return float(edge_count * (2.0 * local_node_bits + relation_bits) + node_count)


def select_nonoverlapping_mdl(
    occurrences: Sequence[MdlOccurrence],
    *,
    local_improvement: bool = True,
) -> tuple[MdlOccurrence, ...]:
    """Greedy weighted set packing followed by deterministic one-for-many swaps."""
    positive = [item for item in occurrences if item.mdl_gain > 0]

    def density(item: MdlOccurrence) -> tuple[float, float, str, int]:
        removed = max(1, len(item.nodes) - 1)
        return (
            item.mdl_gain / removed,
            item.mdl_gain,
            item.dictionary_type_id,
            -item.candidate_index,
        )

    ordered = sorted(positive, key=density, reverse=True)
    selected: list[MdlOccurrence] = []
    occupied: set[int] = set()
    rejected: list[MdlOccurrence] = []
    for item in ordered:
        nodes = set(item.nodes)
        if nodes & occupied:
            rejected.append(item)
            continue
        selected.append(item)
        occupied.update(nodes)

    if local_improvement:
        for challenger in sorted(rejected, key=lambda item: item.mdl_gain, reverse=True):
            challenger_nodes = set(challenger.nodes)
            conflicts = [item for item in selected if set(item.nodes) & challenger_nodes]
            if not conflicts:
                selected.append(challenger)
                continue
            if challenger.mdl_gain <= sum(item.mdl_gain for item in conflicts) + 1e-12:
                continue
            remaining = [item for item in selected if item not in conflicts]
            remaining_nodes = {node for item in remaining for node in item.nodes}
            if challenger_nodes & remaining_nodes:
                continue
            selected = remaining + [challenger]

    return tuple(
        sorted(
            selected,
            key=lambda item: (min(item.nodes), len(item.nodes), item.dictionary_type_id),
        )
    )
