"""Sparse M4 positive/negative controls for Haken hierarchy falsification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .multiscale_config import SyntheticOptions


@dataclass(frozen=True)
class SyntheticGraph:
    name: str
    adjacency: sparse.csr_matrix
    node_ids: tuple[str, ...]
    ground_truth: dict[str, object]


def _symmetric_csr(node_count: int, edges: dict[tuple[int, int], float]) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (left, right), weight in sorted(edges.items()):
        if left == right:
            continue
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((float(weight), float(weight)))
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(node_count, node_count), dtype=np.float64)
    matrix.sum_duplicates()
    return matrix


def hierarchical_positive(options: SyntheticOptions) -> SyntheticGraph:
    """Blocks with strong local rings/chords and weak inter-block bridges.

    The intended condition is explicit multi-timescale structure, not a promise
    that the detector must accept the graph under every threshold choice.
    """
    blocks = options.macro_blocks
    block_size = options.nodes_per_block
    count = blocks * block_size
    edges: dict[tuple[int, int], float] = {}
    block_membership: dict[str, list[int]] = {}
    for block in range(blocks):
        start = block * block_size
        members = list(range(start, start + block_size))
        block_membership[str(block)] = members
        for offset in range(block_size):
            left = start + offset
            for step in (1, 2):
                right = start + ((offset + step) % block_size)
                key = tuple(sorted((left, right)))
                edges[key] = options.internal_weight
    for block in range(blocks):
        left = block * block_size
        right = ((block + 1) % blocks) * block_size
        key = tuple(sorted((left, right)))
        edges[key] = options.bridge_weight
    adjacency = _symmetric_csr(count, edges)
    return SyntheticGraph(
        name="hierarchical_positive",
        adjacency=adjacency,
        node_ids=tuple(f"syn:positive:{index}" for index in range(count)),
        ground_truth={
            "condition": "planted_hierarchical_timescale_separation",
            "macro_blocks": blocks,
            "nodes_per_block": block_size,
            "block_membership": block_membership,
            "internal_weight": options.internal_weight,
            "bridge_weight": options.bridge_weight,
            "seed": options.seed,
        },
    )


def homogeneous_negative(options: SyntheticOptions) -> SyntheticGraph:
    """Matched-size homogeneous circulant control with no planted block hierarchy."""
    count = options.macro_blocks * options.nodes_per_block
    edges: dict[tuple[int, int], float] = {}
    for node in range(count):
        for step in (1, 2):
            other = (node + step) % count
            edges[tuple(sorted((node, other)))] = options.internal_weight
    # Match the positive control's additional bridge-edge count without creating
    # block boundaries: add evenly distributed long chords of the same strong scale.
    generator = np.random.default_rng(options.seed)
    candidates = list(range(count))
    generator.shuffle(candidates)
    added = 0
    for left in candidates:
        if added >= options.macro_blocks:
            break
        right = (left + max(3, count // 3)) % count
        key = tuple(sorted((left, right)))
        if key not in edges:
            edges[key] = options.internal_weight
            added += 1
    adjacency = _symmetric_csr(count, edges)
    return SyntheticGraph(
        name="homogeneous_negative",
        adjacency=adjacency,
        node_ids=tuple(f"syn:negative:{index}" for index in range(count)),
        ground_truth={
            "condition": "no_planted_block_hierarchy",
            "node_count": count,
            "local_offsets": [1, 2],
            "extra_chords": added,
            "uniform_weight": options.internal_weight,
            "seed": options.seed,
        },
    )


def build_synthetic_controls(options: SyntheticOptions) -> tuple[SyntheticGraph, SyntheticGraph]:
    return hierarchical_positive(options), homogeneous_negative(options)


def save_synthetic_graph(graph: SyntheticGraph, directory: str | Path) -> dict[str, Path]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    adjacency_path = destination / "adjacency.npz"
    metadata_path = destination / "ground_truth.json"
    sparse.save_npz(adjacency_path, graph.adjacency)
    metadata_path.write_text(json.dumps({"name": graph.name, **graph.ground_truth}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"adjacency": adjacency_path, "ground_truth": metadata_path}
