"""Bridge between recursive Wishart transitions and grammar_exact_v2.

A transition archive captures the exact directed/relation-layer CSR entries of
one source level before contraction. The coarse matrix remains a derived
analysis view; exact reconstruction uses the grammar archive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .grammar_codec import encode_grammar
from .graphex_components import EdgeRecord
from .relation_adjacency import adjacency_relation_diagnostic


def relation_layers_to_edge_records(
    relation_layers: Mapping[str, sparse.spmatrix],
) -> tuple[EdgeRecord, ...]:
    records: list[EdgeRecord] = []
    edge_id = 0
    for relation, matrix in sorted(relation_layers.items()):
        layer = matrix.tocsr(copy=True)
        layer.sum_duplicates()
        layer.sort_indices()
        rows = np.repeat(np.arange(layer.shape[0]), np.diff(layer.indptr))
        for source, target, weight in zip(
            rows, layer.indices, layer.data, strict=True
        ):
            value = float(weight)
            if not np.isfinite(value):
                raise ValueError("relation layer contains non-finite weight")
            records.append(
                EdgeRecord(
                    edge_id=edge_id,
                    source=int(source),
                    target=int(target),
                    relation=str(relation),
                    weight=value,
                )
            )
            edge_id += 1
    return tuple(records)


def encode_transition_grammar(
    *,
    vertex_count: int,
    relation_layers: Mapping[str, sparse.spmatrix],
    figure_nodes: Sequence[Sequence[int]],
    symbol_types: Mapping[int, str],
    output: str | Path,
    source_adjacency: sparse.spmatrix | None = None,
    grammar_relations: Sequence[str] | None = None,
) -> dict[str, object]:
    records = relation_layers_to_edge_records(relation_layers)
    report = encode_grammar(
        vertex_count,
        records,
        figure_nodes,
        output,
        node_types=symbol_types,
        grammar_relations=grammar_relations,
    )
    result = {
        **report,
        "source_relation_layers": len(relation_layers),
        "scope": "exact source-level relation-layer CSR entries before contraction",
    }
    if source_adjacency is not None:
        result["adjacency_from_relations"] = adjacency_relation_diagnostic(
            source_adjacency,
            relation_layers,
        )
    return result
