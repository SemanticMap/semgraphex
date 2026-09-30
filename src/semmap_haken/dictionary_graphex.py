"""Finite graphon/graphex-style projection induced by grammar symbols.

This module deliberately estimates only an empirical relation-specific block
kernel and symbol mass measure. It does not claim convergence to an
exchangeable graphon/graphex limit, and it does not reinterpret the codec's
W/S/I/R labels as mathematical graphex components.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import sparse


def estimate_dictionary_projection(
    relation_layers: Mapping[str, sparse.spmatrix],
    fine_to_coarse: np.ndarray,
    dictionary_type_by_coarse: Mapping[int, str],
) -> dict[str, object]:
    assignment = np.asarray(fine_to_coarse, dtype=np.int64)
    if assignment.ndim != 1:
        raise ValueError("fine_to_coarse must be one-dimensional")
    if assignment.size == 0:
        return {
            "scope": "finite empirical grammar-induced sparse block model",
            "mass": {},
            "blocks": [],
            "symbol_count": 0,
            "graphex_components": {
                "W": "empty empirical block kernel",
                "S": "not estimated",
                "I": "not estimated",
            },
        }

    fine_symbol = np.array(
        [
            str(dictionary_type_by_coarse.get(int(coarse), "ATOM"))
            for coarse in assignment
        ],
        dtype=object,
    )
    unique, counts = np.unique(fine_symbol, return_counts=True)
    mass_counts = {
        str(symbol): int(count)
        for symbol, count in zip(unique, counts, strict=True)
    }
    total = int(assignment.size)
    mass = {
        symbol: {
            "original_node_count": count,
            "probability_mass": count / total,
        }
        for symbol, count in sorted(mass_counts.items())
    }

    accum: dict[tuple[str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for relation, matrix in sorted(relation_layers.items()):
        layer = matrix.tocsr()
        if layer.shape != (total, total):
            raise ValueError("relation layer size differs from fine_to_coarse")
        rows = np.repeat(np.arange(total), np.diff(layer.indptr))
        for source, target, weight in zip(
            rows, layer.indices, layer.data, strict=True
        ):
            key = (
                str(relation),
                str(fine_symbol[int(source)]),
                str(fine_symbol[int(target)]),
            )
            accum[key][0] += 1.0
            accum[key][1] += float(weight)

    blocks = []
    for (relation, source_symbol, target_symbol), (
        edge_records,
        weight_sum,
    ) in sorted(accum.items()):
        exposure = (
            mass_counts[source_symbol] * mass_counts[target_symbol]
        )
        blocks.append({
            "relation": relation,
            "source_symbol": source_symbol,
            "target_symbol": target_symbol,
            "edge_records": int(edge_records),
            "weight_sum": float(weight_sum),
            "exposure_pairs": int(exposure),
            "record_intensity": (
                float(edge_records / exposure) if exposure else 0.0
            ),
            "weight_intensity": (
                float(weight_sum / exposure) if exposure else 0.0
            ),
        })

    return {
        "scope": (
            "finite empirical grammar-induced typed directed block intensity; "
            "not a fitted exchangeable limit graphon/graphex"
        ),
        "mass": mass,
        "blocks": blocks,
        "symbol_count": len(mass),
        "block_count": len(blocks),
        "graphex_components": {
            "W": "empirical relation-specific block kernel above",
            "S": "not estimated by this projection",
            "I": "not estimated by this projection",
        },
    }


def write_dictionary_projection(
    path: str | Path,
    relation_layers: Mapping[str, sparse.spmatrix],
    fine_to_coarse: np.ndarray,
    dictionary_type_by_coarse: Mapping[int, str],
) -> dict[str, object]:
    result = estimate_dictionary_projection(
        relation_layers,
        fine_to_coarse,
        dictionary_type_by_coarse,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
