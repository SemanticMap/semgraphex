"""Aligned empirical distances across grammar-induced block models.

The symbol IDs are persistent, so adjacent projections can be compared without
solving a graphon relabeling problem. These diagnostics are not cut distance
and do not establish convergence to a graphon/graphex limit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def _mass_map(model: Mapping[str, object]) -> dict[str, float]:
    raw = model.get("mass", {})
    return {
        str(symbol): float(values["probability_mass"])
        for symbol, values in raw.items()
    }


def _block_map(
    model: Mapping[str, object],
    field: str,
) -> dict[tuple[str, str, str], float]:
    return {
        (
            str(row["relation"]),
            str(row["source_symbol"]),
            str(row["target_symbol"]),
        ): float(row[field])
        for row in model.get("blocks", [])
    }


def aligned_projection_distance(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, float]:
    """Compare two finite models in their persistent symbol coordinates."""
    left_mass = _mass_map(left)
    right_mass = _mass_map(right)
    symbols = set(left_mass) | set(right_mass)
    mass_tv = 0.5 * sum(
        abs(left_mass.get(symbol, 0.0) - right_mass.get(symbol, 0.0))
        for symbol in symbols
    )

    left_record = _block_map(left, "record_intensity")
    right_record = _block_map(right, "record_intensity")
    left_weight = _block_map(left, "weight_intensity")
    right_weight = _block_map(right, "weight_intensity")
    block_keys = (
        set(left_record) | set(right_record) | set(left_weight) | set(right_weight)
    )

    record_l1 = 0.0
    weight_l1 = 0.0
    unweighted_record_l1 = 0.0
    unweighted_weight_l1 = 0.0
    relations: set[str] = set()
    for relation, source_symbol, target_symbol in block_keys:
        relations.add(relation)
        source_mass = 0.5 * (
            left_mass.get(source_symbol, 0.0)
            + right_mass.get(source_symbol, 0.0)
        )
        target_mass = 0.5 * (
            left_mass.get(target_symbol, 0.0)
            + right_mass.get(target_symbol, 0.0)
        )
        exposure_weight = source_mass * target_mass
        key = (relation, source_symbol, target_symbol)
        record_delta = abs(
            left_record.get(key, 0.0) - right_record.get(key, 0.0)
        )
        weight_delta = abs(
            left_weight.get(key, 0.0) - right_weight.get(key, 0.0)
        )
        record_l1 += exposure_weight * record_delta
        weight_l1 += exposure_weight * weight_delta
        unweighted_record_l1 += record_delta
        unweighted_weight_l1 += weight_delta

    relation_count = max(1, len(relations))
    return {
        "mass_total_variation": float(mass_tv),
        "record_intensity_weighted_l1": float(record_l1 / relation_count),
        "weight_intensity_weighted_l1": float(weight_l1 / relation_count),
        "record_intensity_unweighted_l1": float(unweighted_record_l1),
        "weight_intensity_unweighted_l1": float(unweighted_weight_l1),
    }


def write_multiscale_graph_report(
    run_dir: str | Path,
) -> dict[str, object]:
    run = Path(run_dir)
    projections: list[tuple[str, dict[str, object]]] = []
    for transition in sorted(run.glob("transition_???_???")):
        path = transition / "dictionary_graphex.json"
        if path.is_file():
            projections.append(
                (
                    transition.name,
                    json.loads(path.read_text(encoding="utf-8")),
                )
            )

    distances = []
    for (left_name, left), (right_name, right) in zip(
        projections, projections[1:], strict=False
    ):
        distances.append({
            "from": left_name,
            "to": right_name,
            **aligned_projection_distance(left, right),
        })

    result = {
        "scope": (
            "aligned empirical distances over persistent grammar symbols; "
            "not cut distance and not evidence of graphon/graphex convergence"
        ),
        "projection_count": len(projections),
        "adjacent_distances": distances,
        "mass_basis": "original level-0 node membership",
    }
    (run / "multiscale_graph_model.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
