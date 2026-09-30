"""Consolidated exact decoder for a recursive Wishart graph hierarchy.

The archive stores the final coarse graph once plus, for every contraction,
only grammar data for edges touching contracted figures. Edges between
unchanged singleton nodes are inherited exactly from the next coarser level.

This format currently requires hierarchy aggregation="sum".
"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from .grammar_binary import bits_to_float
from .grammar_codec import (
    _read_internal,
    _read_occurrences,
    _read_ports,
    decode_grammar,
    encode_grammar,
)
from .grammar_types import InternalShape, PortSpec, ShapeEdge
from .graphex_components import EdgeRecord
from .transition_grammar import relation_layers_to_edge_records

FORMAT = "semmap_hierarchy_exact_v1"


def _load_relation_layers(level: Path) -> dict[str, sparse.csr_matrix]:
    index = json.loads(
        (level / "relations" / "index.json").read_text(encoding="utf-8")
    )
    return {
        str(relation): sparse.load_npz(level / "relations" / filename).tocsr()
        for relation, filename in sorted(index.items())
    }


def _normalized_records(
    rows: list[tuple[int, int, str, float]],
) -> tuple[EdgeRecord, ...]:
    ordered = sorted(
        rows,
        key=lambda row: (str(row[2]), int(row[0]), int(row[1])),
    )
    keys = [(row[0], row[1], row[2]) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "decoded hierarchy produced duplicate relation/source/target entries"
        )
    return tuple(
        EdgeRecord(
            edge_id=index,
            source=int(source),
            target=int(target),
            relation=str(relation),
            weight=float(weight),
        )
        for index, (source, target, relation, weight) in enumerate(ordered)
    )


def _partition_from_occurrence_nodes(
    vertex_count: int,
    occurrence_nodes: list[tuple[int, ...]],
) -> tuple[np.ndarray, set[int], dict[int, int]]:
    figures = [tuple(sorted(nodes)) for nodes in occurrence_nodes]
    used = {node for group in figures for node in group}
    if len(used) != sum(len(group) for group in figures):
        raise ValueError("transition occurrence groups overlap")
    groups = figures + [
        (node,) for node in range(vertex_count) if node not in used
    ]
    groups = sorted(groups, key=lambda group: (min(group), len(group), group))
    assignment = np.empty(vertex_count, dtype=np.int64)
    figure_set = set(figures)
    figure_coarse: set[int] = set()
    singleton_coarse_to_fine: dict[int, int] = {}
    for coarse, group in enumerate(groups):
        assignment[list(group)] = coarse
        if group in figure_set:
            figure_coarse.add(coarse)
        else:
            if len(group) != 1:
                raise AssertionError("non-figure group must be singleton")
            singleton_coarse_to_fine[coarse] = group[0]
    return assignment, figure_coarse, singleton_coarse_to_fine


def _decode_transition_delta(
    *,
    target_records: tuple[EdgeRecord, ...],
    manifest: dict[str, object],
    shapes_raw: list[dict],
    variants_raw: list[dict],
    occurrences_data: bytes,
    internal_data: bytes,
    ports_data: bytes,
) -> tuple[int, tuple[EdgeRecord, ...]]:
    source_n = int(manifest["vertex_count"])
    relations = tuple(str(value) for value in manifest["relations"])

    shapes = {
        int(item["shape_id"]): InternalShape(
            shape_id=int(item["shape_id"]),
            node_types=tuple(str(value) for value in item["node_types"]),
            edges=tuple(
                ShapeEdge(int(edge[0]), int(edge[1]), int(edge[2]))
                for edge in item["edges"]
            ),
        )
        for item in shapes_raw
    }
    variants = {
        int(item["variant_id"]): {
            PortSpec(int(port[0]), int(port[1]), str(port[2]))
            for port in item["ports"]
        }
        for item in variants_raw
    }
    occurrence_rows = _read_occurrences(occurrences_data)
    occurrences = {
        row.occurrence_id: row for row in occurrence_rows
    }
    _, figure_coarse, singleton_reverse = _partition_from_occurrence_nodes(
        source_n,
        [tuple(row.shape_to_fine_nodes) for row in occurrence_rows],
    )

    rows: list[tuple[int, int, str, float]] = []

    # Edges whose coarse endpoints are both untouched singletons have a
    # one-to-one preimage under sum aggregation and need no transition payload.
    for edge in target_records:
        if edge.source in figure_coarse or edge.target in figure_coarse:
            continue
        try:
            source = singleton_reverse[int(edge.source)]
            target = singleton_reverse[int(edge.target)]
        except KeyError as error:
            raise ValueError(
                "target edge references a coarse node absent from transition partition"
            ) from error
        rows.append((source, target, edge.relation, edge.weight))

    for item in _read_internal(internal_data):
        occurrence = occurrences.get(item.occurrence_id)
        if occurrence is None:
            raise ValueError("internal delta references missing occurrence")
        shape = shapes.get(occurrence.symbol_id)
        if shape is None:
            raise ValueError("internal delta references missing shape")
        if not 0 <= item.shape_edge_index < len(shape.edges):
            raise ValueError("internal delta references missing shape edge")
        shape_edge = shape.edges[item.shape_edge_index]
        source = occurrence.shape_to_fine_nodes[shape_edge.source]
        target = occurrence.shape_to_fine_nodes[shape_edge.target]
        rows.append(
            (
                source,
                target,
                relations[shape_edge.relation_id],
                bits_to_float(item.weight_bits),
            )
        )

    for item in _read_ports(ports_data):
        occurrence = occurrences.get(item.occurrence_id)
        if occurrence is None:
            raise ValueError("port delta references missing occurrence")
        if not 0 <= item.local_port < len(occurrence.shape_to_fine_nodes):
            raise ValueError("port delta references missing local node")
        expected = PortSpec(
            item.local_port,
            item.relation_id,
            item.direction,
        )
        if expected not in variants.get(occurrence.variant_id, set()):
            raise ValueError("port delta is inconsistent with interface variant")
        local = occurrence.shape_to_fine_nodes[item.local_port]
        if item.direction == "out":
            source, target = local, item.external_endpoint
        else:
            source, target = item.external_endpoint, local
        rows.append(
            (
                source,
                target,
                relations[item.relation_id],
                bits_to_float(item.weight_bits),
            )
        )

    return source_n, _normalized_records(rows)


def build_hierarchy_archive(
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, object]:
    """Build and verify final-graph + reverse-transition exact archive."""
    run = Path(run_dir)
    hierarchy_path = run / "hierarchy.json"
    if not hierarchy_path.is_file():
        raise ValueError("hierarchy archive requires hierarchy.json")
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    if hierarchy.get("options", {}).get("aggregation") != "sum":
        raise ValueError(
            "hierarchy_exact_v1 currently requires wishart.aggregation=sum"
        )

    levels = sorted(run.glob("level_???"))
    if not levels:
        raise ValueError("run contains no hierarchy levels")
    final_level = levels[-1]
    final_layers = _load_relation_layers(final_level)
    final_records = relation_layers_to_edge_records(final_layers)
    final_n = sparse.load_npz(final_level / "adjacency.npz").shape[0]

    transitions = []
    for transition in sorted(run.glob("transition_???_???")):
        exact = transition / "grammar_exact_v2.zip"
        if not exact.is_file():
            continue
        parts = transition.name.split("_")
        transitions.append(
            {
                "source_level": int(parts[1]),
                "target_level": int(parts[2]),
                "path": transition,
                "exact": exact,
            }
        )

    expected_transition_count = max(0, len(levels) - 1)
    if len(transitions) != expected_transition_count:
        raise ValueError(
            "every contraction must have grammar_exact_v2.zip before "
            "building a consolidated hierarchy archive"
        )

    target = Path(output) if output is not None else run / "hierarchy_exact_v1.zip"
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="semmap-hierarchy-codec-") as tmp:
        final_path = Path(tmp) / "final_graph.zip"
        encode_grammar(final_n, final_records, (), final_path)

        manifest = {
            "format": FORMAT,
            "aggregation": "sum",
            "final_level": int(final_level.name.split("_")[1]),
            "transition_count": len(transitions),
            "transitions": [
                {
                    "source_level": row["source_level"],
                    "target_level": row["target_level"],
                }
                for row in transitions
            ],
            "scope": (
                "exact relation-layer CSR entries across the recursive "
                "hierarchy; final graph stored once, transition residual "
                "edges between untouched singleton nodes are inherited"
            ),
        }

        with zipfile.ZipFile(
            target,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as outer:
            outer.writestr(
                "manifest.json",
                json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
            )
            outer.writestr(
                "final_graph.zip",
                final_path.read_bytes(),
                compress_type=zipfile.ZIP_STORED,
            )
            for row in transitions:
                prefix = (
                    f"transition_{row['source_level']:03d}_"
                    f"{row['target_level']:03d}"
                )
                with zipfile.ZipFile(row["exact"]) as source:
                    for name in (
                        "manifest.json",
                        "shapes.json",
                        "variants.json",
                        "rules.json",
                        "occurrences.bin",
                        "internal.bin",
                        "ports.bin",
                    ):
                        outer.writestr(
                            f"{prefix}/{name}",
                            source.read(name),
                        )

        decoded_n, decoded = decode_hierarchy(target)
        level0_layers = _load_relation_layers(levels[0])
        expected = relation_layers_to_edge_records(level0_layers)
        if decoded_n != sparse.load_npz(levels[0] / "adjacency.npz").shape[0]:
            target.unlink(missing_ok=True)
            raise AssertionError("hierarchy codec vertex roundtrip failed")
        if decoded != expected:
            target.unlink(missing_ok=True)
            raise AssertionError("hierarchy codec relation roundtrip failed")

        baseline_path = Path(tmp) / "level0_baseline.zip"
        encode_grammar(decoded_n, expected, (), baseline_path)
        baseline_bytes = baseline_path.stat().st_size

    report = {
        "format": FORMAT,
        "archive_bytes": target.stat().st_size,
        "baseline_level0_bytes": baseline_bytes,
        "net_saved_bytes": baseline_bytes - target.stat().st_size,
        "compression_ratio": (
            target.stat().st_size / baseline_bytes if baseline_bytes else None
        ),
        "transitions": len(transitions),
        "final_nodes": int(final_n),
        "level0_nodes": int(decoded_n),
        "roundtrip_exact": True,
    }
    (run / "hierarchy_codec_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def decode_hierarchy(
    archive_path: str | Path,
) -> tuple[int, tuple[EdgeRecord, ...]]:
    """Decode consolidated hierarchy from final level back to level 0."""
    with zipfile.ZipFile(archive_path) as outer:
        manifest = json.loads(outer.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported hierarchy archive format")
        if manifest.get("aggregation") != "sum":
            raise ValueError("unsupported hierarchy aggregation")

        final_bytes = outer.read("final_graph.zip")
        current_n, current_records = decode_grammar(io.BytesIO(final_bytes))

        transitions = sorted(
            manifest["transitions"],
            key=lambda row: int(row["source_level"]),
            reverse=True,
        )
        expected_target_level = int(manifest["final_level"])
        for transition in transitions:
            source_level = int(transition["source_level"])
            target_level = int(transition["target_level"])
            if target_level != expected_target_level:
                raise ValueError("hierarchy transitions are not contiguous")
            prefix = f"transition_{source_level:03d}_{target_level:03d}"
            transition_manifest = json.loads(
                outer.read(f"{prefix}/manifest.json")
            )
            source_n, current_records = _decode_transition_delta(
                target_records=current_records,
                manifest=transition_manifest,
                shapes_raw=json.loads(outer.read(f"{prefix}/shapes.json")),
                variants_raw=json.loads(outer.read(f"{prefix}/variants.json")),
                occurrences_data=outer.read(f"{prefix}/occurrences.bin"),
                internal_data=outer.read(f"{prefix}/internal.bin"),
                ports_data=outer.read(f"{prefix}/ports.bin"),
            )
            current_n = source_n
            expected_target_level = source_level

        if expected_target_level != 0 and transitions:
            raise ValueError("hierarchy archive does not reach level 0")
        return current_n, current_records
