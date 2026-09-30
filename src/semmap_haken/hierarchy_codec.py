"""Consolidated exact decoder for a recursive Wishart graph hierarchy.

The archive stores the final coarse graph once plus, for every contraction,
only grammar data for edges touching contracted figures. Edges between
unchanged singleton nodes are inherited exactly from the next coarser level.

This format currently requires hierarchy aggregation="sum".
"""
from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from .exact_edge_baseline import encode_edge_baseline_bytes
from .grammar_binary import (
    bits_to_float,
    read_uvarint,
    write_uvarint,
)
from .grammar_codec import (
    _read_internal,
    _read_occurrences,
    _read_ports,
    decode_grammar,
    encode_grammar,
)
from .grammar_streams import (
    STREAM_LAYOUT as COMPACT_STREAM_LAYOUT,
    read_internal as _read_internal_compact,
    read_occurrences as _read_occurrences_compact,
    read_ports as _read_ports_compact,
)
from .grammar_types import InterfaceVariant, InternalShape, PortSpec, ShapeEdge
from .graphex_components import EdgeRecord
from .transition_grammar import relation_layers_to_edge_records
from .relation_adjacency import adjacency_relation_diagnostic, sum_relation_layers

FORMAT = "semmap_hierarchy_exact_v1"


def _common_prefix_bytes(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _encode_level0_memberships(level: Path) -> bytes:
    raw = json.loads((level / "membership.json").read_text(encoding="utf-8"))
    count = len(raw)
    if set(map(int, raw)) != set(range(count)):
        raise ValueError("level-0 membership IDs must be contiguous")
    stream = io.BytesIO()
    write_uvarint(stream, count)
    previous = b""
    for node in range(count):
        value = raw[str(node)]
        concepts = value.get("original_concepts", []) if isinstance(value, dict) else value
        if isinstance(concepts, str):
            concepts = [concepts]
        concepts = [str(item) for item in concepts]
        if not concepts:
            raise ValueError(f"missing original concepts for level-0 node {node}")
        write_uvarint(stream, len(concepts))
        for concept in concepts:
            encoded = concept.encode("utf-8")
            prefix = _common_prefix_bytes(previous, encoded)
            suffix = encoded[prefix:]
            write_uvarint(stream, prefix)
            write_uvarint(stream, len(suffix))
            stream.write(suffix)
            previous = encoded
    return stream.getvalue()


def _decode_memberships(data: bytes) -> tuple[tuple[str, ...], ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    rows: list[tuple[str, ...]] = []
    previous = b""
    for _ in range(count):
        item_count = read_uvarint(stream)
        if item_count <= 0:
            raise ValueError("membership row must not be empty")
        concepts = []
        for _ in range(item_count):
            prefix = read_uvarint(stream)
            suffix_size = read_uvarint(stream)
            if prefix > len(previous):
                raise ValueError("invalid membership prefix length")
            suffix = stream.read(suffix_size)
            if len(suffix) != suffix_size:
                raise ValueError("truncated membership suffix")
            encoded = previous[:prefix] + suffix
            concepts.append(encoded.decode("utf-8"))
            previous = encoded
        rows.append(tuple(concepts))
    if stream.read(1):
        raise ValueError("trailing membership bytes")
    return tuple(rows)


def _relation_filename(relation: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", relation).strip("_") or "relation"
    return safe + ".npz"


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
    variant_objects = {
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
    variants = {
        variant_id: set(variant.ports)
        for variant_id, variant in variant_objects.items()
    }
    stream_layout = manifest.get("stream_layout")
    if stream_layout == COMPACT_STREAM_LAYOUT:
        occurrence_rows = _read_occurrences_compact(
            occurrences_data,
            shapes,
        )
        internal_rows = _read_internal_compact(
            internal_data,
            occurrence_rows,
            shapes,
        )
        port_rows = _read_ports_compact(
            ports_data,
            occurrence_rows,
            variant_objects,
        )
    elif stream_layout is None:
        occurrence_rows = _read_occurrences(occurrences_data)
        internal_rows = _read_internal(internal_data)
        port_rows = _read_ports(ports_data)
    else:
        raise ValueError(f"unsupported grammar stream layout: {stream_layout}")
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

    for item in internal_rows:
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

    for item in port_rows:
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
    level0 = levels[0]
    level0_layers = _load_relation_layers(level0)
    level0_adjacency = sparse.load_npz(level0 / "adjacency.npz").tocsr()
    adjacency_diagnostic = adjacency_relation_diagnostic(
        level0_adjacency,
        level0_layers,
    )
    membership_bytes = _encode_level0_memberships(level0)

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
            "membership_encoding": "front-coded UTF-8 original_concepts",
            "adjacency_mode": (
                "relation_sum"
                if adjacency_diagnostic["matches"]
                else "stored_level0_npz"
            ),
            "level0_adjacency_from_relations": adjacency_diagnostic,
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
            outer.writestr("membership.bin", membership_bytes)
            if not adjacency_diagnostic["matches"]:
                outer.writestr(
                    "level0_adjacency.npz",
                    (level0 / "adjacency.npz").read_bytes(),
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
        expected = relation_layers_to_edge_records(level0_layers)
        if decoded_n != level0_adjacency.shape[0]:
            target.unlink(missing_ok=True)
            raise AssertionError("hierarchy codec vertex roundtrip failed")
        if decoded != expected:
            target.unlink(missing_ok=True)
            raise AssertionError("hierarchy codec relation roundtrip failed")

        # Keep the historical residual-only grammar baseline for continuity,
        # but compare new compression claims against a compact non-grammar
        # binary bundle containing the same level-0 memberships (and the same
        # raw-adjacency fallback, when one is required).
        baseline_path = Path(tmp) / "level0_legacy_grammar_baseline.zip"
        encode_grammar(decoded_n, expected, (), baseline_path)
        legacy_baseline_bytes = baseline_path.stat().st_size

        edge_baseline_bytes = encode_edge_baseline_bytes(decoded_n, expected)
        bundle_buffer = io.BytesIO()
        with zipfile.ZipFile(
            bundle_buffer,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as baseline_bundle:
            baseline_bundle.writestr(
                "edge_baseline.zip",
                edge_baseline_bytes,
                compress_type=zipfile.ZIP_STORED,
            )
            baseline_bundle.writestr("membership.bin", membership_bytes)
            if not adjacency_diagnostic["matches"]:
                baseline_bundle.writestr(
                    "level0_adjacency.npz",
                    (level0 / "adjacency.npz").read_bytes(),
                    compress_type=zipfile.ZIP_STORED,
                )
        binary_bundle_baseline_bytes = len(bundle_buffer.getvalue())

    with zipfile.ZipFile(target) as built_archive:
        entry_compressed_bytes = {
            info.filename: int(info.compress_size)
            for info in built_archive.infolist()
        }
    archive_bytes = target.stat().st_size
    compressed_payload_bytes = sum(entry_compressed_bytes.values())
    container_overhead_bytes = archive_bytes - compressed_payload_bytes

    report = {
        "format": FORMAT,
        "archive_bytes": archive_bytes,
        # Historical comparator retained so existing result readers do not
        # silently change meaning.
        "baseline_level0_bytes": legacy_baseline_bytes,
        "baseline_level0_legacy_grammar_bytes": legacy_baseline_bytes,
        "baseline_level0_binary_edge_bytes": len(edge_baseline_bytes),
        "baseline_level0_binary_bundle_bytes": binary_bundle_baseline_bytes,
        "net_saved_bytes": legacy_baseline_bytes - archive_bytes,
        "net_saved_vs_binary_bundle_bytes": (
            binary_bundle_baseline_bytes - archive_bytes
        ),
        "compression_ratio": (
            archive_bytes / legacy_baseline_bytes
            if legacy_baseline_bytes else None
        ),
        "compression_ratio_binary_bundle": (
            archive_bytes / binary_bundle_baseline_bytes
            if binary_bundle_baseline_bytes else None
        ),
        "entry_compressed_bytes": entry_compressed_bytes,
        "zip_container_overhead_bytes": container_overhead_bytes,
        "transitions": len(transitions),
        "final_nodes": int(final_n),
        "level0_nodes": int(decoded_n),
        "roundtrip_exact": True,
        "membership_nodes": len(_decode_memberships(membership_bytes)),
        "adjacency_mode": manifest["adjacency_mode"],
        "level0_adjacency_from_relations": adjacency_diagnostic,
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


def decode_hierarchy_bundle(
    archive_path: str | Path,
) -> tuple[
    int,
    tuple[EdgeRecord, ...],
    tuple[tuple[str, ...], ...],
    bytes | None,
]:
    """Decode graph records, level-0 memberships, and optional raw adjacency."""
    n, records = decode_hierarchy(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        memberships = _decode_memberships(archive.read("membership.bin"))
        raw_adjacency = (
            archive.read("level0_adjacency.npz")
            if manifest.get("adjacency_mode") == "stored_level0_npz"
            else None
        )
    if len(memberships) != n:
        raise ValueError("membership count differs from decoded node count")
    return n, records, memberships, raw_adjacency


def write_decoded_hierarchy(
    archive_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Materialize a decoded hierarchy in the standard level directory form."""
    n, records, memberships, raw_adjacency = decode_hierarchy_bundle(
        archive_path
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    relation_dir = target / "relations"
    relation_dir.mkdir(exist_ok=True)

    by_relation: dict[str, list[EdgeRecord]] = {}
    for record in records:
        by_relation.setdefault(record.relation, []).append(record)

    relation_index = {}
    layers = {}
    for relation, rows in sorted(by_relation.items()):
        source = np.fromiter(
            (row.source for row in rows), dtype=np.int64, count=len(rows)
        )
        destination = np.fromiter(
            (row.target for row in rows), dtype=np.int64, count=len(rows)
        )
        weights = np.fromiter(
            (row.weight for row in rows), dtype=np.float64, count=len(rows)
        )
        layer = sparse.csr_matrix(
            (weights, (source, destination)),
            shape=(n, n),
            dtype=np.float64,
        )
        layer.sum_duplicates()
        layer.sort_indices()
        filename = _relation_filename(relation)
        sparse.save_npz(relation_dir / filename, layer)
        relation_index[relation] = filename
        layers[relation] = layer

    (relation_dir / "index.json").write_text(
        json.dumps(relation_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if raw_adjacency is not None:
        (target / "adjacency.npz").write_bytes(raw_adjacency)
        adjacency_mode = "stored_level0_npz"
    else:
        adjacency = sum_relation_layers(layers, shape=(n, n))
        sparse.save_npz(target / "adjacency.npz", adjacency)
        adjacency_mode = "relation_sum"

    membership_payload = {
        str(index): {
            "original_concepts": list(values),
            "concept_concat": " | ".join(values),
        }
        for index, values in enumerate(memberships)
    }
    (target / "membership.json").write_text(
        json.dumps(
            membership_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    report = {
        "format": FORMAT,
        "node_count": n,
        "edge_records": len(records),
        "relations": len(layers),
        "adjacency_mode": adjacency_mode,
        "source_archive": str(archive_path),
        "roundtrip_materialized": True,
    }
    (target / "DECODED.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "COMPLETED").write_text("decoded\n", encoding="utf-8")
    return report
