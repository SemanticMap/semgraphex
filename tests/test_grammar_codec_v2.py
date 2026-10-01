"""Behavioral tests for the recursive-ready grammar_exact_v2 codec."""
from __future__ import annotations

import json
import zipfile

import pytest

from semmap_haken.edge_table import EdgeTable
from semmap_haken.grammar_binary import float_to_bits
from semmap_haken.grammar_codec import (
    FORMAT,
    decode_grammar,
    encode_grammar,
    inspect_grammar_archive,
)
from semmap_haken.graphex_components import EdgeRecord


def repeated_graph():
    edges = (
        # Figure A: triangle + parallel relation + self-loop.
        EdgeRecord(10, 0, 1, "RelatedTo", 0.125),
        EdgeRecord(11, 1, 2, "RelatedTo", 0.25),
        EdgeRecord(12, 2, 0, "RelatedTo", 0.5),
        EdgeRecord(13, 0, 2, "IsA", 0.75),
        EdgeRecord(14, 0, 2, "IsA", 0.875),
        EdgeRecord(15, 1, 1, "Self", 1.125),
        # Figure B: isomorphic internal structure, different weights.
        EdgeRecord(20, 3, 4, "RelatedTo", 1.25),
        EdgeRecord(21, 4, 5, "RelatedTo", 1.5),
        EdgeRecord(22, 5, 3, "RelatedTo", 1.75),
        EdgeRecord(23, 3, 5, "IsA", 2.0),
        EdgeRecord(24, 3, 5, "IsA", 2.25),
        EdgeRecord(25, 4, 4, "Self", 2.5),
        # Different interfaces for the same core shape.
        EdgeRecord(30, 1, 6, "AtLocation", 3.0),
        EdgeRecord(31, 7, 4, "UsedFor", 3.25),
        # Figure-to-figure boundary: must be stored exactly once as a port edge.
        EdgeRecord(32, 2, 3, "Bridge", 3.5),
        # Pure residual edge.
        EdgeRecord(40, 8, 9, "Residual", 4.0),
    )
    return edges, ((0, 1, 2), (3, 4, 5))


def test_edge_table_roundtrip_preserves_exact_weight_bits():
    edges, _ = repeated_graph()
    table = EdgeTable.from_records(edges)
    table.validate(vertex_count=10)
    restored = table.to_records()
    assert restored == edges
    assert [
        float_to_bits(row.weight) for row in restored
    ] == [
        float_to_bits(row.weight) for row in edges
    ]
    assert table.nbytes > 0


def test_grammar_v2_roundtrip_reuses_shape_and_separates_interfaces(tmp_path):
    edges, figures = repeated_graph()
    archive = tmp_path / "grammar-v2.zip"
    report = encode_grammar(10, edges, figures, archive)

    assert report["format"] == FORMAT
    assert report["roundtrip_exact"]
    assert report["shapes"] == 1
    assert report["interface_variants"] == 2
    assert report["occurrences"] == 2
    assert report["internal_edge_records"] == 12
    assert report["port_edge_records"] == 3
    assert report["residual_edge_records"] == 1
    assert report["archive_bytes"] == archive.stat().st_size
    assert report["baseline_binary_bytes"] > 0
    assert report["baseline_json_bytes"] == report["baseline_bytes"]
    assert report["compression_ratio_binary_baseline"] == (
        report["archive_bytes"] / report["baseline_binary_bytes"]
    )

    n, restored = decode_grammar(archive)
    assert n == 10
    assert restored == edges
    assert [float_to_bits(e.weight) for e in restored] == [
        float_to_bits(e.weight) for e in edges
    ]

    meta = inspect_grammar_archive(archive)
    assert meta["shape_count"] == 1
    assert meta["variant_count"] == 2
    assert "ports.bin" in meta["entry_compressed_bytes"]

    with zipfile.ZipFile(archive) as z:
        assert "occurrences.bin" in z.namelist()
        assert "occurrences.json" not in z.namelist()


def test_cross_figure_edge_is_not_duplicated(tmp_path):
    edges, figures = repeated_graph()
    archive = tmp_path / "cross.zip"
    report = encode_grammar(10, edges, figures, archive)
    assert report["port_edge_records"] == 3
    _, restored = decode_grammar(archive)
    assert sum(edge.edge_id == 32 for edge in restored) == 1


def test_recursive_node_types_are_part_of_shape_identity(tmp_path):
    edges = (
        EdgeRecord(0, 0, 1, "r", 1.0),
        EdgeRecord(1, 2, 3, "r", 1.0),
    )
    archive = tmp_path / "typed.zip"
    report = encode_grammar(
        4,
        edges,
        ((0, 1), (2, 3)),
        archive,
        node_types={0: "GT_CHILD", 2: "GT_OTHER"},
    )
    assert report["shapes"] == 2
    assert decode_grammar(archive) == (4, edges)


def test_empty_graph_and_no_figures_are_exact(tmp_path):
    empty = tmp_path / "empty.zip"
    report = encode_grammar(3, (), (), empty)
    assert report["roundtrip_exact"]
    assert decode_grammar(empty) == (3, ())

    edge = EdgeRecord(9, 0, 2, "r", 1.25)
    residual = tmp_path / "residual.zip"
    report = encode_grammar(3, (edge,), (), residual)
    assert report["shapes"] == 0
    assert report["residual_edge_records"] == 1
    assert decode_grammar(residual) == (3, (edge,))


def test_invalid_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        encode_grammar(
            3,
            (EdgeRecord(0, 0, 1, "r", 1.0),),
            ((0, 1), (1, 2)),
            tmp_path / "overlap.zip",
        )
    with pytest.raises(ValueError, match="non-finite"):
        encode_grammar(
            2,
            (EdgeRecord(0, 0, 1, "r", float("nan")),),
            (),
            tmp_path / "nan.zip",
        )


def test_interface_variant_ignores_external_edge_multiplicity(tmp_path):
    edges = (
        EdgeRecord(0, 0, 1, "core", 1.0),
        EdgeRecord(1, 2, 3, "core", 2.0),
        EdgeRecord(2, 1, 4, "external", 3.0),
        EdgeRecord(3, 3, 5, "external", 4.0),
        EdgeRecord(4, 3, 6, "external", 5.0),
    )
    archive = tmp_path / "multiplicity.zip"
    report = encode_grammar(
        7,
        edges,
        ((0, 1), (2, 3)),
        archive,
    )
    assert report["shapes"] == 1
    # Both occurrences expose the same typed/directed port schema.  The
    # different number of external records belongs in bindings, not variants.
    assert report["interface_variants"] == 1
    assert report["port_edge_records"] == 3
    assert decode_grammar(archive) == (7, edges)


def test_relation_subset_keeps_full_graph_exact(tmp_path):
    edges = (
        EdgeRecord(0, 0, 1, "RelatedTo", 1.0),
        EdgeRecord(1, 1, 2, "RelatedTo", 2.0),
        EdgeRecord(2, 0, 2, "IsA", 3.0),
        EdgeRecord(3, 2, 3, "UsedFor", 4.0),
    )
    archive = tmp_path / "relatedto-grammar.zip"
    report = encode_grammar(
        4,
        edges,
        ((0, 1, 2),),
        archive,
        grammar_relations=("RelatedTo",),
    )
    assert decode_grammar(archive) == (4, edges)
    assert report["roundtrip_exact"]
    assert report["grammar_relations"] == ["RelatedTo"]
    assert report["grammar_relation_edge_records"] == 2
    assert report["non_grammar_relation_edge_records"] == 2
    # Only RelatedTo becomes reusable internal structure. The IsA edge inside
    # the figure and UsedFor boundary edge stay exact residual corrections.
    assert report["internal_edge_records"] == 2
    assert report["port_edge_records"] == 0
    assert report["residual_edge_records"] == 2
