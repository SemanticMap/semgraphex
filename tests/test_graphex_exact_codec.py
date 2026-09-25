"""Behavioral tests for exclusive classification and actual archive decoding."""
from __future__ import annotations

import json
import zipfile

import pytest

from semmap_haken.graphex_components import (
    EdgeRecord, classify_edges, validate_partition,
)
from semmap_haken.graphex_codec import decode_graph, encode_graph


def example():
    edges = (
        EdgeRecord(0, 0, 1, "RelatedTo", 0.75),
        EdgeRecord(1, 1, 2, "RelatedTo", 1.25),
        EdgeRecord(2, 2, 0, "RelatedTo", 0.25),
        EdgeRecord(3, 3, 4, "RelatedTo", 0.75),
        EdgeRecord(4, 4, 5, "RelatedTo", 1.25),
        EdgeRecord(5, 5, 3, "RelatedTo", 0.25),
        EdgeRecord(6, 0, 6, "AtLocation", 0.5),  # outbound leaf
        EdgeRecord(7, 7, 3, "AtLocation", 0.5),  # inbound leaf
        EdgeRecord(8, 2, 4, "RelatedTo", 2.0),  # figure boundary -> R
        EdgeRecord(9, 8, 9, "IsA", 0.6),        # isolated pair
        EdgeRecord(10, 0, 2, "IsA", 0.3),
        EdgeRecord(11, 0, 2, "IsA", 0.4),        # parallel record
        EdgeRecord(12, 10, 10, "Self", 0.8),    # loop -> R
    )
    return edges, ((0, 1, 2), (3, 4, 5))


def test_partition_excludes_cross_figure_ports_and_preserves_multiplicity():
    edges, figures = example()
    assigned = classify_edges(11, edges, figures)
    validate_partition(edges, assigned)
    parts = {item.edge.edge_id: item.part for item in assigned}
    assert [parts[i] for i in range(13)] == [
        "W", "W", "W", "W", "W", "W", "S", "S", "R", "I",
        "W", "W", "R",
    ]


def test_no_false_dust_for_rare_bridge_or_parallel_pairs():
    edges = (EdgeRecord(0, 0, 1, "r", 1.0),
             EdgeRecord(1, 0, 1, "r", 2.0),
             EdgeRecord(2, 1, 2, "r", 3.0))
    assert {item.part for item in classify_edges(3, edges, ())} == {"R"}


def test_disjoint_figures_and_unique_record_ids_are_required():
    edges, figures = example()
    with pytest.raises(ValueError, match="overlapping"):
        classify_edges(11, edges, ((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="edge_id"):
        classify_edges(11, (edges[0], edges[0]), figures)


def test_archive_roundtrip_and_measured_bytes(tmp_path):
    edges, figures = example()
    target = tmp_path / "graphex.zip"
    result = encode_graph(11, edges, figures, target)
    assert result["roundtrip_exact"]
    assert result["archive_bytes"] == target.stat().st_size
    assert result["shapes"] >= 1
    assert result["partition"]["R"] == 2
    assert decode_graph(target) == (11, edges)
    with zipfile.ZipFile(target) as archive:
        meta = json.loads(archive.read("manifest.json"))
        assert meta["format"] == "graphex_exact_v1"
        assert meta["edge_count"] == len(edges)
        assert len(archive.read("symbols.bin")) * 8 >= meta["symbol_bits"]


def test_empty_graph_and_unsupported_version(tmp_path):
    target = tmp_path / "empty.zip"
    encode_graph(2, (), (), target)
    assert decode_graph(target) == (2, ())
    with zipfile.ZipFile(target, "a") as archive:
        archive.writestr("manifest.json", '{"format":"old"}')
    with pytest.raises(ValueError, match="unsupported"):
        decode_graph(target)


def test_nonfinite_weight_is_not_silently_serialized(tmp_path):
    with pytest.raises(ValueError):
        encode_graph(2, (EdgeRecord(0, 0, 1, "r", float("nan")),),
                     (), tmp_path / "bad.zip")
