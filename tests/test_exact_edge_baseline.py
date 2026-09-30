"""Tests for the compact exact non-grammar edge baseline."""
from __future__ import annotations

from semmap_haken.exact_edge_baseline import (
    decode_edge_baseline,
    encode_edge_baseline,
    encode_edge_baseline_bytes,
)
from semmap_haken.grammar_binary import float_to_bits
from semmap_haken.graphex_components import EdgeRecord


def records():
    return (
        EdgeRecord(0, 0, 1, "IsA", 0.1),
        EdgeRecord(1, 1, 0, "IsA", 0.2),
        EdgeRecord(2, 1, 2, "RelatedTo", 1.0),
        EdgeRecord(3, 2, 2, "Self", 3.141592653589793),
    )


def test_binary_baseline_roundtrip_and_implicit_ids(tmp_path):
    path = tmp_path / "baseline.zip"
    report = encode_edge_baseline(3, records(), path)
    assert report["roundtrip_exact"]
    assert report["implicit_edge_ids"]
    assert report["archive_bytes"] == path.stat().st_size
    n, restored = decode_edge_baseline(path)
    assert n == 3
    assert restored == records()
    assert [float_to_bits(row.weight) for row in restored] == [
        float_to_bits(row.weight) for row in records()
    ]


def test_binary_baseline_preserves_nonsequential_ids(tmp_path):
    source = (
        EdgeRecord(10, 0, 1, "r", 0.5),
        EdgeRecord(3, 1, 0, "r", 0.75),
    )
    path = tmp_path / "explicit-ids.zip"
    report = encode_edge_baseline(2, source, path)
    assert not report["implicit_edge_ids"]
    assert decode_edge_baseline(path) == (2, source)


def test_in_memory_baseline_is_decodable(tmp_path):
    payload = encode_edge_baseline_bytes(3, records())
    path = tmp_path / "memory.zip"
    path.write_bytes(payload)
    assert decode_edge_baseline(path) == (3, records())
