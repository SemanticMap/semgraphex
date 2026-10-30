"""Exact two-level hierarchy with parallel assertions, loops, IDs and corrupt checkpoint."""
from __future__ import annotations

import numpy as np

from semmap_haken.grammar_codec import encode_grammar
from semmap_haken.graphex_components import EdgeRecord
from semmap_haken.hypothesis.recursive_codec import (
    _assert_records, _state, _valid_checkpoint, _write_archive,
    contract_records, decode_recursive,
)
from semmap_haken.hypothesis.scan_checkpoint import atomic_json, sha256_file


def original():
    return (
        EdgeRecord(0, 0, 1, "R", -0.0),
        EdgeRecord(1, 0, 1, "R", 0.5),
        EdgeRecord(2, 2, 3, "R", 2.0),
        EdgeRecord(3, 2, 3, "R", 3.0),
        EdgeRecord(4, 1, 4, "I", 4.0),
        EdgeRecord(5, 3, 5, "I", 5.0),
        EdgeRecord(6, 6, 7, "R", 6.0),
        EdgeRecord(7, 7, 7, "L", 7.5),
        EdgeRecord(8, 5, 6, "R", 9.0),
        EdgeRecord(9, 7, 1, "R", 10.0),
    )


def test_two_level_exact_decode_and_hierarchy_symbols(tmp_path):
    records = original()
    ids = {"node_ids": [f"n{i}" for i in range(8)],
           "assertion_ids": [f"a{i}" for i in range(10)],
           "relation_names": ["R", "I", "L"], "graph_sha256": "synthetic"}
    baseline = tmp_path / "base.zip"
    _write_archive(None, original_ids=ids, n=8, edges=records,
                   original_count=8, original_assertions=10, output=baseline)
    assert decode_recursive(baseline) == (8, records, ids)

    first = tmp_path / "step0.zip"
    figures = [(0, 1), (2, 3)]
    encode_grammar(8, records, figures, first, symbol_coding="huffman")
    coarse_n, coarse_edges, _, types = contract_records(
        8, records, figures, {}, 0, first
    )
    assert coarse_n == 6
    assert len(types) >= 2
    level1 = tmp_path / "level1.zip"
    _write_archive(baseline, original_ids=ids, n=coarse_n,
                   edges=coarse_edges, original_count=8, original_assertions=10,
                   output=level1, transition=first)
    assert decode_recursive(level1) == (8, records, ids)
    n, saved, memberships, node_types, nsteps = _state(level1)
    assert nsteps == 1 and n == 6 and saved == coarse_edges
    assert memberships[0] == (0, 1)
    assert node_types[0].startswith("L0_S")

    second = tmp_path / "step1.zip"
    second_figures = [(0, 2), (1, 3)]
    encode_grammar(n, coarse_edges, second_figures, second,
                   node_types=node_types, symbol_coding="huffman")
    next_n, next_edges, _, next_types = contract_records(
        n, coarse_edges, second_figures, node_types, 1, second
    )
    assert next_n == 4 and any(x.startswith("L1_S") for x in next_types.values())
    final = tmp_path / "level2.zip"
    _write_archive(level1, original_ids=ids, n=next_n,
                   edges=next_edges, original_count=8,
                   original_assertions=10, output=final, transition=second)
    n0, decoded, names = decode_recursive(final)
    assert n0 == 8 and names == ids
    _assert_records(decoded, records)
    assert _state(final)[-1] == 2


def test_hierarchy_checkpoint_sha_and_torn_files(tmp_path):
    records = original()
    ids = {"node_ids": [str(x) for x in range(8)],
           "assertion_ids": [str(x) for x in range(10)],
           "relation_names": ["R", "I", "L"], "graph_sha256": "synthetic"}
    archive = tmp_path / "level_000.zip"
    _write_archive(None, original_ids=ids, n=8, edges=records,
                   original_count=8, original_assertions=10, output=archive)
    identity = {"graph_sha256": "synthetic", "seed": 42}
    meta = archive.with_suffix(".json")
    atomic_json(meta, {"identity": identity,
                       "sha256": sha256_file(archive)})
    assert _valid_checkpoint(archive, meta, identity=identity,
                             expected_n=8, expected_edges=records)
    data = bytearray(archive.read_bytes())
    data[len(data)//3] ^= 1
    archive.write_bytes(data)
    assert not _valid_checkpoint(archive, meta, identity=identity,
                                 expected_n=8, expected_edges=records)
    (tmp_path / "level_001.zip.tmp").write_bytes(b"partial")
    assert not _valid_checkpoint(tmp_path / "level_001.zip",
                                 tmp_path / "level_001.json",
                                 identity=identity, expected_n=8,
                                 expected_edges=records)
