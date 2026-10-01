"""Unit tests are deliberately tiny; the 100k run needs an L4 Colab runtime."""
import io
import json

import numpy as np
from scipy import sparse

from semmap_haken.glove_graphex_100k import (
    align_glove, baseline_zip, block_model, coarse_graph, connector_profiles,
    decode_exact, decode_labels, discover_pairs, encode_exact, huffman_bits, huffman_symbols,
    read_members, unbinary_edges, binary_edges, varint, read_varint,
)
from semmap_haken.graphex_components import EdgeRecord


def example_graph():
    edges = (
        EdgeRecord(0, 0, 1, "IsA", 1.5),
        EdgeRecord(1, 2, 3, "IsA", 2.0),
        EdgeRecord(2, 0, 4, "RelatedTo", 0.5),
        EdgeRecord(3, 5, 6, "PartOf", 0.25),
        EdgeRecord(4, 1, 0, "HasA", 0.75),
        EdgeRecord(5, 3, 2, "HasA", 0.75),
        EdgeRecord(6, 4, 4, "Self", 1.0),
    )
    members = [[f"/c/en/concept_{i}"] for i in range(7)]
    labels = np.array([0, 0, 0, 0, -1, -1, -1], np.int32)
    rows, cols, values = zip(*[(e.source, e.target, e.weight) for e in edges])
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(7, 7)).tocsr()
    output = io.BytesIO(); sparse.save_npz(output, matrix)
    return edges, members, labels, matrix, output.getvalue()


def test_varint_and_binary_roundtrip():
    edges, *_ = example_graph()
    for value in (0, 127, 128, 2**31, 2**63 - 1):
        assert read_varint(io.BytesIO(varint(value))) == value
    relations = sorted({e.relation for e in edges})
    actual = unbinary_edges(binary_edges(edges, relations), relations, len(edges))
    assert sorted(actual, key=lambda e: e.edge_id) == list(edges)


def test_huffman_and_lossless_archive(tmp_path):
    edges, members, labels, _, adj = example_graph()
    figures = discover_pairs(edges, labels, max_figures=8, min_support=2)
    assert len(figures) == 2
    assert len({item[2] for item in figures}) == 1
    bits, count = huffman_bits(["a", "b", "a"], {"a": "0", "b": "1"})
    assert huffman_symbols(bits, count, 3, {"a": "0", "b": "1"}) == ["a", "b", "a"]
    model = block_model(edges, labels)
    connectors = connector_profiles(edges, labels)
    assert connectors["0"]["out_by_relation"]["RelatedTo"] == 0.5
    baseline = baseline_zip(tmp_path / "baseline.zip", adj, members, edges)
    assert baseline > 0
    for selected in ([], figures):
        path = tmp_path / f"{len(selected)}.zip"
        report = encode_exact(path, adj, members, edges, labels, selected,
                              block=model, connectors=connectors)
        assert report["roundtrip_exact"]
        assert decode_exact(path) == (7, members, edges, adj)
        assert np.array_equal(decode_labels(path), labels)
        assert report["archive_bytes"] == path.stat().st_size


def test_directional_quotient_and_preserved_weight():
    _, _, labels, adjacency, _ = example_graph()
    mapping, quotient = coarse_graph(adjacency, labels)
    assert quotient.shape[0] < adjacency.shape[0]
    assert np.isclose(adjacency.sum(), quotient.sum())
    assert quotient[mapping[0], mapping[4]] != quotient[mapping[4], mapping[0]]


def test_glove_alignment_oov_and_phrase(tmp_path):
    glove = tmp_path / "glove.txt"
    glove.write_text("ice 1 0\ncream 0 1\nhello 1 1\n", encoding="utf8")
    members = [["/c/en/ice_cream"], ["/c/ru/led"], ["/c/en/hello_world"]]
    report = align_glove(members, glove, tmp_path / "embedding", dimension=2)
    assert report["full"] == 1 and report["partial"] == 1 and report["oov"] == 1
    vectors = np.load(tmp_path / "embedding" / "vectors.npy")
    assert np.isclose(np.linalg.norm(vectors[0]), 1)
    assert not np.any(vectors[1])


def test_noncontiguous_membership_rejected(tmp_path):
    (tmp_path / "membership.json").write_text(json.dumps({"0": ["/c/en/x"], "2": ["/c/en/y"]}))
    try:
        read_members(tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid node membership accepted")


def test_wishart_is_ranking_prior_not_hard_filter():
    edges = (
        EdgeRecord(0, 0, 1, "IsA", 1.0),
        EdgeRecord(1, 2, 3, "IsA", 1.0),
    )
    labels = np.array([0, 1, 2, 3], dtype=np.int32)
    ranked = discover_pairs(
        edges, labels, max_figures=8, min_support=2, semantic_prior="ranking"
    )
    filtered = discover_pairs(
        edges, labels, max_figures=8, min_support=2, semantic_prior="filter"
    )
    disabled = discover_pairs(
        edges, labels, max_figures=8, min_support=2, semantic_prior="disabled"
    )
    assert len(ranked) == 2
    assert len(disabled) == 2
    assert filtered == []
