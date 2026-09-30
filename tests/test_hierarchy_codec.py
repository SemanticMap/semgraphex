"""Multi-level exact tests for hierarchy_exact_v1."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.hierarchy_codec import (
    build_hierarchy_archive,
    decode_hierarchy,
    decode_hierarchy_bundle,
    write_decoded_hierarchy,
)
from semmap_haken.transition_grammar import (
    encode_transition_grammar,
    relation_layers_to_edge_records,
)


def _write_level(run, level, matrix):
    level_dir = run / f"level_{level:03d}"
    relation_dir = level_dir / "relations"
    relation_dir.mkdir(parents=True)
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    sparse.save_npz(level_dir / "adjacency.npz", matrix)
    sparse.save_npz(relation_dir / "r.npz", matrix)
    (relation_dir / "index.json").write_text(
        json.dumps({"r": "r.npz"}), encoding="utf-8"
    )
    membership = {
        str(node): {
            "original_concepts": [f"/c/en/l{level}_n{node}"],
            "concept_concat": f"/c/en/l{level}_n{node}",
        }
        for node in range(matrix.shape[0])
    }
    (level_dir / "membership.json").write_text(
        json.dumps(membership), encoding="utf-8"
    )
    return level_dir


def _contract(matrix, groups):
    n = matrix.shape[0]
    assignment = np.empty(n, dtype=np.int64)
    used = {node for group in groups for node in group}
    all_groups = [tuple(sorted(group)) for group in groups]
    all_groups += [(node,) for node in range(n) if node not in used]
    all_groups = sorted(
        all_groups, key=lambda group: (min(group), len(group), group)
    )
    for coarse, group in enumerate(all_groups):
        assignment[list(group)] = coarse
    p = sparse.csr_matrix(
        (
            np.ones(n),
            (np.arange(n), assignment),
        ),
        shape=(n, len(all_groups)),
    )
    return (p.T @ matrix @ p).tocsr()


def test_two_transition_hierarchy_decodes_exact_level_zero(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    level0 = sparse.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            (
                np.array([0, 2, 1, 3, 4]),
                np.array([1, 3, 4, 4, 5]),
            ),
        ),
        shape=(6, 6),
    )
    figures0 = ((0, 1), (2, 3))
    level1 = _contract(level0, figures0)

    # Second transition contracts the two macro nodes created above.
    figures1 = ((0, 1),)
    level2 = _contract(level1, figures1)

    _write_level(run, 0, level0)
    _write_level(run, 1, level1)
    _write_level(run, 2, level2)

    transition0 = run / "transition_000_001"
    transition0.mkdir()
    encode_transition_grammar(
        vertex_count=6,
        relation_layers={"r": level0},
        figure_nodes=figures0,
        symbol_types={},
        output=transition0 / "grammar_exact_v2.zip",
        source_adjacency=level0,
    )

    transition1 = run / "transition_001_002"
    transition1.mkdir()
    encode_transition_grammar(
        vertex_count=4,
        relation_layers={"r": level1},
        figure_nodes=figures1,
        symbol_types={0: "GT_A", 1: "GT_B"},
        output=transition1 / "grammar_exact_v2.zip",
        source_adjacency=level1,
    )

    (run / "hierarchy.json").write_text(
        json.dumps({"options": {"aggregation": "sum"}}),
        encoding="utf-8",
    )

    report = build_hierarchy_archive(run)
    assert report["roundtrip_exact"]
    assert report["transitions"] == 2
    assert report["final_nodes"] == 3
    assert report["level0_nodes"] == 6

    expected = relation_layers_to_edge_records({"r": level0})
    archive = run / "hierarchy_exact_v1.zip"
    assert decode_hierarchy(archive) == (6, expected)

    n, bundle_records, memberships, raw_adjacency = decode_hierarchy_bundle(
        archive
    )
    assert n == 6
    assert bundle_records == expected
    assert memberships[0] == ("/c/en/l0_n0",)
    assert len(memberships) == 6
    assert raw_adjacency is None

    decoded_dir = tmp_path / "decoded"
    materialized = write_decoded_hierarchy(archive, decoded_dir)
    assert materialized["roundtrip_materialized"]
    assert (decoded_dir / "COMPLETED").is_file()
    assert (decoded_dir / "membership.json").is_file()
    assert sparse.load_npz(decoded_dir / "adjacency.npz").shape == (6, 6)


def test_hierarchy_codec_rejects_non_sum_aggregation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_level(run, 0, sparse.csr_matrix((1, 1), dtype=float))
    (run / "hierarchy.json").write_text(
        json.dumps({"options": {"aggregation": "mean_density"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregation=sum"):
        build_hierarchy_archive(run)


def test_grammar_codec_cli_verify_and_decode(tmp_path, capsys):
    from semmap_haken.grammar_cli import main

    run = tmp_path / "run"
    run.mkdir()
    matrix = sparse.csr_matrix(
        ([1.0, 2.0], ([0, 1], [1, 2])),
        shape=(3, 3),
    )
    _write_level(run, 0, matrix)
    (run / "hierarchy.json").write_text(
        json.dumps({"options": {"aggregation": "sum"}}),
        encoding="utf-8",
    )
    report = build_hierarchy_archive(run)
    assert report["roundtrip_exact"]
    archive = run / "hierarchy_exact_v1.zip"

    assert main(["verify", "--archive", str(archive)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"]
    assert verified["nodes"] == 3

    output = tmp_path / "cli-decoded"
    assert main([
        "decode",
        "--archive",
        str(archive),
        "--output",
        str(output),
    ]) == 0
    decoded = json.loads(capsys.readouterr().out)
    assert decoded["roundtrip_materialized"]
    assert (output / "COMPLETED").is_file()
