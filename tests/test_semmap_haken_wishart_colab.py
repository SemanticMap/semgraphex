from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from semmap_haken.wishart_colab import (
    DriveCheckpointSync,
    preflight_colab_storage,
    stage_from_drive,
    sync_tree,
)
from semmap_haken.wishart_dynamics import (
    _binary_topology,
    _sampled_betweenness_unweighted,
    _sampled_clustering,
)
from semmap_haken.wishart_metrics import EgoCandidate, relation_js_neighbors


def _candidate(relation_weights: dict[str, float]) -> EgoCandidate:
    n = 3
    adjacency = sparse.csr_matrix(
        np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
    )
    layers: dict[str, sparse.csr_matrix] = {}
    for relation, weight in relation_weights.items():
        layer = sparse.csr_matrix((n, n), dtype=float)
        layer[0, 1] = weight
        layer.eliminate_zeros()
        layers[relation] = layer
    return EgoCandidate(0, np.arange(n), adjacency, layers)


def test_relation_js_chunking_is_block_size_invariant() -> None:
    candidates = (
        _candidate({"IsA": 1.0}),
        _candidate({"IsA": 1.0, "UsedFor": 1.0}),
        _candidate({"UsedFor": 1.0}),
        _candidate({}),
    )
    tiny = relation_js_neighbors(candidates, k=3, block_size=1)
    wide = relation_js_neighbors(candidates, k=3, block_size=4)
    np.testing.assert_array_equal(tiny.indices, wide.indices)
    np.testing.assert_allclose(tiny.distances, wide.distances, atol=1e-12)


def test_csr_sampled_betweenness_matches_networkx_when_all_sources_used() -> None:
    graph = nx.path_graph(5)
    adjacency = nx.to_scipy_sparse_array(graph, format="csr", dtype=float)
    topology = _binary_topology(adjacency)
    actual = _sampled_betweenness_unweighted(
        topology,
        sample_count=5,
        rng=np.random.default_rng(1),
    )
    expected_map = nx.betweenness_centrality(
        graph,
        normalized=False,
        weight=None,
    )
    expected = np.array([expected_map[i] for i in range(5)], dtype=float)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_csr_sampled_clustering_matches_networkx_when_all_nodes_used() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(4), format="csr", dtype=float)
    topology = _binary_topology(adjacency)
    actual = _sampled_clustering(
        topology,
        sample_count=4,
        rng=np.random.default_rng(2),
    )
    expected = float(np.mean(list(nx.clustering(graph).values())))
    assert actual is not None
    assert np.isclose(actual, expected)


def test_drive_checkpoint_does_not_publish_completed_until_final(tmp_path: Path) -> None:
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    local.mkdir()
    (local / "level_000").mkdir()
    (local / "level_000" / "adjacency.npz").write_bytes(b"graph")
    (local / "COMPLETED").write_text("complete\n", encoding="utf-8")

    partial = sync_tree(local, drive, final=False)
    assert partial.copied_files == 1
    assert (drive / "level_000" / "adjacency.npz").is_file()
    assert not (drive / "COMPLETED").exists()

    final = sync_tree(local, drive, final=True)
    assert final.copied_files >= 1
    assert (drive / "COMPLETED").is_file()


def test_drive_checkpoint_hook_writes_event_metadata_and_publishes_completed_last(
    tmp_path: Path,
) -> None:
    local = tmp_path / "run"
    drive = tmp_path / "durable"
    local.mkdir()
    (local / "input.json").write_text("{}", encoding="utf-8")
    (local / "COMPLETED").write_text("complete\n", encoding="utf-8")
    hook = DriveCheckpointSync(drive_run_dir=drive)

    hook(local, {"stage": "completed", "level": 0})
    payload = json.loads((drive / "DRIVE_CHECKPOINT.json").read_text(encoding="utf-8"))
    assert payload["event"]["stage"] == "completed"
    assert (drive / "input.json").is_file()
    assert not (drive / "COMPLETED").exists()

    hook(local, {"stage": "published", "level": 0})
    payload = json.loads((drive / "DRIVE_CHECKPOINT.json").read_text(encoding="utf-8"))
    assert payload["event"]["stage"] == "published"
    assert (drive / "COMPLETED").is_file()


def test_stage_completed_prepared_tree_to_scratch(tmp_path: Path) -> None:
    drive_prepared = tmp_path / "drive" / "prepare-abc"
    drive_prepared.mkdir(parents=True)
    (drive_prepared / "adjacency.npz").write_bytes(b"abc")
    (drive_prepared / "COMPLETED").write_text("complete\n", encoding="utf-8")
    scratch = tmp_path / "scratch" / "prepare-abc"

    stats = stage_from_drive(
        drive_prepared,
        scratch,
        require_completed=True,
    )
    assert stats.copied_files == 2
    assert (scratch / "adjacency.npz").read_bytes() == b"abc"
    assert (scratch / "COMPLETED").is_file()


def test_colab_preflight_uses_local_scratch_capacity(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 1024)
    scratch = tmp_path / "scratch"
    drive = tmp_path / "drive"
    result = preflight_colab_storage(
        scratch_root=scratch,
        source_path=source,
        drive_root=drive,
        scratch_multiplier=1.0,
        minimum_extra_bytes=0,
    )
    assert result.ok is True
    assert result.source_bytes == 1024
    assert result.recommended_scratch_bytes == 1024


def test_colab_cli_end_to_end_with_fake_drive_and_tiny_conceptnet(
    tmp_path: Path,
) -> None:
    from semmap_haken.wishart_colab_cli import main

    drive_root = tmp_path / "drive"
    drive_data = drive_root / "data"
    drive_data.mkdir(parents=True)
    source_fixture = Path("tests/fixtures/conceptnet_tiny.tsv")
    dataset = drive_data / "conceptnet_tiny.tsv"
    dataset.write_bytes(source_fixture.read_bytes())

    scratch = tmp_path / "scratch"
    exit_code = main([
        "--config",
        "configs/wishart_conceptnet_small.yaml",
        "--drive-root",
        str(drive_root),
        "--scratch-root",
        str(scratch),
        "--dataset-drive",
        "data/conceptnet_tiny.tsv",
        "--run-name",
        "e2e",
        "--keep-scratch",
    ])
    assert exit_code == 0
    durable = drive_root / "runs" / "e2e"
    assert (durable / "input.json").is_file()
    assert (durable / "hierarchy.json").is_file()
    assert (durable / "COLAB_RUN.json").is_file()
    assert (durable / "DRIVE_CHECKPOINT.json").is_file()
    assert (durable / "dictionary" / "graph_types.jsonl").is_file()
    assert (durable / "dictionary" / "grammar.jsonl").is_file()
    assert (durable / "dictionary" / "huffman.json").is_file()
    assert (durable / "level_000" / "symbolic_nodes.jsonl").is_file()
    assert (durable / "COMPLETED").is_file()
