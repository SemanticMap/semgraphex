"""Colab GPU, per-level restart and Drive configuration contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.graph_build import PreparedGraph
from semmap_haken.wishart_config import DictionaryOptions, WishartOptions
from semmap_haken.wishart_metrics import extract_ego_candidates


def _tiny_graph() -> PreparedGraph:
    n = 8
    node_ids = tuple(f"/c/en/concept{i}" for i in range(n))
    rows: list[int] = []
    cols: list[int] = []
    provenance: list[dict[str, object]] = []
    for index in range(n - 1):
        rows.extend((index, index + 1))
        cols.extend((index + 1, index))
        provenance.append({
            "start_uri": node_ids[index],
            "end_uri": node_ids[index + 1],
            "relation": "RelatedTo",
            "aggregate_contribution": 1.0,
        })
    matrix = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n),
        dtype=float,
    )
    return PreparedGraph(matrix, node_ids, {"node_count": n}, tuple(provenance))


def _options() -> WishartOptions:
    return WishartOptions(
        radius=1,
        max_ego_nodes=3,
        candidate_limit=8,
        k_neighbors=2,
        min_cluster_size=1,
        min_cluster_mass=1.0,
        min_figure_nodes=2,
        max_figures_per_level=10,
        max_levels=2,
        min_graph_nodes=2,
        slow_modes=2,
        mfpt_pairs=2,
        mfpt_walks_per_pair=1,
        mfpt_max_steps=20,
        betweenness_samples=4,
        clustering_samples=4,
        distance_samples=4,
        random_seed=7,
    )


def test_parallel_ego_extraction_matches_serial_exactly() -> None:
    graph = _tiny_graph()
    relation_layers = {"RelatedTo": graph.adjacency}
    kwargs = {
        "radius": 1,
        "max_ego_nodes": 3,
        "candidate_limit": 8,
        "seed": 7,
    }
    serial = extract_ego_candidates(
        graph.adjacency, relation_layers, workers=1, **kwargs,
    )
    parallel = extract_ego_candidates(
        graph.adjacency, relation_layers, workers=2, **kwargs,
    )
    assert [item.center for item in serial] == [item.center for item in parallel]
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(left.nodes, right.nodes)
        np.testing.assert_allclose(left.adjacency.toarray(), right.adjacency.toarray())
        assert left.boundary_signature == right.boundary_signature


def test_backend_selection_falls_back_to_cpu_but_rejects_forced_missing_cuda() -> None:
    from semmap_haken.wishart_gpu import resolve_device

    assert resolve_device("auto", cuda_available=False) == "cpu"
    assert resolve_device("cpu", cuda_available=True) == "cpu"
    assert resolve_device("auto", cuda_available=True) == "cuda"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", cuda_available=False)


def test_checkpoint_rejects_config_mismatch_and_corrupt_latest(tmp_path: Path) -> None:
    from semmap_haken.wishart_resume import (
        load_latest_checkpoint,
        write_level_checkpoint,
    )

    graph = _tiny_graph()
    kwargs = {
        "current": graph.adjacency,
        "relation_layers": {"RelatedTo": graph.adjacency},
        "memberships": {i: (name,) for i, name in enumerate(graph.node_ids)},
        "symbol_types": {},
        "dictionary": {"GT_1": "test"},
        "family_registry": {"WF_1": "test"},
        "current_huffman": {"GT_1": "0"},
        "level_summaries": [{"level": 0}],
        "transition_summaries": [],
        "config_hash": "config-A",
        "input_hash": "graph-A",
        "code_revision": "revision-A",
    }
    write_level_checkpoint(tmp_path, next_level=0, **kwargs)
    write_level_checkpoint(tmp_path, next_level=1, **kwargs)
    latest = tmp_path / "checkpoints" / "level_001" / "state.pkl.gz"
    latest.write_bytes(b"corrupt")

    state = load_latest_checkpoint(
        tmp_path,
        config_hash="config-A",
        input_hash="graph-A",
        code_revision="revision-A",
    )
    assert state is not None
    assert state["next_level"] == 0
    with pytest.raises(ValueError, match="configuration"):
        load_latest_checkpoint(
            tmp_path, config_hash="changed",
            input_hash="graph-A", code_revision="revision-A",
        )


def test_runner_resume_after_interruption_reuses_dictionary_state(tmp_path: Path) -> None:
    from semmap_haken.wishart_hierarchy import run_wishart_hierarchy

    graph = _tiny_graph()
    run_dir = tmp_path / "run"
    checkpoint_events: list[int] = []

    def interrupt_after_contraction(_root: Path, event: dict[str, object]) -> None:
        if event.get("stage") == "checkpoint":
            next_level = int(event["next_level"])
            checkpoint_events.append(next_level)
            if next_level == 1:
                raise RuntimeError("simulated session loss")

    with pytest.raises(RuntimeError, match="simulated session loss"):
        run_wishart_hierarchy(
            graph,
            directed=False,
            options=_options(),
            dictionary_options=DictionaryOptions(
                boundary_sensitive=False, frequency_scan_batch_size=3,
                min_support=2,
            ),
            output_dir=run_dir,
            checkpoint_hook=interrupt_after_contraction,
            checkpoint_config_hash="config-A",
            checkpoint_input_hash="graph-A",
            checkpoint_code_revision="revision-A",
        )
    assert checkpoint_events[:2] == [0, 1]

    resumed = run_wishart_hierarchy(
        graph,
        directed=False,
        options=_options(),
        dictionary_options=DictionaryOptions(
            boundary_sensitive=False, frequency_scan_batch_size=3,
            min_support=2,
        ),
        output_dir=run_dir,
        resume=True,
        checkpoint_config_hash="config-A",
        checkpoint_input_hash="graph-A",
        checkpoint_code_revision="revision-A",
    )
    assert resumed.final_nodes < 8
    assert (run_dir / "COMPLETED").exists()
    hierarchy = json.loads((run_dir / "hierarchy.json").read_text(encoding="utf-8"))
    assert len([item for item in hierarchy["levels_detail"] if item["level"] == 0]) == 1
    assert (run_dir / "checkpoints" / "level_001" / "manifest.json").is_file()
    assert (run_dir / "dictionary" / "graph_types.jsonl").is_file()


def test_colab_cli_loads_configuration_from_drive(tmp_path: Path) -> None:
    from semmap_haken.wishart_colab_cli import main

    root = tmp_path / "drive"
    (root / "config").mkdir(parents=True)
    config = root / "config" / "experiment.yaml"
    config.write_bytes(Path("configs/wishart_conceptnet_small.yaml").read_bytes())
    (root / "data").mkdir()
    (root / "data" / "conceptnet_tiny.tsv").write_bytes(
        Path("tests/fixtures/conceptnet_tiny.tsv").read_bytes()
    )
    code = main([
        "--config-drive", "config/experiment.yaml",
        "--drive-root", str(root),
        "--scratch-root", str(tmp_path / "scratch"),
        "--dataset-drive", "data/conceptnet_tiny.tsv",
        "--run-name", "drive-config",
        "--device", "cpu",
        "--keep-scratch",
    ])
    assert code == 0
    run = root / "runs" / "drive-config"
    assert (run / "COMPLETED").is_file()
    manifest = json.loads((run / "input.json").read_text(encoding="utf-8"))
    assert manifest["config_source"] == str(config)
    assert manifest["device"]["selected"] == "cpu"


def test_graphlet_forwards_cuda_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from semmap_haken import wishart_metrics as metrics

    candidate_graph = _tiny_graph()
    candidates = extract_ego_candidates(
        candidate_graph.adjacency,
        {"RelatedTo": candidate_graph.adjacency},
        radius=1, max_ego_nodes=3, candidate_limit=8, seed=7,
    )
    seen: dict[str, object] = {}

    def fake_knn(
        features: sparse.spmatrix, k: int, metric: str, *,
        device: str, gpu_batch_size: int, min_gpu_types: int,
    ) -> metrics.NeighborGraph:
        seen.update(metric=metric, device=device, batch=gpu_batch_size, minimum=min_gpu_types)
        count = features.shape[0]
        return metrics.NeighborGraph(
            np.empty((count, 0), dtype=np.int64),
            np.empty((count, 0), dtype=float), {"backend": "test"},
        )

    monkeypatch.setattr(metrics, "_knn_from_features", fake_knn)
    metrics.build_neighbor_graph(
        candidates, metric="graphlet", k=2,
        wl_iterations=2, feature_dim=32, graphlet_size=3, graphlet_samples=4,
        transport_rank=4, transport_max_candidates=16, fgw_alpha=0.5,
        relation_js_block_size=8, seed=7, device="cuda",
        gpu_batch_size=7, min_gpu_types=11,
    )
    assert seen == {"metric": "cosine", "device": "cuda", "batch": 7, "minimum": 11}


def test_gpu_diagnostics_has_actionable_reason_on_cpu() -> None:
    from semmap_haken.wishart_gpu import cuda_diagnostics

    payload = cuda_diagnostics()
    assert "torch_importable" in payload
    assert "cuda_available" in payload
    if not payload["cuda_available"]:
        assert isinstance(payload.get("reason"), str) and payload["reason"]


def test_resume_device_mismatch_explains_separate_runs() -> None:
    from semmap_haken.wishart_colab_cli import _assert_resume_device_consistent

    _assert_resume_device_consistent("cpu", "cpu")
    with pytest.raises(ValueError, match="new run name"):
        _assert_resume_device_consistent("cpu", "cuda")


def test_discovery_process_pool_matches_serial() -> None:
    from semmap_haken.graph_dictionary import GraphDictionary
    from semmap_haken.wishart_hierarchy import _discover_types

    graph = _tiny_graph()
    candidates = extract_ego_candidates(
        graph.adjacency, {"RelatedTo": graph.adjacency},
        radius=1, max_ego_nodes=3, candidate_limit=8, seed=7,
    )
    baseline = _discover_types(
        candidates, GraphDictionary(boundary_sensitive=False),
        level=0, cpu_workers=1,
    )
    parallel = _discover_types(
        candidates, GraphDictionary(boundary_sensitive=False),
        level=0, cpu_workers=2,
    )
    assert baseline == parallel


def test_full_scan_spawn_process_pool_matches_serial() -> None:
    from semmap_haken.graph_dictionary import GraphDictionary
    from semmap_haken.wishart_hierarchy import _discover_types, _scan_known_types

    graph = _tiny_graph()
    candidates = extract_ego_candidates(
        graph.adjacency, {"RelatedTo": graph.adjacency},
        radius=1, max_ego_nodes=3, candidate_limit=8, seed=7,
    )
    def run(workers: int):
        dictionary = GraphDictionary(boundary_sensitive=False)
        type_ids, _ = _discover_types(
            candidates, dictionary, level=0, cpu_workers=workers,
        )
        return _scan_known_types(
            graph.adjacency, {"RelatedTo": graph.adjacency}, dictionary,
            level=0, discovery_candidates=candidates, discovery_type_ids=type_ids,
            symbol_types={}, wishart_options=_options(),
            dictionary_options=DictionaryOptions(
                boundary_sensitive=False, frequency_scan="full",
                frequency_scan_batch_size=3, min_support=2,
            ),
            cpu_workers=workers,
        )
    assert run(1) == run(2)


def test_sparse_relation_contraction_parallel_matches_serial() -> None:
    from semmap_haken.wishart_hierarchy import _contract_relation_layers

    graph = _tiny_graph()
    layers = {
        "RelatedTo": graph.adjacency,
        "Half": graph.adjacency.multiply(0.5).tocsr(),
    }
    assignment = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    serial = _contract_relation_layers(layers, assignment, aggregation="sum", workers=1)
    parallel = _contract_relation_layers(layers, assignment, aggregation="sum", workers=2)
    for name in layers:
        np.testing.assert_array_equal(serial[name].indptr, parallel[name].indptr)
        np.testing.assert_array_equal(serial[name].indices, parallel[name].indices)
        np.testing.assert_array_equal(serial[name].data, parallel[name].data)


def test_wl_feature_process_pool_matches_serial() -> None:
    from semmap_haken.wishart_metrics import typed_wl_features

    graph = _tiny_graph()
    candidates = extract_ego_candidates(
        graph.adjacency, {"RelatedTo": graph.adjacency},
        radius=1, max_ego_nodes=3, candidate_limit=8, seed=7,
    )
    serial = typed_wl_features(candidates, iterations=2, dimension=128, workers=1)
    parallel = typed_wl_features(candidates, iterations=2, dimension=128, workers=2)
    np.testing.assert_array_equal(serial.indptr, parallel.indptr)
    np.testing.assert_array_equal(serial.indices, parallel.indices)
    np.testing.assert_array_equal(serial.data, parallel.data)


def test_threaded_diagnostics_are_serial_equivalent() -> None:
    from semmap_haken.wishart_dynamics import (
        _binary_topology, _sampled_clustering, _sampled_path_distances,
    )
    topology = _binary_topology(_tiny_graph().adjacency)
    a = _sampled_clustering(
        topology, sample_count=8, rng=np.random.default_rng(19),
        workers=1,
    )
    b = _sampled_clustering(
        topology, sample_count=8, rng=np.random.default_rng(19),
        workers=3,
    )
    assert a == b
    serial = _sampled_path_distances(
        topology, sample_count=8, rng=np.random.default_rng(19),
        workers=1,
    )
    parallel = _sampled_path_distances(
        topology, sample_count=8, rng=np.random.default_rng(19),
        workers=3,
    )
    np.testing.assert_array_equal(serial, parallel)


def test_parallel_graphlet_features_are_byte_identical() -> None:
    from semmap_haken.wishart_metrics import graphlet_features

    graph = _tiny_graph()
    candidates = extract_ego_candidates(
        graph.adjacency, {"RelatedTo": graph.adjacency},
        radius=1, max_ego_nodes=3, candidate_limit=8, seed=7,
    )
    one = graphlet_features(
        candidates, graphlet_size=3, samples=4,
        dimension=128, seed=7, workers=1,
    )
    many = graphlet_features(
        candidates, graphlet_size=3, samples=4,
        dimension=128, seed=7, workers=2,
    )
    np.testing.assert_array_equal(one.indptr, many.indptr)
    np.testing.assert_array_equal(one.indices, many.indices)
    np.testing.assert_array_equal(one.data, many.data)


def test_runner_records_per_level_phase_timing(tmp_path: Path) -> None:
    from semmap_haken.wishart_hierarchy import run_wishart_hierarchy

    output = tmp_path / "timed"
    run_wishart_hierarchy(
        _tiny_graph(), directed=False,
        options=_options(),
        dictionary_options=DictionaryOptions(
            boundary_sensitive=False, min_support=2,
            frequency_scan_batch_size=3,
        ),
        output_dir=output,
    )
    hierarchy = json.loads((output / "hierarchy.json").read_text(encoding="utf-8"))
    first_level = hierarchy["levels_detail"][0]
    timings = first_level["phase_timing_seconds"]
    assert timings["dynamic_snapshot"] >= 0
    assert timings["discovery"] >= 0
    assert timings["full_scan"] >= 0
    assert timings["wishart_knn_clustering"] >= 0
    assert timings["mdl_scoring"] >= 0


def test_sampled_brandes_spawn_matches_serial_bit_for_bit() -> None:
    from semmap_haken.wishart_dynamics import (
        _binary_topology, _sampled_betweenness_unweighted,
    )
    topology = _binary_topology(_tiny_graph().adjacency)
    serial = _sampled_betweenness_unweighted(
        topology, sample_count=6, rng=np.random.default_rng(47), workers=1,
    )
    process_pool = _sampled_betweenness_unweighted(
        topology, sample_count=6, rng=np.random.default_rng(47), workers=2,
    )
    np.testing.assert_array_equal(serial, process_pool)
