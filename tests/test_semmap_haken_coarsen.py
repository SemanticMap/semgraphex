"""Focused B1 contracts for topology-only one-step Haken coarsening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.coarsen import build_partition
from semmap_haken.haken_embedding import build_haken_embedding
from semmap_haken.modes import ModeResult


def _modes() -> ModeResult:
    vectors = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 0.1, 2.0],
            [10.0, 0.2, 1.0],
            [10.0, 3.0, 0.0],
        ]
    )
    from semmap_haken.modes import BetaSelection, DimensionSelection

    return ModeResult(
        eigenvalues=np.array([1.0, 0.8, 0.2]),
        eigenvectors=vectors,
        residual_norms=np.zeros(3),
        growth_rates=np.array([-0.1, -0.2, -0.8]),
        relaxation_times=np.array([10.0, 5.0, 1.25]),
        inverse_participation_ratios=np.ones(3),
        participation_ratios=np.ones(3),
        localization_scores=np.ones(3),
        degree_correlations=np.zeros(3),
        eigengaps=np.array([0.2, 0.6]),
        timescale_gaps=np.array([0.1, 0.2]),
        beta_selection=BetaSelection(1.0, 0.05, 0.8, 1.0, 0.5, -0.5, -0.6, False, ()),
        selection=DimensionSelection(2, 2, 2, "agreement", (2,)),
        solver={},
    )


def _path_adjacency(size: int) -> sparse.csr_matrix:
    row = np.arange(size - 1)
    return sparse.csr_matrix(
        (np.ones(2 * (size - 1)), (np.r_[row, row + 1], np.r_[row + 1, row])),
        shape=(size, size),
    )


def test_embedding_excludes_perron_and_uses_spectral_arrays_only() -> None:
    embedding = build_haken_embedding(_modes(), weighting="none")

    assert embedding.coordinates.shape == (4, 2)
    assert np.array_equal(embedding.coordinates, _modes().eigenvectors[:, 1:3])
    assert embedding.selected_mode_indices == (1, 2)
    assert "candidate slow-mode coordinates" in embedding.caveats[0]


def test_embedding_optional_relaxation_weighting_is_explicit() -> None:
    embedding = build_haken_embedding(_modes(), weighting="relaxation_time")

    assert np.allclose(embedding.coordinates[:, 0], _modes().eigenvectors[:, 1] * 5.0)
    assert np.allclose(embedding.coordinates[:, 1], _modes().eigenvectors[:, 2] * 1.25)


def test_connectivity_partition_is_complete_deterministic_and_only_merges_edges() -> None:
    adjacency = _path_adjacency(4)
    coordinates = np.array([[0.0], [0.0], [2.0], [2.0]])

    first = build_partition(adjacency, coordinates, method="connectivity_matching", target_reduction=0.5, seed=7)
    second = build_partition(adjacency, coordinates, method="connectivity_matching", target_reduction=0.5, seed=7)

    assert np.array_equal(first.fine_to_coarse, second.fine_to_coarse)
    assert first.requested_merges == first.achieved_merges == 2
    assert sorted(node for cluster in first.clusters for node in cluster) == [0, 1, 2, 3]
    assert all(len(cluster) in {1, 2} for cluster in first.clusters)
    assert all(adjacency[left, right] != 0 for left, right, _distance in first.merges)


def test_connectivity_partition_records_shortfall_when_edges_are_insufficient() -> None:
    adjacency = sparse.csr_matrix((4, 4))
    partition = build_partition(adjacency, np.arange(4, dtype=float)[:, None], method="connectivity_matching", target_reduction=0.5, seed=0)

    assert partition.achieved_merges == 0
    assert partition.shortfall_reason == "insufficient_disjoint_adjacent_pairs"
    assert partition.coarse_node_count == 4


def test_unconstrained_partition_avoids_dense_pairwise_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinates = np.arange(200, dtype=float)[:, None]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unconstrained matching must not allocate an N-by-N distance matrix")

    monkeypatch.setattr(np, "subtract", forbidden)
    partition = build_partition(sparse.csr_matrix((200, 200)), coordinates, method="unconstrained_matching", target_reduction=0.5, seed=0)

    assert partition.achieved_merges == 100
    assert partition.coarse_node_count == 100


def test_cli_one_step_coarsening_end_to_end_smoke(tmp_path: Path) -> None:
    """Prepare -> M1 -> M2 CLI smoke on a synthetic graph with enough nodes for eigsh."""
    import json

    from semmap_haken.cli import main

    size = 12
    lines = []
    for index in range(size):
        left = (index - 1) % size
        right = (index + 1) % size
        for neighbour in (left, right):
            lines.append(
                f"/a/r/edge\t/r/RelatedTo\t/c/en/n{index:02d}\t/c/en/n{neighbour:02d}\t"
                + json.dumps({"weight": 1.0, "dataset": "fixture", "sources": [], "license": " fixture"})
            )
    fixture = tmp_path / "fixture.tsv"
    fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = workspace / "profiles.yaml"
    profile.write_text("name: smoke\nexpected_max_nodes: 1000\n", encoding="utf-8")

    def write_config(name: str, *, coarsening_block: str, prepared_dir: str) -> Path:
        config = workspace / name
        config.write_text(
            "\n".join(
                [
                    f"paths: {{workspace_root: {workspace}, data_root: data, cache_root: cache, runs_root: runs}}",
                    f"dataset: {{source: fixture, path: {fixture}, language: en, relations: [RelatedTo], min_weight: 1.0, max_nodes: 1000, component: largest}}",
                    "graph: {directed: false, weight_transform: log1p, operator: normalized_adjacency}",
                    f"runtime: {{profile: smoke, random_seed: 1729, resource_profile: {profile}}}",
                    "execution: {backend: cpu, dtype: float64, workers: 1, threads_per_worker: 1}",
                    "dynamics: {model: linear, alpha: 1.0, beta: auto_critical, time_start: 0.0, time_stop: 2.0, time_steps: 9, perturbations_per_kind: 1, perturbation_amplitude: 1.0, perturbation_seed: 1729, random_sparse_fraction: 0.25, storage_policy: all, max_storage_mb: 32}",
                    f"spectral: {{prepared_graph_dir: {prepared_dir}, top_k: 8, max_r: 4}}",
                    coarsening_block,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    import contextlib
    import io as _io

    with contextlib.redirect_stdout(_io.StringIO()) as stream:
        assert main(["prepare", "--config", str(write_config("prepare.yaml", coarsening_block="", prepared_dir="null"))]) == 0
    prepared_dir = stream.getvalue().strip()
    run_config = write_config(
        "run_m2.yaml",
        coarsening_block="coarsening: {enabled: true, methods: [connectivity_matching, unconstrained_matching], target_reduction: 0.4, seed: 1729}",
        prepared_dir=prepared_dir,
    )
    with contextlib.redirect_stdout(_io.StringIO()) as stream:
        assert main(["run", "--config", str(run_config)]) == 0
    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    run_dir = Path(payload["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"] == {"spectral": "completed", "dynamics": "completed", "coarsening": "completed"}
    assert set(manifest["random_seeds"]) == {"runtime", "perturbations", "coarsening"}
    for method in ("connectivity_matching", "unconstrained_matching"):
        metrics = json.loads((run_dir / "coarsening" / method / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["schema_version"] == 1
        result = metrics["result"]
        assert result["fine_node_count"] == size and result["coarse_node_count"] >= 3
        assert metrics["execution"]["cpu_reference_enforced"] is True
        mapping = json.loads((run_dir / "coarsening" / method / "mapping.json").read_text(encoding="utf-8"))
        merged = [sorted(fine_ids) for fine_ids in mapping["parent_child"].values()]
        assert sorted(node for members in merged for node in members) == [f"/c/en/n{index:02d}" for index in range(size)]
        if method == "connectivity_matching":
            fine_graph = __import__("semmap_haken.graph_build", fromlist=["load_prepared_graph"]).load_prepared_graph(prepared_dir)
            for merge in mapping["merges"]:
                assert fine_graph.adjacency[int(merge["left"]), int(merge["right"])] != 0
            assert sparse.load_npz(run_dir / "coarsening" / method / "quotient.npz").shape[0] == result["coarse_node_count"]
    # M1-only behavior remains when coarsening is absent.
    m1_config = write_config(
        "run_m1.yaml",
        coarsening_block="",
        prepared_dir=prepared_dir,
    )
    with contextlib.redirect_stdout(_io.StringIO()) as stream:
        assert main(["run", "--config", str(m1_config)]) == 0
    m1_payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    m1_manifest = json.loads((Path(m1_payload["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert m1_manifest["stages"] == {"spectral": "completed", "dynamics": "completed"}
    assert m1_payload["coarsening"] is None


def test_m2_forces_cpu_float64_reference_execution_and_records_artifact_checksums(tmp_path: Path) -> None:
    """M2 must override a non-reference execution request rather than mislabel it."""
    import contextlib
    import hashlib
    import io as _io
    import json

    from semmap_haken.cli import main

    size = 8
    fixture = tmp_path / "ring.tsv"
    records = []
    for index in range(size):
        neighbour = (index + 1) % size
        records.extend(
            [
                f"/a/r/edge\t/r/RelatedTo\t/c/en/n{index}\t/c/en/n{neighbour}\t"
                + json.dumps({"weight": 1.0, "dataset": "fixture", "sources": [], "license": "fixture"}),
                f"/a/r/edge\t/r/RelatedTo\t/c/en/n{neighbour}\t/c/en/n{index}\t"
                + json.dumps({"weight": 1.0, "dataset": "fixture", "sources": [], "license": "fixture"}),
            ]
        )
    fixture.write_text("\n".join(records) + "\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    profile = workspace / "profile.yaml"
    workspace.mkdir()
    profile.write_text("name: smoke\nexpected_max_nodes: 100\n", encoding="utf-8")

    def write_config(name: str, prepared_dir: str) -> Path:
        config = workspace / name
        config.write_text(
            "\n".join(
                [
                    f"paths: {{workspace_root: {workspace}, runs_root: runs}}",
                    f"dataset: {{source: fixture, path: {fixture}, language: en, relations: [RelatedTo], min_weight: 1.0, max_nodes: 100, component: largest}}",
                    "graph: {directed: false, weight_transform: raw, operator: normalized_adjacency}",
                    f"runtime: {{profile: smoke, resource_profile: {profile}}}",
                    "execution: {backend: cpu, dtype: float32, workers: 1, threads_per_worker: 1}",
                    "dynamics: {model: linear, alpha: 1.0, beta: 0.5, time_start: 0.0, time_stop: 1.0, time_steps: 3, perturbations_per_kind: 1, random_sparse_fraction: 0.25, storage_policy: all, max_storage_mb: 32}",
                    f"spectral: {{prepared_graph_dir: {prepared_dir}, top_k: 5, max_r: 3}}",
                    "coarsening: {enabled: true, methods: [connectivity_matching, unconstrained_matching], target_reduction: 0.25}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    with contextlib.redirect_stdout(_io.StringIO()) as stream:
        assert main(["prepare", "--config", str(write_config("prepare.yaml", "null"))]) == 0
    prepared_dir = stream.getvalue().strip()
    with contextlib.redirect_stdout(_io.StringIO()) as stream:
        assert main(["run", "--config", str(write_config("run.yaml", prepared_dir))]) == 0
    run_dir = Path(json.loads(stream.getvalue().strip().splitlines()[-1])["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_telemetry"]["execution"]["backend"] == "cpu"
    assert manifest["execution_telemetry"]["execution"]["dtype"] == "float64"
    for method in ("connectivity_matching", "unconstrained_matching"):
        method_dir = run_dir / "coarsening" / method
        metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["execution"]["backend"] == "cpu"
        assert metrics["execution"]["dtype"] == "float64"
        for name, expected_checksum in metrics["artifacts_checksums"].items():
            assert hashlib.sha256((method_dir / name).read_bytes()).hexdigest() == expected_checksum
