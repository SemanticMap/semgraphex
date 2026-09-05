from __future__ import annotations

from pathlib import Path

from semmap_haken.cli import main
from semmap_haken.manifest import RunManifest


def test_prepare_cli_creates_sparse_artifact_and_manifest(tmp_path: Path, capsys) -> None:
    fixture = Path(__file__).parent / "fixtures" / "conceptnet_tiny.tsv"
    config = tmp_path / "smoke.yaml"
    config.write_text(f"""
paths: {{workspace_root: {tmp_path}, data_root: data, cache_root: cache, runs_root: runs}}
dataset: {{source: fixture, path: {fixture}, language: en, relations: [RelatedTo, IsA], min_weight: 1.0, max_nodes: 10, max_rows: 1, component: largest}}
graph: {{directed: false, weight_transform: raw, operator: normalized_adjacency}}
""", encoding="utf-8")
    assert main(["prepare", "--config", str(config)]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    assert (run_dir / "adjacency.npz").is_file()
    manifest = RunManifest.read_json(run_dir / "manifest.json")
    assert manifest.stages["prepare"] == "completed"
    assert manifest.resumability["source_identity"]["max_rows"] == 1
    assert "sha256" not in manifest.resumability["source_identity"]


def test_prepare_cli_accepts_empty_relations_as_no_relation_filter(tmp_path: Path, capsys) -> None:
    fixture = Path(__file__).parent / "fixtures" / "conceptnet_tiny.tsv"
    config = tmp_path / "all-relations.yaml"
    config.write_text(f"""
paths: {{workspace_root: {tmp_path}, data_root: data, cache_root: cache, runs_root: runs}}
dataset: {{source: fixture, path: {fixture}, language: "", relations: [], min_weight: 1.0, max_nodes: 10, max_rows: 3, component: largest}}
graph: {{directed: false, weight_transform: raw, operator: normalized_adjacency}}
""", encoding="utf-8")

    assert main(["prepare", "--config", str(config)]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    metadata = __import__("json").loads((run_dir / "graph_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parser_report"]["accepted"] == 3
    assert metadata["data_report"]["node_count"] > 0


def test_m1_cli_prepare_to_run_persists_checked_spectral_and_dynamics_artifacts(tmp_path: Path, capsys) -> None:
    fixture = Path(__file__).parent / "fixtures" / "conceptnet_tiny.tsv"
    prepare_config = tmp_path / "prepare.yaml"
    prepare_config.write_text(f"""
paths: {{workspace_root: {tmp_path}, data_root: data, cache_root: cache, runs_root: runs}}
dataset: {{source: fixture, path: {fixture}, language: en, relations: [], min_weight: 1.0, max_nodes: 10, component: largest}}
graph: {{directed: false, weight_transform: raw, operator: normalized_adjacency}}
""", encoding="utf-8")
    assert main(["prepare", "--config", str(prepare_config)]) == 0
    prepared = Path(capsys.readouterr().out.strip())
    run_config = tmp_path / "run.yaml"
    run_config.write_text(f"""
paths: {{workspace_root: {tmp_path}, data_root: data, cache_root: cache, runs_root: runs}}
dataset: {{source: fixture, language: en, relations: [], min_weight: 1.0, max_nodes: 10, component: largest}}
graph: {{directed: false, weight_transform: raw, operator: normalized_adjacency}}
runtime: {{random_seed: 11}}
dynamics: {{model: linear, alpha: 1.0, beta: 0.5, time_start: 0.0, time_stop: 1.0, time_steps: 5, perturbations_per_kind: 1, storage_policy: all}}
spectral: {{prepared_graph_dir: {prepared}, top_k: 3, max_r: 2}}
""", encoding="utf-8")
    assert main(["run", "--config", str(run_config)]) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    manifest = RunManifest.read_json(run_dir / "manifest.json")
    assert manifest.stages == {"spectral": "completed", "dynamics": "completed"}
    assert (run_dir / "COMPLETED").is_file()
    summary = __import__("json").loads((run_dir / "dynamics" / "dynamics_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["spectral_abscissa"] < 0
    assert summary["numeric_checksum_sha256"] == __import__("hashlib").sha256((run_dir / "dynamics" / "dynamics_arrays.npz").read_bytes()).hexdigest()
