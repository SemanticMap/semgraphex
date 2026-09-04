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
