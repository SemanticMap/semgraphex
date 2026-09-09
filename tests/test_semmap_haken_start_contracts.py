from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semmap_haken.run_contracts import validate_prepared_graph_contract, validate_start_run


def _config() -> SimpleNamespace:
    return SimpleNamespace(resolved={
        "dataset": {
            "source": "conceptnet-5.7", "version": "5.7.0", "language": "en",
            "relations": ["RelatedTo", "IsA", "PartOf", "HasA", "UsedFor", "HasProperty"],
            "min_weight": 1.0, "max_nodes": 1000, "component": "largest", "max_rows": 1000,
        },
        "graph": {"directed": False, "self_loop_policy": "exclude", "weight_transform": "log1p", "operator": "normalized_adjacency"},
    })


def test_prepared_contract_requires_fingerprint_and_exact_scientific_config(tmp_path: Path) -> None:
    prepared = tmp_path / "graph"
    prepared.mkdir()
    (prepared / "COMPLETED").write_text("complete\n", encoding="utf-8")
    (prepared / "resolved_config.json").write_text(json.dumps(_config().resolved), encoding="utf-8")
    (prepared / "graph_metadata.json").write_text(json.dumps({"source_identity": {"input_sha256": "a" * 64, "input_size_bytes": 123}}), encoding="utf-8")
    summary = validate_prepared_graph_contract(prepared, _config())
    assert summary["language"] == "en"

    bad = _config().resolved.copy()
    bad = json.loads(json.dumps(bad))
    bad["dataset"]["language"] = ""
    mismatch = SimpleNamespace(resolved=bad)
    with pytest.raises(ValueError, match="dataset.language"):
        validate_prepared_graph_contract(prepared, mismatch)


def test_start_run_gate_requires_both_m2_methods_and_distortion(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "COMPLETED").write_text("complete\n", encoding="utf-8")
    manifest = {
        "stages": {"spectral": "completed", "dynamics": "completed", "coarsening": "completed"},
        "resolved_config": {"coarsening": {"methods": ["connectivity_matching", "unconstrained_matching"]}},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for method in manifest["resolved_config"]["coarsening"]["methods"]:
        directory = run / "coarsening" / method
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({"result": {
            "fine_node_count": 100, "coarse_node_count": 60, "achieved_reduction": 0.4,
            "subspace_projection_distance": 0.2, "slow_eigenvalue_max_abs_error": 0.1,
            "mean_trajectory_relative_error": 0.3,
        }}), encoding="utf-8")
    summary = validate_start_run(run)
    assert summary["status"] == "start_smoke_passed"
    assert set(summary["methods"]) == {"connectivity_matching", "unconstrained_matching"}
