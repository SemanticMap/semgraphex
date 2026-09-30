"""Regression tests for exact internal-shape sharing and boundary diagnostics."""
import gzip
import json

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.wishart_shared_shapes import _restore, export_run, factor_dictionary


def _proto(ports, swapped=False):
    return {
        "nodes": [{"prototype_node": i, "symbol_type": "ATOM", "boundary_ports": ports[i]} for i in range(3)],
        "edges": [{"source": 2 if swapped else 0, "target": 0 if swapped else 1, "relation": "RelatedTo"},
                  {"source": 0 if swapped else 1, "target": 1 if swapped else 2, "relation": "IsA"}],
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def test_factor_dictionary_roundtrip_and_no_mixed_shapes(tmp_path):
    a = _proto(["[]", "[]", "[]"])
    b = _proto(['[["RelatedTo","in",2]]', "[]", "[]"])
    c = _proto(["[]", "[]", "[]"], swapped=True)
    c["edges"].sort(key=lambda e: (e["source"], e["target"], e["relation"]))
    d = _proto(["[]", "[]", "[]"])
    d["edges"][0]["relation"] = "PartOf"
    src = tmp_path / "graph_types.jsonl"
    _write_jsonl(src, [{"type_id": k, "prototype": v, "accepted_frequency": 1} for k, v in zip("abcd", [a, b, c, d])])
    report = factor_dictionary(src, tmp_path / "out")
    assert report["exact_types"] == 4
    assert report["shared_internal_shapes"] == 2
    assert report["prototype_roundtrip_exact"]
    variants = [json.loads(x) for x in (tmp_path / "out/type_variants.jsonl").read_text().splitlines()]
    shapes = [json.loads(x) for x in (tmp_path / "out/internal_shapes.jsonl").read_text().splitlines()]
    assert [_restore(shapes[v["shape_id"]], v) for v in variants] == [a, b, c, d]
    assert variants[2]["shape_to_prototype"] is not None
    with gzip.open(tmp_path / "out/factorized_combined.jsonl.gz", "rt") as stream:
        assert len(stream.read()) > 0


def test_export_run_figure_edges_and_diagnostics(tmp_path):
    run = tmp_path / "run"
    proto = _proto(["[]", "[]", "[]"])
    _write_jsonl(run / "dictionary/graph_types.jsonl", [{"type_id": "GT_1", "prototype": proto}])
    transition = run / "transition_000_001"
    _write_jsonl(transition / "figure_occurrences.jsonl", [{"dictionary_type_id": "GT_1", "wishart_family_id": "WF_1", "fine_nodes": [0, 1, 2], "prototype_to_fine_nodes": [0, 1, 2]}])
    _write_jsonl(transition / "cluster_dynamics.jsonl", [{"coarse_node": 0, "original_concepts": ["cat", "animal", "thing"], "concept_concat": "animal | cat | thing", "external_flow": 3.0, "external_flow_by_relation": {"RelatedTo": 3.0}, "stationary_mass": 0.6, "exit_probabilities": {"1": 1.0}}])
    np.save(transition / "fine_to_coarse.npy", np.array([0, 0, 0, 1], dtype=np.int64))
    source = run / "level_000"
    target = run / "level_001"
    (source / "relations").mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "dynamic_metrics.json").write_text(json.dumps({"mean_degree": 2.0}))
    (target / "dynamic_metrics.json").write_text(json.dumps({"mean_degree": 1.0, "spectral_gap": 0.4, "mfpt_mean": 3.0, "macro_clustering": 0.5, "intercluster_distance_mean": 1.0}))
    (source / "relations/index.json").write_text(json.dumps({"RelatedTo": "rel.npz"}))
    matrix = sparse.csr_matrix(([1.0, 2.0, 3.0], ([0, 1, 3], [1, 3, 0])), shape=(4, 4))
    sparse.save_npz(source / "relations/rel.npz", matrix)
    np.savez_compressed(target / "dynamic_vectors.npz", betweenness=np.array([0.5, 0.0]), stationary_mass=np.array([0.6, 0.4]), slow_eigenvalues=np.array([0.2]), slow_eigenvectors=np.array([[0.3], [0.7]]), sampled_distances=np.array([[0, 1, 1.0]]))
    (run / "COMPLETED").write_text("complete\n")
    report = export_run(run, tmp_path / "export")
    assert report["transitions"][0]["external_relation_entries"] == 2
    figure = json.loads((tmp_path / "export/transition_000_001/figure_dynamics.jsonl").read_text())
    assert figure["concept_concat"] == "animal | cat | thing"
    assert figure["slow_mode_coordinates_target"] == [0.3]
    assert figure["betweenness_macro_node"] == 0.5
    with gzip.open(tmp_path / "export/transition_000_001/external_connections.jsonl.gz", "rt") as stream:
        edges = [json.loads(x) for x in stream]
    assert {(e["source_fine"], e["target_fine"]) for e in edges} == {(1, 3), (3, 0)}
    assert (tmp_path / "export/COMPLETED").is_file()


def test_export_rejects_incomplete_run(tmp_path):
    with pytest.raises(ValueError, match="COMPLETED"):
        export_run(tmp_path, tmp_path / "out")
