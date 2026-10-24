from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semmap_haken.conceptnet import Assertion
from semmap_haken.hypothesis.graph import TypedGraph, edge_dropout, p_sample
from semmap_haken.hypothesis.nulls import degree_preserving_null, typed_degrees
from semmap_haken.hypothesis.projection import estimate_block_kernel
from semmap_haken.hypothesis.inference import fit_models
from semmap_haken.hypothesis.cli import main


def example_graph():
    edges = [("cat", "animal", "IsA"), ("dog", "animal", "IsA"),
             ("bird", "animal", "IsA"), ("cat", "pet", "RelatedTo"),
             ("dog", "pet", "RelatedTo"), ("bird", "sky", "RelatedTo"),
             ("pet", "home", "RelatedTo"), ("sky", "home", "RelatedTo"),
             ("lone-a", "lone-b", "RelatedTo"), ("lone-a", "lone-b", "RelatedTo")]
    statements = [Assertion(f"/a/{i}", f"/r/{r}", a, b, float(i+1), None, None, None)
                  for i, (a,b,r) in enumerate(edges)]
    return TypedGraph.from_assertions(statements)


def test_identity_direction_types_parallel_edges_and_isolated_pair():
    graph = example_graph()
    assert graph.edge_count == 10
    assert graph.assertion_ids[0] == "/a/0"
    assert len(set(graph.relation_names)) == 2
    ia, ib = graph.node_ids.index("lone-a"), graph.node_ids.index("lone-b")
    assert sum((graph.source == ia) & (graph.target == ib)) == 2
    assert not np.any((graph.source == ib) & (graph.target == ia))


def test_p_sampling_is_induced_then_drops_isolates():
    graph = example_graph()
    full = p_sample(graph, p=1.0, seed=17)
    assert full.edge_count == graph.edge_count
    empty = p_sample(graph, p=0.0, seed=17)
    assert empty.edge_count == 0 and empty.vertex_count == 0
    sampled = p_sample(graph, p=0.50, seed=17)
    assert set(sampled.assertion_ids).issubset(set(graph.assertion_ids))
    assert sampled.vertex_count == len(set(sampled.source.tolist()+sampled.target.tolist()))
    for a, b in zip(sampled.source, sampled.target):
        assert 0 <= a < sampled.vertex_count and 0 <= b < sampled.vertex_count


def test_edge_dropout_retains_vertex_ids_and_provenance():
    graph = example_graph()
    left = edge_dropout(graph, keep_probability=0.4, seed=7)
    right = edge_dropout(graph, keep_probability=0.4, seed=7)
    assert left.vertex_count == graph.vertex_count
    assert left.assertion_ids == right.assertion_ids
    assert set(left.assertion_ids).issubset(graph.assertion_ids)


def test_null_preserves_per_relation_in_and_out_degree():
    graph = example_graph()
    shuffled, diag = degree_preserving_null(graph, seed=9, swaps_per_edge=8)
    out1, in1 = typed_degrees(graph)
    out2, in2 = typed_degrees(shuffled)
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(in1, in2)
    np.testing.assert_array_equal(shuffled.relation, graph.relation)
    assert diag["attempted"] >= diag["accepted"]
    assert all(s.startswith("null:") for s in shuffled.assertion_ids)
    assert np.array_equal(shuffled.source, graph.source)
    assert shuffled.edge_count == graph.edge_count


def test_projection_is_directional_typed_and_mass_aware():
    graph = example_graph()
    labels = np.array([0 if node in {"cat", "dog", "bird", "lone-a"} else 1
                       for node in graph.node_ids])
    result = estimate_block_kernel(graph, labels, device="cpu", batch_size=3)
    assert result["backend"] == "cpu"
    assert result["block_mass"] == [4.0, 5.0]
    assert sum(x["edge_count"] for x in result["entries"]) == graph.edge_count
    assert sum(x["weight_sum"] for x in result["entries"]) == pytest.approx(sum(graph.weight))
    isa = next(x for x in result["entries"] if x["relation"] == "IsA")
    assert isa["source_block"] == 0 and isa["target_block"] == 1
    assert isa["edge_count"] == 3
    assert isa["exposure_pairs"] == 20
    assert isa["edge_intensity"] == pytest.approx(3/20)


def test_projection_rejects_missing_explicit_cuda():
    graph = example_graph()
    labels = np.zeros(graph.vertex_count, dtype=np.int64)
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA is available")
    except Exception:
        pass
    with pytest.raises(RuntimeError, match="CUDA"):
        estimate_block_kernel(graph, labels, device="cuda")


def test_pilot_writes_reproducible_artifacts(tmp_path: Path):
    rows = [f"/a/{i}\t/r/{r}\t{a}\t{b}\t" + json.dumps({"weight": float(i+1)})
            for i,(a,b,r) in enumerate([("/c/en/a", "/c/en/b", "IsA"),
                                       ("/c/en/a", "/c/en/c", "RelatedTo"),
                                       ("/c/en/d", "/c/en/e", "RelatedTo")])]
    source = tmp_path/"tiny.tsv"
    source.write_text("\n".join(rows)+"\n", encoding="utf-8")
    output = tmp_path/"run"
    assert main(["pilot", "--input", str(source), "--output", str(output),
                 "--nulls", "2", "--workers", "2", "--device", "cpu"]) == 0
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    assert summary["graph"]["edges"] == 3
    assert summary["nulls"]["count"] == 2
    assert summary["projection"]["backend"] == "cpu"
    assert (output/"COMPLETED").is_file()
    assert (output/"typed_graph.npz").is_file()


def test_inference_dependency_message_or_results():
    graph = example_graph()
    try:
        import graph_tool.all  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="graph-tool"):
            fit_models(graph, seed=0)
    else:
        result = fit_models(graph, seed=0, threads=1)
        assert set(result["mdl_bits"]) == {"SBM", "DC-SBM", "nested-DC-SBM"}
        assert all(np.isfinite(list(result["mdl_bits"].values())))
        assert all(len(level) == graph.vertex_count for level in result["partitions"])
