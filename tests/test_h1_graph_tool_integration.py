"""Requires actual graph-tool; a separate mandatory CI job installs it.

This test is intentionally not silently skipped if graph-tool is unavailable.
"""
from __future__ import annotations

import numpy as np

from semmap_haken.hypothesis.inference import fit_one
from semmap_haken.hypothesis.nulls import flat_dc_sbm_null, typed_degrees
from semmap_haken.hypothesis.synthetic_benchmarks import planted_typed_graph
from semmap_haken.hypothesis.runner import H1Config, run_h1


def test_real_layered_inference_and_conditional_flat_sampler():
    import graph_tool.all  # noqa: F401; should be mandatory for this CI job
    synthetic = planted_typed_graph(seed=7, nested=True, n_groups=4, per_group=9)
    g = synthetic.graph
    assert g.edge_count > 0
    fits = {kind: fit_one(g, kind=kind, seed=13, threads=1)
            for kind in ("SBM", "DC-SBM", "nested-DC-SBM")}
    for result in fits.values():
        assert np.isfinite(result["mdl_bits"])
        assert result["partitions"] and len(result["partitions"][0]) == g.vertex_count
    flat_labels = np.asarray(fits["DC-SBM"]["partitions"][0])
    control, meta = flat_dc_sbm_null(g, flat_labels, seed=5)
    assert meta["ensemble"] == "conditional_flat_dc_sbm"
    for left, right in zip(typed_degrees(control), typed_degrees(g)):
        np.testing.assert_array_equal(left, right)


def test_real_graph_tool_end_to_end_artifacts(tmp_path):
    import graph_tool.all  # noqa: F401
    g = planted_typed_graph(seed=10, nested=True, n_groups=4, per_group=9).graph
    report = run_h1(g, tmp_path/"real-h1",
                    config=H1Config(seed=12, restarts=1, null_replicates=1,
                                    workers=1, negatives=4, bootstrap_repetitions=20))
    assert report["fit_counts"]["real"] == 3
    assert report["fit_counts"]["null"] == 6
    assert all(len(info["mdlgain_controls"]) == 1 for info in report["nulls"].values())
    assert report["decision"]["status"] in {"inconclusive", "not_supported"}
    assert (tmp_path/"real-h1"/"COMPLETED").exists()



def test_scientific_command_line_creates_report(tmp_path):
    from semmap_haken.hypothesis.h1_cli import main
    from semmap_haken.conceptnet import Assertion
    import json
    g = planted_typed_graph(seed=10, nested=False, n_groups=4, per_group=9).graph
    data = tmp_path / "input.tsv"
    data.write_text("".join(
        f"{assertion}\t{g.relation_names[int(r)]}\t{g.node_ids[int(u)]}\t"
        f"{g.node_ids[int(v)]}\t" + json.dumps({"weight": float(weight)}) + "\n"
        for assertion, u, v, r, weight in zip(g.assertion_ids, g.source, g.target,
                                               g.relation, g.weight)
    ), encoding="utf-8")
    output = tmp_path / "cli-h1"
    assert main(["--input", str(data), "--output", str(output),
                 "--restarts", "1", "--nulls", "0", "--negatives", "4",
                 "--bootstrap", "20", "--workers", "1"]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["assertions"] == g.edge_count
    assert report["decision"]["status"] in {"not_supported", "inconclusive"}



def test_real_graph_tool_spawn_parallel_restarts(tmp_path):
    """Exercise real multi-process graph-tool fits; not just mocked CPU tasks."""
    import graph_tool.all  # noqa: F401
    graph = planted_typed_graph(seed=17, nested=True, n_groups=4, per_group=9).graph
    report = run_h1(
        graph, tmp_path / "parallel",
        config=H1Config(seed=17, restarts=2, null_replicates=0,
                        workers=2, negatives=4, bootstrap_repetitions=20),
    )
    assert report["fit_counts"]["real"] == 6
    assert report["resource_budget"]["workers"] == 2
    assert len(list((tmp_path / "parallel" / "tasks").glob("real-*.json"))) == 6
    assert (tmp_path / "parallel" / "COMPLETED").exists()
