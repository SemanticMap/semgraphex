"""End-to-end H1 task graph and resume contracts; injected test backend only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semmap_haken.conceptnet import Assertion
from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis.runner import H1Config, run_h1


def fixture_graph() -> TypedGraph:
    records = []
    for i in range(30):
        u, v = f"/c/en/n{i}", f"/c/en/n{(i+2)%30}"
        records.append(Assertion(f"/a/{i}", "/r/IsA" if i%2 else "/r/RelatedTo",
                                 u, v, 1., None, None, None))
    return TypedGraph.from_assertions(records)


def test_checkpoint_resume_and_config_guard(tmp_path: Path):
    graph = fixture_graph()
    calls = []
    def fake_fit(g, *, kind, seed, threads):
        calls.append((kind, seed))
        labels = [i%2 for i in range(g.vertex_count)]
        return {"kind": kind, "seed": seed,
                "mdl_bits": {"SBM": 120., "DC-SBM": 100., "nested-DC-SBM": 80.}[kind],
                "partitions": [labels, [0]*g.vertex_count] if kind == "nested-DC-SBM" else [labels],
                "blocks_by_level": [2, 1] if kind == "nested-DC-SBM" else [2],
                "threads": threads}
    def fake_flat(g, p, *, seed):
        from semmap_haken.hypothesis.nulls import stub_matching_null
        return stub_matching_null(g, seed=seed)
    cfg = H1Config(seed=71, restarts=2, null_replicates=2, workers=1, negatives=6,
                   bootstrap_repetitions=20)
    result = run_h1(graph, tmp_path/"research", config=cfg,
                    fit_fn=fake_fit, flat_sampler=fake_flat)
    assert (tmp_path/"research"/"COMPLETED").exists()
    assert len(calls) == 3 * 2 * (1 + 2*2)
    assert result["fit_counts"]["real"] == 6
    assert result["fit_counts"]["null"] == 24
    assert result["status"] in ("inconclusive", "not_supported", "supported")
    assert set(result["nulls"]) == {"configuration", "conditional_flat_dc_sbm"}
    assert result["decision"]["status"] != "supported"  # only two controls
    first_count = len(calls)
    second = run_h1(graph, tmp_path/"research", config=cfg, resume=True,
                    fit_fn=fake_fit, flat_sampler=fake_flat)
    assert len(calls) == first_count
    assert second["fit_counts"] == result["fit_counts"]
    with pytest.raises(ValueError, match="configuration|manifest"):
        run_h1(graph, tmp_path/"research", config=H1Config(
            seed=72, restarts=2, null_replicates=2, workers=1, negatives=6,
            bootstrap_repetitions=20), resume=True, fit_fn=fake_fit, flat_sampler=fake_flat)


def test_resume_rechecks_graph_identity(tmp_path: Path):
    g = fixture_graph()
    def fake_fit(g, *, kind, seed, threads):
        return {"kind": kind, "seed": seed, "mdl_bits": 10.,
                "partitions": [[i%2 for i in range(g.vertex_count)]],
                "blocks_by_level": [2]}
    def fake_flat(g, p, *, seed):
        from semmap_haken.hypothesis.nulls import stub_matching_null
        return stub_matching_null(g, seed=seed)
    cfg = H1Config(seed=1, restarts=1, null_replicates=0, workers=1, negatives=2,
                   bootstrap_repetitions=20)
    run_h1(g, tmp_path/"run", config=cfg, fit_fn=fake_fit, flat_sampler=fake_flat)
    changed = g.edge_subset(np.arange(g.edge_count) != 0)
    with pytest.raises(ValueError, match="graph|manifest|identity"):
        run_h1(changed, tmp_path/"run", config=cfg, resume=True,
               fit_fn=fake_fit, flat_sampler=fake_flat)



def test_resume_recomputes_only_failed_fit(tmp_path: Path):
    graph = fixture_graph()
    state = {"calls": 0, "fail": True}
    def flaky_backend(g, *, kind, seed, threads):
        state["calls"] += 1
        if state["fail"] and state["calls"] == 3:
            raise RuntimeError("simulated Colab runtime loss")
        return {"kind": kind, "seed": seed,
                "mdl_bits": {"SBM": 10., "DC-SBM": 9., "nested-DC-SBM": 8.}[kind],
                "partitions": [[i%2 for i in range(g.vertex_count)]],
                "blocks_by_level": [2]}
    config = H1Config(seed=1, restarts=2, null_replicates=0, workers=1,
                      negatives=3, bootstrap_repetitions=20)
    path = tmp_path/"interrupted"
    with pytest.raises(RuntimeError, match="simulated"):
        run_h1(graph, path, config=config, fit_fn=flaky_backend)
    assert (path / "manifest.json").exists()
    assert not (path / "COMPLETED").exists()
    assert len(list((path / "tasks").glob("*.json"))) == 2
    state["fail"] = False
    report = run_h1(graph, path, config=config, resume=True, fit_fn=flaky_backend)
    assert report["fit_counts"]["real"] == 6
    assert state["calls"] == 7  # two cached, one failed, four retried
    assert (path / "COMPLETED").exists()
