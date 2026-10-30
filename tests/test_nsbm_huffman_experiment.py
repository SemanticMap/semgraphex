"""Synthetic end-to-end checks for nSBM-guided selection without graph-tool."""
from __future__ import annotations

import json
import numpy as np

from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis.grammar_experiment import run, edge_records
from semmap_haken.grammar_codec import decode_grammar


def sample():
    edges = []
    for base in (0, 4, 8):
        edges.extend([(base, base+1, 0, 1.0),
                      (base+1, base+2, 1, 2.0),
                      (base+2, base, 0, 3.0),
                      (base+1, base+3, 1, 4.0)])
    edges.append((2, 4, 0, 7.0))
    return TypedGraph(
        tuple(f"n{i}" for i in range(12)), ("RelatedTo", "IsA"),
        np.array([e[0] for e in edges], dtype=np.int64),
        np.array([e[1] for e in edges], dtype=np.int64),
        np.array([e[2] for e in edges], dtype=np.int32),
        np.array([e[3] for e in edges], dtype=np.float64),
        tuple(f"a{i}" for i in range(len(edges))),
    )


def test_edge_adapter_keeps_order_multiplicity():
    graph = sample()
    records = edge_records(graph)
    assert len(records) == graph.edge_count
    assert [item.edge_id for item in records] == list(range(graph.edge_count))


def test_smoke_nsbm_plain_shuffle_all_exact(tmp_path):
    graph = sample()
    labels = np.array([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int32)
    result = run(graph, tmp_path / "run", profile="smoke", seed=3,
                 partition=labels, cpu_budget=2, device="cpu",
                 modes=("nsbm", "plain", "shuffled"), max_ego_nodes=5,
                 min_figure_nodes=3, min_support=1)
    assert result["status"] == "complete"
    assert (tmp_path / "run" / "COMPLETED").exists()
    assert set(result["ratios_full_bundle"]) == {"nsbm", "plain", "shuffled"}
    for mode in ("nsbm", "plain", "shuffled"):
        for coding in ("varint", "huffman"):
            n, rows = decode_grammar(tmp_path / "run" / f"grammar_{mode}_{coding}.zip")
            assert n == graph.vertex_count
            assert rows == edge_records(graph)
            assert result["methods"][mode]["coding"][coding]["roundtrip_exact"]
    again = run(graph, tmp_path / "run", profile="smoke", seed=3,
                partition=labels, cpu_budget=2, device="cpu", resume=True,
                modes=("nsbm", "plain", "shuffled"), max_ego_nodes=5,
                min_figure_nodes=3, min_support=1)
    assert again["ratios_full_bundle"] == result["ratios_full_bundle"]


def test_new_cannot_clobber_run_and_manifest_rejects_changed_input(tmp_path):
    graph = sample()
    out = tmp_path / "run"
    run(graph, out, profile="smoke", device="cpu", modes=("plain",),
        discovery=4, cpu_budget=1)
    import pytest
    with pytest.raises(ValueError):
        run(graph, out, profile="smoke", device="cpu", modes=("plain",),
            discovery=4, cpu_budget=1)
    changed = TypedGraph(
        graph.node_ids, graph.relation_names, graph.source, graph.target,
        graph.relation, graph.weight * 2.0, graph.assertion_ids,
    )
    with pytest.raises(ValueError):
        run(changed, out, profile="smoke", device="cpu", modes=("plain",),
            discovery=4, cpu_budget=1, resume=True)
