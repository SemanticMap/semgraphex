"""Contract tests for the ConceptNet 10k typed H1 release (first fail, then implement)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semmap_haken.conceptnet import Assertion
from semmap_haken.hypothesis.synthetic_benchmarks import planted_typed_graph
from semmap_haken.hypothesis.hierarchy_validation import hierarchy_recovery, hierarchy_restart_stability
from semmap_haken.hypothesis.conceptnet_10k import prepare_conceptnet_sample
from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis.predict import ranking_metrics


def test_synthetic_relation_labels_do_not_encode_vertex_parity():
    sample = planted_typed_graph(seed=123, nested=True, n_groups=4, per_group=20)
    graph = sample.graph
    s, t = graph.source, graph.target
    parity = (s + t) % 2
    # Old generator was an exact parity oracle; relation identity must be a
    # stochastic label independent of artificial node indices.
    match = np.mean((graph.relation == 0) == (parity == 1))
    assert 0.20 < match < 0.80
    again = planted_typed_graph(seed=123, nested=True, n_groups=4, per_group=20)
    np.testing.assert_array_equal(graph.relation, again.graph.relation)
    assert sample.truth["relation_assignment"] == "seeded_independent_draw"


def test_recovery_matches_true_fine_and_coarse_levels_despite_label_permutations():
    fine = [0,0,1,1,2,2,3,3]
    coarse = [0,0,0,0,1,1,1,1]
    levels = [[20,20,10,10,40,40,30,30],
              [91,91,91,91,4,4,4,4], [0]*8]
    result = hierarchy_recovery(levels, {"fine":fine,"coarse":coarse})
    assert result["fine"]["ari"] == pytest.approx(1.0)
    assert result["coarse"]["ari"] == pytest.approx(1.0)
    assert result["fine"]["level"] == 0
    assert result["coarse"]["level"] == 1
    assert result["levels_evaluated"] == 2


def test_recovery_and_restart_metrics_detect_wrong_levels():
    fine = [0,0,1,1,2,2,3,3]
    coarse = [0,0,0,0,1,1,1,1]
    wrong = [[0,1,0,1,0,1,0,1], [0]*8]
    poor = hierarchy_recovery(wrong, {"fine":fine,"coarse":coarse})
    assert poor["fine"]["ari"] < .5
    assert poor["coarse"]["ari"] < .5
    consistency = hierarchy_restart_stability([wrong, [coarse, [0]*8]])
    assert consistency["pair_count"] == 1
    assert consistency["finest_ari_mean"] < .5


def _write_fixture(path: Path, n: int = 48):
    """Connected English ring with three relation labels plus foreign edges."""
    rows = []
    labels = ("IsA", "PartOf", "UsedFor")
    for i in range(n):
        u=f"/c/en/w{i:03d}"
        v=f"/c/en/w{(i+1)%n:03d}"
        rows.append([f"/a/{i}",f"/r/{labels[i%3]}",u,v,
                     json.dumps({"weight":1.0, "dataset":"/d/test"})])
    for i in range(5):
        rows.append([f"/a/es/{i}","/r/IsA",f"/c/es/e{i}",f"/c/es/e{i+1}",
                     '{"weight":1.0}'])
    rows.append(["/a/excluded", "/r/Antonym", "/c/en/w000", "/c/en/w040",
                 '{"weight":1.0}'])
    path.write_text("".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def test_conceptnet_10k_sampler_keeps_exact_nodes_provenance_and_relation_layers(tmp_path: Path):
    source=tmp_path/"source.tsv"
    _write_fixture(source)
    destination=tmp_path/"sample.tsv"
    summary=prepare_conceptnet_sample(
        source, destination, nodes=32, seed=7, language="en",
        relations=("IsA","PartOf","UsedFor"), max_assertions=200,
    )
    from semmap_haken.conceptnet import stream_assertions
    graph=TypedGraph.from_assertions(stream_assertions(destination))
    assert graph.vertex_count == 32
    assert set(graph.relation_names) == {"/r/IsA","/r/PartOf","/r/UsedFor"}
    assert all(id.startswith("/a/") for id in graph.assertion_ids)
    assert summary["selected_nodes"] == 32
    assert summary["selected_assertions"] == graph.edge_count
    assert sum(summary["by_relation"].values()) == graph.edge_count
    assert all(count > 0 for count in summary["by_relation"].values())
    assert summary["selection_policy"].startswith("relation_stratified")
    destination2=tmp_path/"sample2.tsv"
    again=prepare_conceptnet_sample(
        source, destination2, nodes=32, seed=7, language="en",
        relations=("IsA","PartOf","UsedFor"), max_assertions=200,
    )
    assert destination.read_bytes() == destination2.read_bytes()
    assert again["sha256"] == summary["sha256"]


def test_sample_capacity_and_missing_relation_fail_loudly(tmp_path: Path):
    source=tmp_path/"source.tsv"
    _write_fixture(source)
    with pytest.raises(ValueError,match="limit|assertions"):
        prepare_conceptnet_sample(source,tmp_path/"out.tsv",nodes=32,
                                  relations=("IsA","PartOf"),max_assertions=4)
    with pytest.raises(ValueError,match="available|requested|vertices"):
        prepare_conceptnet_sample(source,tmp_path/"out2.tsv",nodes=100,
                                  relations=("IsA",),max_assertions=200)
    with pytest.raises(ValueError,match="relation"):
        prepare_conceptnet_sample(source,tmp_path/"out3.tsv",nodes=32,
                                  relations=("IsA","HasPrerequisite"),max_assertions=200)


def test_hits_at_10_reports_inapplicable_case_and_relation_breakdown():
    assertions=[]
    for i in range(40):
        assertions.append(Assertion(f"/a/{i}","/r/IsA" if i%2 else "/r/PartOf",
                                    f"/c/en/n{i:03d}",
                                    f"/c/en/n{(i+1)%40:03d}",1.,None,None,None))
    g=TypedGraph.from_assertions(assertions)
    from semmap_haken.hypothesis.splits import split_graph
    s=split_graph(g,seed=3,train_fraction=.6,validation_fraction=.2)
    labels=np.arange(g.vertex_count)//5
    short=ranking_metrics(s.train,s.test,labels,seed=9,negatives=4,known=g,degree_corrected=True)
    assert short["mrr"] is not None
    assert short["hits_at_10"] is None
    assert short["hits_at_10_queries"] == 0
    long=ranking_metrics(s.train,s.test,labels,seed=9,negatives=25,known=g,degree_corrected=True)
    assert long["hits_at_10_queries"] > 0
    assert 0 <= long["hits_at_10"] <= 1
    assert set(long["by_relation"]) == {"IsA","PartOf"}
    assert sum(d["queries"] for d in long["by_relation"].values()) == long["queries"]



def test_sampler_handles_actual_10k_vertex_budget(tmp_path: Path):
    """10k genuine selected nodes: scalable, not merely a 32-node smoke."""
    source=tmp_path / "ring.tsv"
    relations=("RelatedTo", "IsA", "PartOf", "UsedFor")
    with source.open("w", encoding="utf-8") as stream:
        for i in range(10_500):
            stream.write(
                f"/a/ring/{i}\t/r/{relations[i%len(relations)]}\t"
                f"/c/en/r{i:05d}\t/c/en/r{(i+1)%10500:05d}\t"
                '{"weight":1.0}\n'
            )
    dest=tmp_path/"cn10k.tsv"
    result=prepare_conceptnet_sample(
        source, dest, nodes=10_000, seed=18,
        relations=relations, max_assertions=20_000)
    assert result["selected_nodes"] == 10_000
    assert sum(result["by_relation"].values()) == result["selected_assertions"]
    assert all(result["by_relation"][r] > 0 for r in relations)
    assert len({uri for a in __import__("semmap_haken.conceptnet",fromlist=["stream_assertions"]).stream_assertions(dest)
                for uri in (a.start_uri,a.end_uri)}) == 10_000
