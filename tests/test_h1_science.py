"""Scientific H1 regression tests. Add functionality only after observing RED."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semmap_haken.conceptnet import Assertion
from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis.splits import split_graph
from semmap_haken.hypothesis.predict import score_candidates, ranking_metrics
from semmap_haken.hypothesis.nulls import stub_matching_null, typed_degrees
from semmap_haken.hypothesis.statistics import empirical_upper_tail, holm_adjust, h1_verdict
from semmap_haken.hypothesis.synthetic_benchmarks import planted_typed_graph


def small_graph() -> TypedGraph:
    raw = [
        ("a","b","R"), ("a","b","R"), ("b","a","R"),  # same undirected group
        ("a","c","R"), ("c","a","R"),
        ("d","e","R"), ("e","d","R"),
        ("c","d","Q"), ("d","c","Q"),
        ("f","g","Q"), ("g","f","Q"),
        ("a","d","Q"), ("b","d","Q"), ("a","e","Q"), ("c","f","Q"),
        ("g","a","R"), ("g","b","R"), ("b","f","Q"),
    ]
    assertions = [
        Assertion(f"/a/{i}", f"/r/{r}", u, v, 1.0, None, None, None)
        for i, (u, v, r) in enumerate(raw)
    ]
    return TypedGraph.from_assertions(assertions)


def test_grouped_split_has_no_assertion_or_reverse_leakage():
    graph = small_graph()
    result = split_graph(graph, seed=41, train_fraction=.6, validation_fraction=.2)
    assert all(x.node_ids == graph.node_ids for x in (result.train, result.validation, result.test))
    key_sets = []
    for subset in (result.train, result.validation, result.test):
        key_sets.append({
            (min(int(u),int(v)), max(int(u),int(v)), int(r))
            for u,v,r in zip(subset.source, subset.target, subset.relation)
        })
    assert not key_sets[0] & key_sets[1]
    assert not key_sets[0] & key_sets[2]
    assert not key_sets[1] & key_sets[2]
    assert sum(x.edge_count for x in (result.train, result.validation, result.test)) == graph.edge_count
    assert all(x.edge_count > 0 for x in (result.train, result.validation, result.test))
    other = split_graph(graph, seed=41, train_fraction=.6, validation_fraction=.2)
    assert other.test.assertion_ids == result.test.assertion_ids


def test_prediction_scores_and_ranking_are_train_only():
    graph = small_graph()
    partition = np.array([0 if n in {"a","b","c"} else 1 for n in graph.node_ids])
    idx = {n: i for i,n in enumerate(graph.node_ids)}
    relation = graph.relation_names.index("/r/R")
    before = score_candidates(graph, partition, idx["a"], relation,
                              np.array([idx["b"],idx["g"]]), degree_corrected=True)
    assert before.shape == (2,)
    assert np.all(np.isfinite(before))
    assert np.all(before >= 0)
    # Change the candidate set, never the training data, to compute ranking.
    split = split_graph(graph, seed=3, train_fraction=.65, validation_fraction=.15)
    metric = ranking_metrics(split.train, split.test, partition, seed=5, negatives=4,
                             degree_corrected=True, known=graph)
    assert 0 <= metric["mrr"] <= 1
    assert 0 <= metric["hits_at_10"] <= 1
    assert metric["queries"] > 0
    assert metric["score_kind"] == "smoothed_block_propensity"


def test_fast_configuration_null_has_exact_typed_directed_degrees():
    graph = small_graph()
    a, da = stub_matching_null(graph, seed=7)
    b, db = stub_matching_null(graph, seed=7)
    for d0, d1 in zip(typed_degrees(graph), typed_degrees(a)):
        np.testing.assert_array_equal(d0, d1)
    assert np.array_equal(a.target, b.target)
    assert all(u.startswith("null:") for u in a.assertion_ids)
    assert da["ensemble"] == "directed_typed_stub_matching"
    assert da["self_loops_allowed"] is True
    assert db == da


def test_mc_pvalue_holm_and_verdict_guards():
    d = empirical_upper_tail(6.0, [1, 2, 3, 4, 5])
    assert d["p_value"] == pytest.approx(1/6)
    assert d["exceedances"] == 0
    assert holm_adjust([0.01, 0.03, 0.20]) == pytest.approx([0.03, 0.06, 0.20])
    assert h1_verdict(mdl_gain=30, levels=3, p_configuration=.001, p_flat=.004,
                      paired_mrr_delta=.01, paired_ci_low=-.001,
                      noninferiority_margin=.01)["status"] == "supported"
    assert h1_verdict(mdl_gain=30, levels=1, p_configuration=.001, p_flat=.004,
                      paired_mrr_delta=.01, paired_ci_low=-.001,
                      noninferiority_margin=.01)["status"] == "not_supported"
    assert h1_verdict(mdl_gain=30, levels=3, p_configuration=None, p_flat=None,
                      paired_mrr_delta=.01, paired_ci_low=-.001,
                      noninferiority_margin=.01)["status"] == "inconclusive"


def test_synthetic_truth_flat_and_nested_cases():
    flat = planted_typed_graph(seed=7, nested=False, n_groups=4, per_group=15)
    nested = planted_typed_graph(seed=7, nested=True, n_groups=4, per_group=15)
    assert flat.graph.vertex_count == nested.graph.vertex_count == 60
    assert flat.graph.edge_count > 0 and nested.graph.edge_count > 0
    assert flat.truth["nested"] is False
    assert nested.truth["nested"] is True
    assert len(nested.truth["coarse"]) == 60
    assert len(set(nested.truth["coarse"])) < len(set(nested.truth["fine"]))



def test_negative_candidates_do_not_enumerate_entire_large_vocabulary(monkeypatch):
    from semmap_haken.hypothesis import predict
    def forbid_setdiff(*args, **kwargs):
        raise AssertionError("O(number-of-nodes) set difference per query is forbidden")
    monkeypatch.setattr(predict.np, "setdiff1d", forbid_setdiff)
    rng = np.random.default_rng(17)
    choice = predict._negative_candidates(rng, 100_000, {0, 2, 3, 4, 5}, count=100)
    assert len(choice) == 100
    assert len(set(choice.tolist())) == 100
    assert not set(choice.tolist()) & {0, 2, 3, 4, 5}


def test_paired_bootstrap_groups_queries_by_source():
    from semmap_haken.hypothesis.statistics import paired_mrr_interval
    a = [1, .9, .9, .4, .3, .5]
    b = [.9, .7, .8, .5, .3, .4]
    clusters = [0, 0, 0, 1, 1, 2]
    first = paired_mrr_interval(a, b, seed=9, repetitions=100, clusters=clusters)
    second = paired_mrr_interval(a, b, seed=9, repetitions=100, clusters=clusters)
    assert first == second
    assert first["clusters"] == 3
    assert first["ci_low"] <= first["mean_difference"] <= first["ci_high"]
