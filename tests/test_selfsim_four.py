"""Four scientific self-similarity tests and guarded negative controls."""
from __future__ import annotations

import json
import numpy as np
import pytest

from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis.selfsim_metrics import (
    TRIAD_CLASSES, TRIAD_LOOKUP, box_covering, dynamics, graphon,
    motif_profile, structural, support, typed_counts, validate_levels,
)
from semmap_haken.hypothesis.selfsim_experiment import (
    nested_degree_stratified_permutation, typed_degree_null, four_methods,
    load_levels, run, scores,
)


def fixture(n=24):
    src, dst, rel = [], [], []
    for i in range(n):
        for u, v, r in ((i, (i+1)%n, 0), (i, (i+2)%n, 1),
                        (i, (i+8)%n, 0), (i, (i//4)*4, 1)):
            src.append(u); dst.append(v); rel.append(r)
    graph = TypedGraph(
        tuple(f"/c/en/n{i}" for i in range(n)), ("RelatedTo", "IsA"),
        np.array(src,dtype=np.int64),np.array(dst,dtype=np.int64),
        np.array(rel,dtype=np.int32),
        np.ones(len(src),dtype=np.float64),
        tuple(f"/a/{i}" for i in range(len(src)))
    )
    levels = [
        np.arange(n,dtype=np.int32)//3,
        np.arange(n,dtype=np.int32)//6,
        np.arange(n,dtype=np.int32)//12,
    ]
    return graph,levels


def test_nested_contract_and_permutation_preserves_masses():
    g, levels = fixture()
    good=validate_levels(levels,g.vertex_count)
    assert [len(set(p)) for p in good]==[8,4,2]
    shuffled=nested_degree_stratified_permutation(g, good, 100)
    for before,after in zip(good,shuffled):
        assert sorted(np.bincount(before))==sorted(np.bincount(after))
    with pytest.raises(ValueError,match="not nested"):
        validate_levels([levels[1],levels[0]],g.vertex_count)
    with pytest.raises(ValueError,match="every original"):
        validate_levels([np.arange(10)],g.vertex_count)


def test_relation_stub_matching_keeps_typed_directed_degrees():
    g, _=fixture()
    h=typed_degree_null(g, 14)
    assert h.vertex_count == g.vertex_count
    assert h.edge_count == g.edge_count
    assert np.array_equal(g.source,h.source)
    for r in range(len(g.relation_names)):
        mask = g.relation == r
        assert np.array_equal(np.bincount(g.target[mask],minlength=g.vertex_count),
                              np.bincount(h.target[mask],minlength=h.vertex_count))
        assert np.array_equal(np.bincount(g.source[mask],minlength=g.vertex_count),
                              np.bincount(h.source[mask],minlength=h.vertex_count))


def test_all_four_metrics_on_three_nested_scales():
    g,levels=fixture()
    res = four_methods(g,validate_levels(levels,g.vertex_count),
                       seed=9,motif_samples=200,pair_samples=700,
                       box_repeats=2,radii=(0,1,2,3),slow_modes=3)
    assert set(res)=={"structural","graphon","haken","boxes"}
    assert [x["blocks"] for x in res["structural"]["levels"]]==[24,8,4,2]
    assert [x["blocks"] for x in res["graphon"]["levels"]]==[8,4,2]
    assert res["structural"]["levels"][0]["motifs"]["sample_count"]==200
    assert len(res["structural"]["adjacent_motif_js"])==2
    assert res["structural"]["levels"][-1]["motifs"]["sample_count"]==0
    # Two blocks cannot contain a 3-node motif; undefined != zero similarity.
    assert len(res["graphon"]["adjacent_kernel"])==2
    assert 0<=res["graphon"]["levels"][0]["holdout_auc_conditional"]<=1
    for x in res["haken"]["levels"]:
        assert 0<=x["slow_subspace_overlap"]<=1+1e-10
        assert x["compared_nontrivial_modes"]<=x["blocks"]-1
    b=res["boxes"]
    assert [x["box_count"] for x in b["rows"] if x["radius"]==0]==[24,24]
    assert all(x["Z_q"]["1.0"]==pytest.approx(1) for x in b["rows"])
    assert set(scores(res))=={
        "motif_similarity_negative_js","kernel_similarity_negative_js",
        "haken_slow_overlap","box_power_over_exp_delta_aic"}


def test_nsbm_multiscale_resume_and_corruption(tmp_path,monkeypatch):
    graph,levels=fixture()
    import semmap_haken.hypothesis.selfsim_experiment as ex
    monkeypatch.setattr(ex,"_git_revision",lambda: "test-revision")
    args=dict(seed=5,null_repeats=1,jobs=1,motif_samples=100,
              pair_samples=200,box_repeats=1,radii=(0,1,2,3),slow_modes=2)
    out=tmp_path/"run"
    report=run(graph,levels,out,**args)
    assert (out/"COMPLETED").is_file()
    assert report["block_counts"]==[8,4,2]
    assert report["null_repeats_per_ensemble"]==1
    resumed=run(graph,levels,out,resume=True,**args)
    assert resumed["identity"]==report["identity"]
    (out/"null_typed_degree_rewire_000.json").write_text("{broken")
    (out/"COMPLETED").unlink()
    repaired=run(graph,levels,out,resume=True,**args)
    assert repaired["status"]=="complete"
    assert json.loads((out/"null_typed_degree_rewire_000.json").read_text())["sha256"]
    with pytest.raises(ValueError, match="RESUME"):
        run(graph,levels,out,resume=True,**{**args,"seed":6})


def test_load_partition_npz_and_refuse_mismatch(tmp_path):
    g,levels=fixture()
    p=tmp_path/"levels.npz"
    np.savez_compressed(p,**{f"level_{i:03d}":row
                             for i,row in enumerate(levels)})
    assert [len(set(x)) for x in load_levels(p,g.vertex_count)]==[8,4,2]
    with pytest.raises(ValueError):
        load_levels(p,g.vertex_count+1)


def test_motif_isomorphism_invariant_under_vertex_relabeling():
    from itertools import permutations
    # All 64 three-node directed adjacency patterns canonicalize the same
    # way under all six relabelings.
    arcs=((0,1),(1,0),(0,2),(2,0),(1,2),(2,1))
    for mask in range(64):
        for p in permutations(range(3)):
            alternate=sum(((mask>>arcs.index((p[i],p[j])))&1)<<k
                          for k,(i,j) in enumerate(arcs))
            assert TRIAD_LOOKUP[alternate]==TRIAD_LOOKUP[mask]


def test_zero_wedges_and_zero_rejected_dense_nxn(monkeypatch):
    graph=TypedGraph(("a","b","c"),("r",),
                     np.array([0]),np.array([1]),
                     np.array([0],dtype=np.int32),
                     np.array([1.]),("id",))
    p=motif_profile(support(graph),seed=1,samples=10)
    assert p["sample_count"]==0
    # 10k-node tensors should never be built on the original scale;
    # only one-level block quotient tensors are allocated.
    graph,levels=fixture()
    rows=structural(graph,levels,seed=1,samples=20)["levels"]
    assert rows[0]["blocks"]==graph.vertex_count


def test_cli_on_prepared_h1_fixture(tmp_path,monkeypatch):
    import semmap_haken.hypothesis.selfsim_experiment as ex
    from semmap_haken.hypothesis.runner import _save_graph, _sha_bytes
    graph,levels=fixture()
    input_dir=tmp_path/"fit_data"
    _save_graph(graph,input_dir)
    partition_file=tmp_path/"partitions.npz"
    np.savez_compressed(partition_file,
                        **{f"level_{i:03d}":p for i,p in enumerate(levels)})
    source=tmp_path/"source.json"
    source.write_text(json.dumps({"graph_sha256":_sha_bytes(graph)}))
    monkeypatch.setattr(ex,"_git_revision",lambda:"test-revision")
    output=tmp_path/"cli"
    args=["--graph-dir",str(input_dir),"--partitions",str(partition_file),
          "--source-manifest",str(source),"--output",str(output),
          "--null-repeats","0","--jobs","1","--motif-samples","20",
          "--pair-samples","50","--box-repeats","1","--box-radii","0,1,2,3"]
    assert ex.main(args)==0
    assert (output/"COMPLETED").exists()
    # A proper second CLI call requires identical mode semantics:
    resumed=args.copy()
    resumed[resumed.index("--null-repeats")+1]="0"
    resumed.extend(["--mode","RESUME"])
    assert ex.main(resumed)==0
