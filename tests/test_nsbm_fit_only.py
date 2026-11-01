"""nSBM-only ConceptNet-100k preparation: no grammar / correct checkpoint lineage."""
from __future__ import annotations

import json
import numpy as np
import pytest

from semmap_haken.hypothesis.nsbm_fit_only import run
from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis import nsbm_fit_only as fitting
from semmap_haken.hypothesis.runner import _sha_bytes, _save_graph


def _sample(n=20):
    # Tiny but typed and directed; full file has all n original vertices.
    from semmap_haken.conceptnet import Assertion
    from semmap_haken.hypothesis.graph import TypedGraph
    return TypedGraph.from_assertions([
        Assertion(f"/a/{i}", "/r/RelatedTo" if i%2 else "/r/IsA",
                  f"/c/en/n{i}", f"/c/en/n{(i+1)%n}", 1.,
                  None,None,None) for i in range(n)
    ])


def _write_assertions(path, graph):
    with path.open("w",encoding="utf-8") as f:
        for i in range(graph.edge_count):
            r=graph.relation_names[int(graph.relation[i])]
            u,v=graph.node_ids[int(graph.source[i])],graph.node_ids[int(graph.target[i])]
            f.write(f"{graph.assertion_ids[i]}\t{r}\t{u}\t{v}\t"+
                    json.dumps({"weight":float(graph.weight[i])})+"\n")


def test_nsbm_only_fit_resume_and_corrupt_partition(tmp_path,monkeypatch):
    source=tmp_path/"graph.tsv"
    g=_sample(20)
    _write_assertions(source,g)
    calls=[]
    def fake_fit(graph,output,*,seed,restarts,cpu_budget):
        calls.append(seed)
        _save_graph(graph,output/"fit_data")
        labels=[np.arange(20)//2,np.arange(20)//5,np.arange(20)//10]
        np.savez_compressed(output/"nsbm_partitions.npz",
            **{f"level_{i:03d}":x for i,x in enumerate(labels)})
        return labels,{"best_mdl_bits":3.0,"selected_seed":seed}
    monkeypatch.setattr(fitting,"_git_revision",lambda:"pinned-test")
    monkeypatch.setattr(fitting,"_fit_partition",fake_fit)
    out=tmp_path/"fit_run"
    r=run(source,out,vertices=20,restarts=2,seed=44,cpu_budget=1)
    assert r["block_counts"]==[10,4,2]
    assert (out/"COMPLETED").exists()
    assert len(calls)==1
    again=run(source,out,vertices=20,restarts=2,seed=44,cpu_budget=1,
              resume=True)
    assert again==r and len(calls)==1
    arr=bytearray((out/"nsbm_partitions.npz").read_bytes())
    arr[31]^=1
    (out/"nsbm_partitions.npz").write_bytes(arr)
    repaired=run(source,out,vertices=20,restarts=2,seed=44,cpu_budget=1,
                 resume=True)
    assert len(calls)==2 and repaired["block_counts"]==[10,4,2]
    assert (out/"COMPLETED").exists()
    with pytest.raises(ValueError,match="mismatch"):
        run(source,out,vertices=20,restarts=2,seed=45,cpu_budget=1,
            resume=True)


def test_nsbm_only_refuses_wrong_node_count(tmp_path,monkeypatch):
    source=tmp_path/"graph.tsv"
    _write_assertions(source,_sample(20))
    monkeypatch.setattr(fitting,"_git_revision",lambda:"pinned-test")
    with pytest.raises(ValueError, match="exactly"):
        run(source,tmp_path/"out",vertices=100000,restarts=1,cpu_budget=1)
