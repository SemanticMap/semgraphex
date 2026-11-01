"""Regression tests for incomplete Drive → Colab nSBM staging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semmap_haken.conceptnet import Assertion
from semmap_haken.hypothesis.graph import TypedGraph
from semmap_haken.hypothesis.runner import _save_graph, _sha_bytes
from semmap_haken.hypothesis.scan_checkpoint import sha256_file, atomic_json
from semmap_haken.hypothesis.selfsim_colab_io import (
    FIT_FILES, ensure_verified_fit_snapshot, stage_fit_snapshot,
)


def _fixture(remote: Path) -> None:
    n = 20
    graph = TypedGraph.from_assertions([
        Assertion(f"/a/{i}", "/r/RelatedTo", f"/c/en/n{i}",
                  f"/c/en/n{(i+1)%n}", 1., None, None, None)
        for i in range(n)
    ])
    remote.mkdir(parents=True, exist_ok=True)
    _save_graph(graph, remote/"fit_data")
    p = [np.arange(n, dtype=np.int32)//2,
         np.arange(n, dtype=np.int32)//4,
         np.arange(n, dtype=np.int32)//8]
    np.savez_compressed(remote/"nsbm_partitions.npz",
                        **{f"level_{i:03d}":v for i, v in enumerate(p)})
    identity = {"graph_sha256":_sha_bytes(graph),"source_tsv_sha256":"example"}
    summary = {"identity":identity, "counts":{"vertices":n},
               "block_counts":[10,5,3],
               "partitions_sha256":sha256_file(remote/"nsbm_partitions.npz")}
    atomic_json(remote/"manifest.json",identity)
    atomic_json(remote/"fit_summary.json",summary)
    (remote/"COMPLETED").write_text("complete\n")


def test_restages_missing_ids_even_with_completed_marker(tmp_path):
    remote=tmp_path/"drive"/"fit"
    local=tmp_path/"content"/"fit"
    _fixture(remote)
    result=stage_fit_snapshot(remote,local,expected_vertices=20)
    assert result["ready"] and result["blocks"]==[10,5,3]
    assert (local/"fit_data"/"ids.json").is_file()
    (local/"fit_data"/"ids.json").unlink()
    assert (local/"COMPLETED").is_file()
    result2=ensure_verified_fit_snapshot(remote,local,expected_vertices=20)
    assert result2["ready"]
    assert (local/"fit_data"/"ids.json").read_bytes()==(
        remote/"fit_data"/"ids.json").read_bytes()


def test_repairs_corrupt_local_without_retraining(tmp_path):
    remote=tmp_path/"drive"/"fit"
    local=tmp_path/"content"/"fit"
    _fixture(remote)
    stage_fit_snapshot(remote,local,expected_vertices=20)
    (local/"nsbm_partitions.npz").write_bytes(b"broken")
    (local/"fit_data"/"source.npy").write_bytes(b"truncated")
    row=stage_fit_snapshot(remote,local,expected_vertices=20)
    assert row["ready"]
    for name in ("nsbm_partitions.npz","fit_data/source.npy"):
        assert sha256_file(local/name)==sha256_file(remote/name)


def test_stale_marker_does_not_mean_complete(tmp_path):
    remote=tmp_path/"drive"/"fit"
    local=tmp_path/"content"/"fit"
    _fixture(remote)
    stage_fit_snapshot(remote,local,expected_vertices=20)
    # The durable Drive snapshot is incomplete even if old local files exist.
    (remote/"fit_data"/"ids.json").unlink()
    result=stage_fit_snapshot(remote,local,expected_vertices=20)
    assert not result["ready"]
    assert "fit_data/ids.json" in result["missing"]
    assert not (local/"COMPLETED").exists()
    with pytest.raises(RuntimeError,match="ids.json"):
        ensure_verified_fit_snapshot(remote,local,expected_vertices=20)


def test_corrupt_published_manifest_fails_loudly(tmp_path):
    remote=tmp_path/"drive"/"fit"
    _fixture(remote)
    (remote/"manifest.json").write_text('{"graph_sha256":"changed"}')
    with pytest.raises(ValueError,match="manifest identity differs"):
        stage_fit_snapshot(remote,tmp_path/"content",expected_vertices=20)


def test_no_marker_is_resume_not_complete(tmp_path):
    remote=tmp_path/"drive"/"fit"
    _fixture(remote)
    (remote/"COMPLETED").unlink()
    row=stage_fit_snapshot(remote,tmp_path/"content",expected_vertices=20)
    assert not row["ready"]
    assert "not published" in row["reason"]


def test_required_files_are_explicit():
    assert "fit_data/ids.json" in FIT_FILES
    assert "nsbm_partitions.npz" in FIT_FILES
    assert "fit_summary.json" in FIT_FILES
