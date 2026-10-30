"""Independence, SHA256 integrity, and interruption recovery for grammar runs."""
from __future__ import annotations

import json
import numpy as np
import pytest

from semmap_haken.hypothesis.grammar_experiment import (
    _degree, _discover, _scan, _views, run,
)
from semmap_haken.hypothesis.scan_checkpoint import (
    load_shard, save_shard, sha256_file, verify_artifacts,
)

from test_nsbm_huffman_experiment import sample


def test_wishart_no_nsbm_and_combined_explicit(tmp_path):
    graph = sample()
    labels = np.repeat(np.arange(3), 4)
    x = run(graph, tmp_path / "wishart", profile="smoke",
            modes=("wishart",), discovery=8, seed=7, cpu_budget=1, device="cpu",
            min_support=1)
    assert x["fit"] is None
    assert x["methods"]["wishart"]["search"]["used_nsbm_labels"] is False
    y = run(graph, tmp_path / "combined", profile="smoke", partition=labels,
            modes=("nsbm", "nsbm_wishart", "shuffled"), discovery=8, seed=7,
            cpu_budget=1, device="cpu", min_support=1)
    assert all(row["search"]["full_scan_centers"] == graph.vertex_count
               and row["search"]["discovery_budget"] == 8
               for row in y["methods"].values())
    assert y["methods"]["nsbm_wishart"]["search"]["used_nsbm_labels"]


def test_scan_checkpoint_rejects_corruption_and_partial(tmp_path):
    p = tmp_path / "shard.json"
    identity = {"input": "abc", "n": 12}
    rows = [["T", [0, 1, 2]]]
    save_shard(p, identity=identity, rows=rows)
    assert load_shard(p, identity=identity) == rows
    data = bytearray(p.read_bytes())
    index = data.find(b'abc')
    data[index] = ord("z")  # same-length tampering is still detected
    p.write_bytes(data)
    assert load_shard(p, identity=identity) is None
    assert load_shard(p, identity={"input": "other", "n": 12}) is None
    (tmp_path / "shard.json.tmp").write_bytes(b"{incomplete")
    assert load_shard(p, identity=identity) is None
    save_shard(p, identity=identity, rows=rows)
    assert load_shard(p, identity=identity) == rows


def test_scan_resumes_after_interruption(tmp_path):
    graph = sample()
    d, _ = _degree(graph, "cpu")
    views = {"full": _views(graph, None)["full"]}
    vocab = _discover(graph, views, None, d, budget=12,
                      max_ego_nodes=5, min_figure_nodes=3,
                      seed=9, extraction_workers=1)
    config = dict(views=views, dictionary=vocab, n=graph.vertex_count,
                  batch=2, max_ego_nodes=5, min_figure_nodes=3,
                  extraction_workers=1, match_workers=1,
                  checkpoint_dir=tmp_path / "scan", identity={"graph": "sample"})
    def interrupt_once(_):
        raise InterruptedError("simulated Colab disconnect")
    with pytest.raises(InterruptedError):
        _scan(**config, checkpoint_hook=interrupt_once)
    first, counts, meta = _scan(**config)
    assert meta["checkpoint_shards_reused"] >= 1
    fresh, fresh_counts, _ = _scan(**{**config, "checkpoint_dir": tmp_path / "fresh"})
    assert first == fresh and counts == fresh_counts
    shard = next((tmp_path / "scan").glob("*.json"))
    raw = bytearray(shard.read_bytes())
    raw[-3] ^= 1
    shard.write_bytes(raw)
    again, same_counts, status = _scan(**config)
    assert status["checkpoint_shards_computed"] >= 1
    assert again == first and same_counts == counts


def test_corrupt_zip_after_completed_forces_repair(tmp_path):
    graph = sample()
    labels = np.repeat(np.arange(3), 4)
    out = tmp_path / "complete"
    args = dict(profile="smoke", seed=3, partition=labels, cpu_budget=1,
                device="cpu", modes=("nsbm",), max_ego_nodes=5,
                min_figure_nodes=3, min_support=1)
    report = run(graph, out, **args)
    obj = report["methods"]["nsbm"]
    assert verify_artifacts(obj, out)
    zip_path = out / "bundle_nsbm_huffman.zip"
    corrupted = bytearray(zip_path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    zip_path.write_bytes(corrupted)
    assert not verify_artifacts(obj, out)
    again = run(graph, out, resume=True, **args)
    assert again["status"] == "complete"
    assert verify_artifacts(again["methods"]["nsbm"], out)
    assert (out / "COMPLETED").is_file()


def test_scan_parallel_matches_serial(tmp_path):
    graph = sample()
    degree, _ = _degree(graph, "cpu")
    views = {"full": _views(graph, None)["full"]}
    dictionary = _discover(graph, views, None, degree, budget=12,
                           max_ego_nodes=5, min_figure_nodes=3,
                           seed=3, extraction_workers=1)
    options = dict(views=views, dictionary=dictionary, n=graph.vertex_count,
                   batch=2, max_ego_nodes=5, min_figure_nodes=3,
                   extraction_workers=1, identity={"source": "small-graph"})
    serial = _scan(**options, match_workers=1,
                   checkpoint_dir=tmp_path / "serial")
    parallel = _scan(**options, match_workers=2,
                     checkpoint_dir=tmp_path / "parallel")
    assert serial[:2] == parallel[:2]


def test_resume_repairs_truncated_json_metadata(tmp_path):
    graph = sample()
    out = tmp_path / "torn"
    args = dict(profile="smoke", seed=15, cpu_budget=1, device="cpu",
                modes=("plain",), discovery=6, min_support=1)
    run(graph, out, **args)
    (out / "method_plain.json").write_bytes(b'{incomplete')
    (out / "report.json").write_bytes(b'{incomplete')
    (out / "method_plain.json.tmp").write_bytes(b'{not published')
    repaired = run(graph, out, resume=True, **args)
    assert repaired["status"] == "complete"
    assert verify_artifacts(repaired["methods"]["plain"], out)
    assert (out / "COMPLETED").exists()
