"""Bounded CPU pipeline, observed overlap, progress and resume (no graph-tool/GPU)."""
from __future__ import annotations

import json
from threading import Event
from types import SimpleNamespace

import pytest

from semmap_haken.graph_dictionary import GraphDictionary
from semmap_haken.hypothesis.grammar_experiment import (
    _scan, _vf2_cpu_plan, run,
)
from test_nsbm_huffman_experiment import sample


def test_prefetch_overlaps_the_first_vf2_batch(monkeypatch, tmp_path):
    from semmap_haken.hypothesis import grammar_experiment as module
    next_batch_started = Event()

    class View:
        calls = 0

        def extract_centers(self, centers, **_):
            self.calls += 1
            if self.calls == 2:
                next_batch_started.set()
            return (SimpleNamespace(nodes=(0, 1, 2)),)

    class MatchPool:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, _func, pieces):
            # If the producer does not run while VF2 processes the current
            # batch, this assertion fails and the test remains bounded.
            assert next_batch_started.wait(3), "next extraction did not overlap VF2"
            return [tuple(None for _ in piece) for piece in pieces]

    monkeypatch.setattr(module, "spawn_pool", lambda **_: MatchPool())
    updates = []
    rows, counts, stats = _scan(
        {"full": View()}, GraphDictionary(boundary_sensitive=False),
        n=8, batch=3, max_ego_nodes=5, min_figure_nodes=3,
        extraction_workers=1, match_workers=2,
        checkpoint_dir=tmp_path, identity={"key": 1},
        progress_hook=lambda done, total: updates.append((done, total)),
    )
    assert rows == [] and not counts
    assert stats["pipeline"] == "one_batch_prefetch"
    assert stats["candidates_extracted"] == stats["candidates_matched"] == 3
    assert stats["centers_processed"] == stats["centers_total"] == 8
    assert stats["checkpoint_shards_computed"] == 1
    assert updates[-1] == (8, 8)
    assert stats["wall_seconds"] >= 0
    assert stats["extraction_seconds"] >= 0
    assert stats["vf2_seconds"] >= 0
    # RESUME skips matching entirely, but still advances progress.
    next_batch_started.clear()
    cached, _, resumed = _scan(
        {"full": View()}, GraphDictionary(boundary_sensitive=False),
        n=8, batch=3, max_ego_nodes=5, min_figure_nodes=3,
        extraction_workers=1, match_workers=2,
        checkpoint_dir=tmp_path, identity={"key": 1},
    )
    assert cached == rows
    assert resumed["checkpoint_shards_reused"] == 1
    assert resumed["candidates_extracted"] == 0


def test_worker_budget_respects_ram_and_cpu(monkeypatch):
    from semmap_haken.hypothesis import grammar_experiment as module
    dictionary = GraphDictionary(boundary_sensitive=False)
    monkeypatch.setattr(module, "_available_cpus", lambda: 12)
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 300 * (1 << 20))
    assert _vf2_cpu_plan(12, dictionary) == (4, 1)
    assert _vf2_cpu_plan(1, dictionary) == (1, 1)
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 12 * (1 << 30))
    extract, match = _vf2_cpu_plan(12, dictionary)
    assert (extract, match) == (4, 8)


def test_all_stages_emit_progress_and_configured_isomorphism(tmp_path, capsys):
    out = tmp_path / "run"
    result = run(
        sample(), out, profile="smoke", modes=("plain",), device="cpu",
        cpu_budget=1, discovery=5, isomorphism_level="typed",
        max_ego_nodes=5, min_figure_nodes=3, min_support=1,
    )
    progress = json.loads((out / "progress.json").read_text())
    assert progress["overall_percent"] == 100.0
    assert progress["phase"] == "completed"
    assert progress["steps_total"] >= 4
    assert result["isomorphism_level"] == "typed"
    assert result["phase_timing_seconds"]["binary_baseline"] >= 0
    assert result["methods"]["plain"]["phase_timing_seconds"]["vf2"] >= 0
    output = capsys.readouterr().out
    assert "PROGRESS " in output
    assert "plain: VF2" in output
    assert "100.0" in output
    cached = run(
        sample(), out, profile="smoke", modes=("plain",), device="cpu",
        cpu_budget=1, discovery=5, isomorphism_level="typed",
        max_ego_nodes=5, min_figure_nodes=3, min_support=1, resume=True,
    )
    assert cached["ratios_full_bundle"] == result["ratios_full_bundle"]
    with pytest.raises(ValueError, match="RESUME"):
        run(sample(), out, profile="smoke", modes=("plain",), device="cpu",
            cpu_budget=1, discovery=5, isomorphism_level="topology",
            max_ego_nodes=5, min_figure_nodes=3, min_support=1, resume=True)


def test_drive_publishes_only_verified_archives_and_completed_last(tmp_path):
    """Exercise the notebook's actual Drive copier without Colab or credentials."""
    import ast
    import hashlib
    import os
    import re
    import shutil
    from pathlib import Path

    notebook = Path(__file__).resolve().parents[1] / (
        "notebooks/21_shared_shapes_t4_l4_colab.ipynb")
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][8]["source"])
    tree = ast.parse(cell)
    fn = next(node for node in tree.body if
              isinstance(node, ast.FunctionDef) and node.name == "sync_checkpoint")
    function = ast.Module(body=[fn], type_ignores=[])
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    archive = local / "recursive" / "level_001.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"sample exact recursive archive")
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (local / "progress.json").write_text('{"overall_percent":40}')
    namespace = {
        "RUN_LOCAL": local, "PERSISTENT": drive,
        "shutil": shutil, "os": os, "re": re, "json": json, "Path": Path,
        "file_sha256": sha,
    }
    exec(compile(function, str(notebook), "exec"), namespace)
    sync = namespace["sync_checkpoint"]

    sync()
    assert (drive / "progress.json").exists()
    assert not (drive / "recursive" / "level_001.zip").exists()
    assert not (drive / "COMPLETED").exists()
    meta = archive.with_suffix(".json")
    meta.write_text(json.dumps({"sha256": "invalid"}))
    sync()
    assert not (drive / "recursive" / "level_001.zip").exists()

    meta.write_text(json.dumps({"sha256": sha(archive)}))
    sync()
    published = drive / "recursive" / "level_001.zip"
    assert published.read_bytes() == archive.read_bytes()
    before = published.stat()
    # Deliberately corrupt with identical size/mtime; copier must repair.
    published.write_bytes(b"z" * len(archive.read_bytes()))
    os.utime(published, ns=(before.st_atime_ns, before.st_mtime_ns))
    sync()
    assert sha(published) == sha(archive)

    (local / "report.json").write_text(json.dumps({
        "methods": {}, "recursive": {
            "archive": str(archive), "archive_sha256": sha(archive),
        }
    }))
    (local / "COMPLETED").write_text("complete\n")
    sync(final=True)
    assert (drive / "COMPLETED").read_text() == "complete\n"
