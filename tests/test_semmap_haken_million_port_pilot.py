"""CPU-parallel and CUDA-preference regression tests for the 1M pilot."""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import semmap_haken.million_port_pilot as pilot
from semmap_haken.port_factorization import (
    compact_json, denormalize_prototype, factorize, restore, snapshot,
)
from semmap_haken.wishart_metrics import EgoCandidate


def _prototype(port: str = "[]", relation: str = "RelatedTo") -> dict:
    return {
        "nodes": [
            {"prototype_node": 0, "symbol_type": "ATOM", "boundary_ports": port},
            {"prototype_node": 1, "symbol_type": "ATOM", "boundary_ports": "[]"},
        ],
        "edges": [{"source": 0, "target": 1, "relation": relation}],
    }


def _candidate() -> EgoCandidate:
    matrix = sparse.csr_matrix(np.array([
        [0, 1, 0], [1, 0, 1], [0, 1, 0],
    ], dtype=float))
    return EgoCandidate(
        center=0, nodes=np.arange(3), adjacency=matrix,
        relation_layers={"RelatedTo": matrix}, boundary_signature=(),
        node_types=(None, None, None),
    )


def test_lossless_ports_and_typed_shapes() -> None:
    originals = {
        "A": _prototype(),
        "B": _prototype('[["RelatedTo","out",3]]'),
        "C": _prototype("[]", "IsA"),
    }
    baseline, packed = factorize(originals)
    assert len(packed["shapes"]) == 2
    assert compact_json(restore(packed)) == compact_json(baseline)
    for row in restore(packed)["types"]:
        assert denormalize_prototype(row) == originals[row[0]]
    summary = snapshot(originals, Counter({"A": 2, "B": 3, "C": 1}))
    assert summary["roundtrip_exact"] and summary["types_in_shared_shapes"] == 2
    assert summary["occurrences_in_shared_shapes"] == 5


def test_parallel_spawn_matches_serial_fingerprints() -> None:
    candidates = (_candidate(),) * 12
    serial, serial_meta = pilot.parallel_fingerprints(candidates, workers=1, batch_size=2)
    parallel, meta = pilot.parallel_fingerprints(candidates, workers=2, batch_size=2)
    assert serial == parallel
    assert len(serial) == len(candidates)
    assert serial_meta["fingerprint_backend"] == "single_process"
    assert meta["fingerprint_backend"] == "process_pool_spawn"
    assert 1 <= meta["worker_processes_observed"] <= 2


def _knn_fixture(monkeypatch: pytest.MonkeyPatch, backend: str, selected: str) -> dict:
    rows = []
    for i in range(6):
        others = sorted(
            [j for j in range(6) if i != j],
            key=lambda j: (abs(i - j), j),
        )
        rows.append(others)
    indices = np.asarray(rows, dtype=np.int64)
    distances = np.asarray(
        [[float(abs(i - j) + 0.1) for j in row] for i, row in enumerate(rows)]
    )
    captured: dict = {}
    def fake_neighbors(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            indices=indices, distances=distances,
            metadata={"backend": backend, "device": selected},
        )
    monkeypatch.setattr(pilot, "build_neighbor_graph", fake_neighbors)
    monkeypatch.setattr(pilot, "cuda_diagnostics",
                        lambda: {"cuda_available": selected == "cuda"})
    monkeypatch.setattr(pilot, "resolve_device", lambda device: selected)
    return captured


def test_cuda_auto_selects_real_cuda_knn_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _knn_fixture(monkeypatch, "torch_cuda", "cuda")
    dictionary = SimpleNamespace(representative=lambda tid: _candidate())
    counts = Counter({f"T{i}": 3 for i in range(6)})
    result = pilot.wishart_diagnostics(
        dictionary, counts, device="auto", workers=2,
        gpu_batch_size=32, seed=1729, output_dir=tmp_path,
    )
    assert result["device_selected"] == "cuda"
    assert result["knn_backend"] == "torch_cuda"
    assert captured["device"] == "cuda"
    assert captured["min_gpu_types"] == 1
    assert captured["cpu_workers"] == 2
    assert (tmp_path / "wishart_labels.npz").is_file()


def test_cuda_backend_mismatch_is_not_silent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _knn_fixture(monkeypatch, "sklearn_cpu", "cuda")
    dictionary = SimpleNamespace(representative=lambda tid: _candidate())
    with pytest.raises(RuntimeError, match="CUDA was selected"):
        pilot.wishart_diagnostics(
            dictionary, Counter({f"T{i}": 3 for i in range(6)}),
            device="auto", workers=2, gpu_batch_size=32,
            seed=1729, output_dir=tmp_path,
        )


def test_explicit_cpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _knn_fixture(monkeypatch, "sklearn_cpu", "cpu")
    dictionary = SimpleNamespace(representative=lambda tid: _candidate())
    result = pilot.wishart_diagnostics(
        dictionary, Counter({f"T{i}": 3 for i in range(6)}),
        device="auto", workers=1, gpu_batch_size=32,
        seed=1729, output_dir=tmp_path,
    )
    assert result["device_selected"] == "cpu"
    assert result["knn_backend"] == "sklearn_cpu"


def test_tiny_end_to_end_cpu_with_spawn(tmp_path: Path) -> None:
    source = tmp_path / "tiny.tsv"
    metadata = json.dumps({"weight": 2.0, "sources": [], "dataset": "unit"})
    lines = []
    for i in range(1, 10):
        lines.append("\t".join([
            f"/a/[/r/RelatedTo/,/c/en/n0/,/c/en/n{i}/]",
            "/r/RelatedTo", "/c/en/n0", f"/c/en/n{i}", metadata,
        ]) + "\n")
    source.write_text("".join(lines), encoding="utf-8")
    report = pilot.run_probe(
        dataset=source, output_dir=tmp_path / "out",
        target_nodes=10, sample_centers=6, budgets=(2, 4),
        seed=1729, max_ego_nodes=8, workers=2,
        device="cpu", source_sha256=pilot.sha256_file(source),
    )
    assert report["actual_nodes"] == 10
    assert report["parallel"]["fingerprint_backend"] == "process_pool_spawn"
    assert report["parallel"]["ego_extraction_backend"] == "thread_pool"
    assert all(row["roundtrip_exact"] for row in report["results"])
    assert (tmp_path / "out" / "COMPLETED").is_file()


def test_notebook_runs_module_and_checks_actual_cuda_backend() -> None:
    notebook = json.loads(
        Path("notebooks/08_conceptnet_1m_port_factorization_cuda_colab.ipynb")
        .read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "DEVICE='auto'" in code
    assert "selected_device=resolve_device(DEVICE)" in code
    assert "knn_backend" in code and "torch_cuda" in code
    assert "--workers" in code and "--device" in code
    assert "semmap_haken.million_port_pilot" in code
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook_cell_{i}")
