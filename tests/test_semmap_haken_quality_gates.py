"""Hermetic integration gates for the Iteration 1 M0 preparation workflow."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.cli import main
from semmap_haken.data_manager import DatasetDescriptor, DownloadError, acquire_dataset, versioned_cache_path
from semmap_haken.graph_build import build_sparse_graph, load_prepared_graph
from semmap_haken.manifest import RunManifest
from semmap_haken.notebook import make_execution_metadata


REPOSITORY = Path(__file__).parents[1]
FIXTURE = REPOSITORY / "tests" / "fixtures" / "conceptnet_tiny.tsv"
PROFILE = REPOSITORY / "configs" / "resource_profiles" / "smoke.yaml"


def _write_config(root: Path, *, dataset_path: Path | None = FIXTURE) -> Path:
    config = root / "smoke.yaml"
    path_value = str(dataset_path) if dataset_path is not None else "null"
    config.write_text(
        f"""
paths: {{workspace_root: {root}, data_root: data, cache_root: cache, runs_root: runs}}
dataset: {{source: offline-tiny, path: {path_value}, language: en, relations: [RelatedTo, IsA], min_weight: 1.0, max_nodes: 1000, component: largest}}
graph: {{directed: false, weight_transform: log1p, operator: normalized_adjacency}}
runtime: {{profile: smoke, random_seed: 1729, resource_profile: {PROFILE}}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def _run_prepare(config: Path) -> Path:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert main(["prepare", "--config", str(config)]) == 0
    return Path(output.getvalue().strip())


def _scientific_config(resolved: dict[str, Any]) -> dict[str, Any]:
    """Exclude environment-specific storage roots from equivalence comparisons."""
    return {key: value for key, value in resolved.items() if key != "paths"}


def _assert_manifest_integrity(run_dir: Path) -> RunManifest:
    manifest = RunManifest.read_json(run_dir / "manifest.json")
    assert manifest.stages == {"prepare": "completed"}
    for artifact in manifest.artifacts.values():
        path = Path(artifact["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        assert manifest.checksums[path.name] == artifact["sha256"]
    return manifest


def test_prepare_manifest_references_and_checksums_are_complete(tmp_path: Path) -> None:
    manifest = _assert_manifest_integrity(_run_prepare(_write_config(tmp_path)))
    assert set(manifest.artifacts) == {"adjacency", "nodes", "metadata", "resolved_config"}
    assert set(manifest.checksums) == {"adjacency.npz", "nodes.json", "graph_metadata.json", "resolved_config.json"}
    metadata_path = Path(manifest.artifacts["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for filename, checksum in metadata["checksums"].items():
        assert hashlib.sha256((metadata_path.parent / filename).read_bytes()).hexdigest() == checksum


def test_notebook_execution_and_cli_prepare_are_scientifically_equivalent(tmp_path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    from nbclient import NotebookClient

    cli_root, notebook_root = tmp_path / "cli", tmp_path / "notebook"
    cli_root.mkdir()
    notebook_root.mkdir()
    cli_run = _run_prepare(_write_config(cli_root))
    notebook_config = _write_config(notebook_root, dataset_path=None)

    notebook = nbformat.read(REPOSITORY / "notebooks" / "01_data_smoke_and_sparse_graph.ipynb", as_version=4)
    source = notebook.cells[1].source
    notebook.cells[1].source = source.replace(
        "CONFIG_TEMPLATE = Path('configs/conceptnet_en_smoke.yaml')",
        f"CONFIG_TEMPLATE = Path({str(notebook_config)!r})",
    )
    NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY)}},
    ).execute()
    notebook_run = next((notebook_root / "runs").glob("prepare-*"))

    cli_graph = load_prepared_graph(cli_run)
    notebook_graph = load_prepared_graph(notebook_run)
    assert cli_graph.node_ids == notebook_graph.node_ids
    assert np.array_equal(cli_graph.adjacency.indptr, notebook_graph.adjacency.indptr)
    assert np.array_equal(cli_graph.adjacency.indices, notebook_graph.adjacency.indices)
    assert np.array_equal(cli_graph.adjacency.data, notebook_graph.adjacency.data)

    cli_manifest = _assert_manifest_integrity(cli_run)
    notebook_manifest = _assert_manifest_integrity(notebook_run)
    assert _scientific_config(cli_manifest.resolved_config) == _scientific_config(notebook_manifest.resolved_config)
    assert cli_manifest.checksums["adjacency.npz"] == notebook_manifest.checksums["adjacency.npz"]
    assert cli_manifest.checksums["nodes.json"] == notebook_manifest.checksums["nodes.json"]
    assert notebook_manifest.notebook_environment["notebook_name"] == "01_data_smoke_and_sparse_graph.ipynb"
    assert notebook_manifest.checksum_equivalent_to_cli is True
    assert notebook_manifest.cli_replay is True


class _Response:
    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self._chunks = iter(chunks)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        value = next(self._chunks, b"")
        if isinstance(value, Exception):
            raise value
        return value


def test_interrupted_download_restarts_cleans_part_and_reuses_verified_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"completed deterministic fixture"
    descriptor = DatasetDescriptor(expected_sha256=hashlib.sha256(content).hexdigest(), expected_size_bytes=len(content), filename="fixture.gz")
    responses = iter([_Response([b"partial", OSError("interrupted")]), _Response([content])])
    monkeypatch.setattr("semmap_haken.data_manager.urllib.request.urlopen", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr("semmap_haken.data_manager.time.sleep", lambda _: None)
    first = acquire_dataset(descriptor, cache_root=tmp_path / "cache", source_url="https://example.invalid/fixture.gz", retries=2)
    assert first.restarted_download is True
    assert first.path.read_bytes() == content
    assert not first.path.with_suffix(".gz.part").exists()

    monkeypatch.setattr("semmap_haken.data_manager.urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("verified cache must prevent network access"))
    second = acquire_dataset(descriptor, cache_root=tmp_path / "cache", source_url="https://example.invalid/fixture.gz")
    assert second.cache_hit and second.source == "cache"


def test_checksum_mismatch_cleans_destination_and_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = DatasetDescriptor(expected_sha256=hashlib.sha256(b"expected").hexdigest(), filename="bad.gz")
    monkeypatch.setattr("semmap_haken.data_manager.urllib.request.urlopen", lambda *_args, **_kwargs: _Response([b"unexpected"]))
    with pytest.raises(DownloadError, match="download failed"):
        acquire_dataset(descriptor, cache_root=tmp_path / "cache", source_url="https://example.invalid/bad.gz", retries=1)
    destination = versioned_cache_path(tmp_path / "cache", descriptor)
    assert not destination.exists()
    assert not destination.with_suffix(".gz.part").exists()


def test_offline_download_and_prepare_never_open_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path)
    monkeypatch.setattr("semmap_haken.data_manager.urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("offline manual path must prevent network access"))
    assert main(["download", "--config", str(config)]) == 0
    _assert_manifest_integrity(_run_prepare(config))


def test_preparation_path_does_not_densify_sparse_adjacency(monkeypatch: pytest.MonkeyPatch) -> None:
    def dense_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("M0 preparation must not materialize a dense adjacency")

    monkeypatch.setattr(sparse.spmatrix, "toarray", dense_access, raising=False)
    monkeypatch.setattr(sparse.spmatrix, "todense", dense_access, raising=False)
    records = list(__import__("semmap_haken.conceptnet", fromlist=["stream_assertions"]).stream_assertions(FIXTURE))
    graph = build_sparse_graph(records, directed=False, weight_transform="raw", component="largest", max_nodes=1000)
    assert sparse.isspmatrix_csr(graph.adjacency)
    assert graph.adjacency.nnz > 0


def test_gzip_parser_matches_plain_fixture(tmp_path: Path) -> None:
    gzip_fixture = tmp_path / "conceptnet_tiny.tsv.gz"
    with gzip.open(gzip_fixture, "wb") as stream:
        stream.write(FIXTURE.read_bytes())
    conceptnet = __import__("semmap_haken.conceptnet", fromlist=["stream_assertions"])
    plain = list(conceptnet.stream_assertions(FIXTURE))
    compressed = list(conceptnet.stream_assertions(gzip_fixture))
    assert [asdict(record) for record in compressed] == [asdict(record) for record in plain]
