"""Adversarial acceptance tests for Iteration 1 Workstream 5B."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from semmap_haken.conceptnet import AssertionParseError, ParseReport, stream_assertions
from semmap_haken.data_manager import DatasetDescriptor, DownloadError, acquire_dataset
from semmap_haken.graph_build import build_sparse_graph, load_prepared_graph


def _assertion(start: str, end: str, relation: str, weight: object = 1.0) -> str:
    return f"/a/test\t/r/{relation}\t{start}\t{end}\t{json.dumps({'weight': weight, 'dataset': 'd', 'sources': ['/s/x'], 'license': 'cc'})}\n"


def test_invalid_boolean_and_nonfinite_weights_fail_or_are_counted(tmp_path: Path) -> None:
    dump = tmp_path / "bad.tsv"
    dump.write_text("".join((_assertion("/c/en/a", "/c/en/b", "RelatedTo", True), _assertion("/c/en/b", "/c/en/c", "IsA", float("nan")))), encoding="utf-8")
    with pytest.raises(AssertionParseError, match="invalid weight"):
        list(stream_assertions(dump))
    report = ParseReport()
    assert list(stream_assertions(dump, invalid_mode="skip_invalid", report=report)) == []
    assert report.rejected["invalid_weight"] == 2
    compressed = tmp_path / "bad.tsv.gz"
    with gzip.open(compressed, "wb") as stream:
        stream.write(dump.read_bytes())
    with pytest.raises(AssertionParseError, match="invalid weight"):
        list(stream_assertions(compressed))


def test_selected_provenance_histogram_and_loop_edge_count_are_exact(tmp_path: Path) -> None:
    dump = tmp_path / "selection.tsv"
    dump.write_text("".join((_assertion("/c/en/a", "/c/en/b", "RelatedTo"), _assertion("/c/en/a", "/c/en/a", "IsA"), _assertion("/c/en/x", "/c/en/y", "PartOf"))), encoding="utf-8")
    graph = build_sparse_graph(stream_assertions(dump), directed=False, weight_transform="raw", component="largest", max_nodes=2, self_loop_policy="exclude")
    assert graph.report["edge_count"] == 1
    assert graph.report["adjacency_nnz"] == 2
    assert graph.report["relation_histogram"] == {"RelatedTo": 1}
    assert graph.report["self_loop_policy"] == "exclude"
    assert graph.selected_edges[0]["relation"] == "RelatedTo"


def test_loader_rejects_partial_or_uncommitted_artifact(tmp_path: Path) -> None:
    tmp_path.joinpath("nodes.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker"):
        load_prepared_graph(tmp_path)


def test_unpinned_official_cache_is_never_verified(tmp_path: Path) -> None:
    descriptor = DatasetDescriptor()
    destination = tmp_path / "datasets" / "conceptnet" / descriptor.version / "unverified" / descriptor.filename
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"unverified")
    with pytest.raises(DownloadError, match="pinned expected SHA-256"):
        acquire_dataset(descriptor, cache_root=tmp_path, retries=0)
