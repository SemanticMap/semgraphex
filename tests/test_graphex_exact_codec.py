"""Behavioral tests for exclusive classification and actual archive decoding."""
from __future__ import annotations

import json
import zipfile

import pytest

from semmap_haken.graphex_components import (
    EdgeRecord, classify_edges, validate_partition,
)
from semmap_haken.graphex_codec import decode_graph, encode_graph


def example():
    edges = (
        EdgeRecord(0, 0, 1, "RelatedTo", 0.75),
        EdgeRecord(1, 1, 2, "RelatedTo", 1.25),
        EdgeRecord(2, 2, 0, "RelatedTo", 0.25),
        EdgeRecord(3, 3, 4, "RelatedTo", 0.75),
        EdgeRecord(4, 4, 5, "RelatedTo", 1.25),
        EdgeRecord(5, 5, 3, "RelatedTo", 0.25),
        EdgeRecord(6, 0, 6, "AtLocation", 0.5),  # outbound leaf
        EdgeRecord(7, 7, 3, "AtLocation", 0.5),  # inbound leaf
        EdgeRecord(8, 2, 4, "RelatedTo", 2.0),  # figure boundary -> R
        EdgeRecord(9, 8, 9, "IsA", 0.6),        # isolated pair
        EdgeRecord(10, 0, 2, "IsA", 0.3),
        EdgeRecord(11, 0, 2, "IsA", 0.4),        # parallel record
        EdgeRecord(12, 10, 10, "Self", 0.8),    # loop -> R
    )
    return edges, ((0, 1, 2), (3, 4, 5))


def test_partition_excludes_cross_figure_ports_and_preserves_multiplicity():
    edges, figures = example()
    assigned = classify_edges(11, edges, figures)
    validate_partition(edges, assigned)
    parts = {item.edge.edge_id: item.part for item in assigned}
    assert [parts[i] for i in range(13)] == [
        "W", "W", "W", "W", "W", "W", "S", "S", "R", "I",
        "W", "W", "R",
    ]


def test_no_false_dust_for_rare_bridge_or_parallel_pairs():
    edges = (EdgeRecord(0, 0, 1, "r", 1.0),
             EdgeRecord(1, 0, 1, "r", 2.0),
             EdgeRecord(2, 1, 2, "r", 3.0))
    assert {item.part for item in classify_edges(3, edges, ())} == {"R"}


def test_disjoint_figures_and_unique_record_ids_are_required():
    edges, figures = example()
    with pytest.raises(ValueError, match="overlapping"):
        classify_edges(11, edges, ((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="edge_id"):
        classify_edges(11, (edges[0], edges[0]), figures)


def test_archive_roundtrip_and_measured_bytes(tmp_path):
    edges, figures = example()
    target = tmp_path / "graphex.zip"
    result = encode_graph(11, edges, figures, target)
    assert result["roundtrip_exact"]
    assert result["archive_bytes"] == target.stat().st_size
    assert result["shapes"] >= 1
    assert result["partition"]["R"] == 2
    assert decode_graph(target) == (11, edges)
    with zipfile.ZipFile(target) as archive:
        meta = json.loads(archive.read("manifest.json"))
        assert meta["format"] == "graphex_exact_v1"
        assert meta["edge_count"] == len(edges)
        assert len(archive.read("symbols.bin")) * 8 >= meta["symbol_bits"]


def test_empty_graph_and_unsupported_version(tmp_path):
    target = tmp_path / "empty.zip"
    encode_graph(2, (), (), target)
    assert decode_graph(target) == (2, ())
    with zipfile.ZipFile(target, "a") as archive:
        archive.writestr("manifest.json", '{"format":"old"}')
    with pytest.raises(ValueError, match="unsupported"):
        decode_graph(target)


def test_nonfinite_weight_is_not_silently_serialized(tmp_path):
    with pytest.raises(ValueError):
        encode_graph(2, (EdgeRecord(0, 0, 1, "r", float("nan")),),
                     (), tmp_path / "bad.zip")


def test_accelerated_cpu_path_matches_exact_classifier():
    from semmap_haken.graphex_components_gpu import classify_edges_accelerated
    edges, figures = example()
    assert classify_edges_accelerated(11, edges, figures, device="cpu",
                                      batch_size=2) == classify_edges(11, edges, figures)


def test_cuda_classifier_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CI runner does not provide a CUDA accelerator")
    from semmap_haken.graphex_components_gpu import classify_edges_accelerated
    edges, figures = example()
    assert classify_edges_accelerated(11, edges, figures, device="cuda",
                                      batch_size=2) == classify_edges(11, edges, figures)


def test_colab_notebook_is_valid_python_and_uses_real_gpu_path():
    import ast
    from pathlib import Path
    notebook_path = Path(__file__).parents[1] / "notebooks/07_graphex_exact_v1_colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    ast.parse(code)
    assert "--device" in code and "gpu_batch_size" in code
    assert "sha256" in code and "COMPLETED" in code



def test_colab_report_cell_works_without_pandas():
    """The summary must work even if NumPy/pandas imports are damaged."""
    import ast
    import contextlib
    import io
    from pathlib import Path

    notebook = json.loads((Path(__file__).parents[1] /
        "notebooks/07_graphex_exact_v1_colab_gpu.ipynb").read_text(encoding="utf-8"))
    summary = "".join(notebook["cells"][16]["source"])
    tree = ast.parse(summary)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   any(alias.name == "pandas" for alias in node.names)
                   for node in ast.walk(tree))
    report = {"level": 0, "classification_device": "cpu", "edge_records": 3,
              "partition": {"W": 1, "S": 1, "I": 0, "R": 1},
              "shapes": 1, "archive_bytes": 100, "baseline_bytes": 150,
              "net_saved_bytes": 50, "stream_bits": 16,
              "roundtrip_exact": True}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(summary, "colab-summary", "exec"),
             {"reports": [report], "run_folder_id": "test-folder"})
    assert "All tested levels" in stream.getvalue()
    assert "50" in stream.getvalue()


def test_colab_bootstrap_handles_python_313_without_legacy_numpy_constraints():
    from pathlib import Path
    notebook = json.loads((Path(__file__).parents[1] /
        "notebooks/07_graphex_exact_v1_colab_gpu.ipynb").read_text(encoding="utf-8"))
    bootstrap = "".join(notebook["cells"][6]["source"])
    assert "sys.version_info" in bootstrap
    assert "numpy.typing" in bootstrap
    assert "pip" in bootstrap


def test_symmetric_csr_records_count_one_structural_neighbor():
    """Reciprocal rows are two records but only one structural neighbor."""
    edges = (
        EdgeRecord(0, 0, 1, "r", 1.0),
        EdgeRecord(1, 1, 0, "r", 1.0),
        EdgeRecord(2, 0, 2, "r", 0.5),
        EdgeRecord(3, 2, 0, "r", 0.5),
        EdgeRecord(4, 3, 4, "r", 0.75),
        EdgeRecord(5, 4, 3, "r", 0.75),
    )
    expected = ["W", "W", "S", "S", "I", "I"]
    actual = classify_edges(5, edges, ((0, 1),))
    assert [row.part for row in actual] == expected
    from semmap_haken.graphex_components_gpu import classify_edges_accelerated
    assert [row.part for row in classify_edges_accelerated(
        5, edges, ((0, 1),), device="cpu")] == expected


def test_multiple_relations_to_one_neighbor_are_star_not_multiple_neighbors():
    edges = (
        EdgeRecord(0, 0, 1, "r", 1.0),
        EdgeRecord(1, 1, 0, "r", 1.0),
        EdgeRecord(2, 0, 2, "r", 0.5),
        EdgeRecord(3, 2, 0, "r", 0.5),
        EdgeRecord(4, 0, 2, "other", 0.25),
        EdgeRecord(5, 2, 0, "other", 0.25),
    )
    assert [item.part for item in classify_edges(3, edges, ((0, 1),))] == [
        "W", "W", "S", "S", "S", "S"]


def test_self_loop_blocks_star_or_dust_at_that_vertex():
    edges = (
        EdgeRecord(0, 0, 1, "r", 1.0),
        EdgeRecord(1, 1, 0, "r", 1.0),
        EdgeRecord(2, 0, 2, "r", 0.5),
        EdgeRecord(3, 2, 0, "r", 0.5),
        EdgeRecord(4, 2, 2, "loop", 1.0),
    )
    assert [item.part for item in classify_edges(3, edges, ((0, 1),))] == [
        "W", "W", "R", "R", "R"]
