"""Tests for deterministic grammar codec benchmarks."""
from __future__ import annotations

from semmap_haken.grammar_benchmark import (
    repeated_triangles,
    run_synthetic_benchmark,
)


def test_repeated_triangle_generator_is_deterministic():
    n, edges, figures = repeated_triangles(3, port_edges_per_figure=1)
    assert n == 12
    assert len(edges) == 15
    assert figures == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert [edge.edge_id for edge in edges] == list(range(len(edges)))


def test_synthetic_benchmark_reports_reuse_and_port_load():
    report = run_synthetic_benchmark(12, port_edges_per_figure=2)
    repeated = report["repeated_exact_motif"]
    residual = report["same_graph_residual_only"]
    ports = report["port_heavy_repeated_motif"]

    assert repeated["shapes"] == 1
    assert repeated["occurrences"] == 12
    assert repeated["internal_edge_records"] == 48
    assert repeated["residual_edge_records"] == 0

    assert residual["shapes"] == 0
    assert residual["occurrences"] == 0
    assert residual["residual_edge_records"] == 48

    assert ports["shapes"] == 1
    assert ports["occurrences"] == 12
    assert ports["port_edge_records"] == 24
    assert ports["binary_ratio"] is not None
