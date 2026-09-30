"""Deterministic synthetic benchmarks for grammar_exact_v2.

These benchmarks are implementation diagnostics, not evidence about ConceptNet.
They answer whether the codec can exploit exact repeated topology at all and
how external-port load changes measured storage.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .grammar_codec import encode_grammar
from .graphex_components import EdgeRecord


def repeated_triangles(
    repeats: int,
    *,
    port_edges_per_figure: int = 0,
) -> tuple[int, tuple[EdgeRecord, ...], tuple[tuple[int, ...], ...]]:
    repeats = int(repeats)
    port_edges_per_figure = int(port_edges_per_figure)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if port_edges_per_figure < 0:
        raise ValueError("port_edges_per_figure must be non-negative")

    motif_nodes = 3 * repeats
    external_nodes = repeats * port_edges_per_figure
    vertex_count = motif_nodes + external_nodes
    rows: list[EdgeRecord] = []
    figures: list[tuple[int, ...]] = []
    edge_id = 0
    external_cursor = motif_nodes

    for index in range(repeats):
        a = 3 * index
        b = a + 1
        c = a + 2
        figures.append((a, b, c))
        for source, target, relation, weight in (
            (a, b, "r", 1.0),
            (b, c, "r", 1.0),
            (c, a, "r", 1.0),
            (a, c, "q", 0.5),
        ):
            rows.append(
                EdgeRecord(edge_id, source, target, relation, weight)
            )
            edge_id += 1
        for _ in range(port_edges_per_figure):
            rows.append(
                EdgeRecord(
                    edge_id,
                    b,
                    external_cursor,
                    "external",
                    0.25,
                )
            )
            edge_id += 1
            external_cursor += 1
    return vertex_count, tuple(rows), tuple(figures)


def run_synthetic_benchmark(
    repeats: int = 500,
    *,
    port_edges_per_figure: int = 2,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="semmap-grammar-benchmark-") as tmp:
        root = Path(tmp)
        vertex_count, edges, figures = repeated_triangles(repeats)

        repeated = encode_grammar(
            vertex_count,
            edges,
            figures,
            root / "repeated.zip",
        )
        residual = encode_grammar(
            vertex_count,
            edges,
            (),
            root / "residual.zip",
        )

        port_n, port_edges, port_figures = repeated_triangles(
            repeats,
            port_edges_per_figure=port_edges_per_figure,
        )
        port_heavy = encode_grammar(
            port_n,
            port_edges,
            port_figures,
            root / "ports.zip",
        )

    def compact(report: dict[str, object]) -> dict[str, object]:
        return {
            "archive_bytes": report["archive_bytes"],
            "binary_baseline_bytes": report["baseline_binary_bytes"],
            "binary_ratio": report["compression_ratio_binary_baseline"],
            "net_saved_vs_binary_bytes": report[
                "net_saved_vs_binary_bytes"
            ],
            "shapes": report["shapes"],
            "interface_variants": report["interface_variants"],
            "occurrences": report["occurrences"],
            "internal_edge_records": report["internal_edge_records"],
            "port_edge_records": report["port_edge_records"],
            "residual_edge_records": report["residual_edge_records"],
            "entry_compressed_bytes": report["entry_compressed_bytes"],
        }

    return {
        "scope": (
            "deterministic synthetic codec diagnostic; not a ConceptNet "
            "compression result"
        ),
        "repeats": int(repeats),
        "port_edges_per_figure": int(port_edges_per_figure),
        "repeated_exact_motif": compact(repeated),
        "same_graph_residual_only": compact(residual),
        "port_heavy_repeated_motif": compact(port_heavy),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--port-edges-per-figure", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(
        run_synthetic_benchmark(
            args.repeats,
            port_edges_per_figure=args.port_edges_per_figure,
        ),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
