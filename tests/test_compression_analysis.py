"""Tests for persisted compression-statistics analysis."""
from __future__ import annotations

import json

import pytest
from semmap_haken.compression_analysis import (
    analyze_compression_run,
    write_compression_report,
)


def test_compression_analysis_uses_binary_baseline_and_payload_shares(tmp_path):
    hierarchy = {
        "levels": 3,
        "initial_nodes": 100,
        "final_nodes": 60,
        "dictionary_size": 7,
        "transitions": [
            {
                "source_level": 0,
                "target_level": 1,
                "fine_nodes": 100,
                "coarse_nodes": 80,
                "exact_transition_codec": {
                    "archive_bytes": 900,
                    "baseline_binary_bytes": 1000,
                    "compression_ratio_binary_baseline": 0.9,
                    "net_saved_vs_binary_bytes": 100,
                    "shapes": 4,
                    "interface_variants": 5,
                    "occurrences": 12,
                    "internal_edge_records": 30,
                    "port_edge_records": 10,
                    "residual_edge_records": 20,
                    "entry_compressed_bytes": {
                        "manifest.json": 50,
                        "shapes.json": 100,
                        "variants.json": 80,
                        "rules.json": 70,
                        "occurrences.bin": 120,
                        "internal.bin": 200,
                        "ports.bin": 150,
                        "residual.bin": 130,
                    },
                },
            },
            {
                "source_level": 1,
                "target_level": 2,
                "fine_nodes": 80,
                "coarse_nodes": 60,
                "exact_transition_codec": {
                    "archive_bytes": 1100,
                    "baseline_binary_bytes": 1000,
                    "compression_ratio_binary_baseline": 1.1,
                    "net_saved_vs_binary_bytes": -100,
                    "shapes": 5,
                    "interface_variants": 8,
                    "occurrences": 9,
                    "internal_edge_records": 20,
                    "port_edge_records": 25,
                    "residual_edge_records": 30,
                    "entry_compressed_bytes": {
                        "ports.bin": 400,
                        "residual.bin": 300,
                    },
                },
            },
        ],
        "hierarchy_exact_codec": {
            "archive_bytes": 1500,
            "baseline_level0_binary_bundle_bytes": 1800,
            "compression_ratio_binary_bundle": 1500 / 1800,
            "roundtrip_exact": True,
        },
    }
    (tmp_path / "hierarchy.json").write_text(
        json.dumps(hierarchy), encoding="utf-8"
    )

    report = analyze_compression_run(tmp_path)
    assert report["summary"]["best_transition_binary_ratio"] == 0.9
    assert report["summary"]["worst_transition_binary_ratio"] == 1.1
    assert report["summary"]["first_transition_not_beating_binary_baseline"] == 1
    assert report["summary"]["hierarchy_roundtrip_exact"] is True
    assert report["transitions"][0]["node_reduction_fraction"] == pytest.approx(0.2)
    assert report["transitions"][1]["largest_entry"] == "ports.bin"
    assert report["transitions"][1]["payload_share"]["ports"] == 400 / 1100

    output = tmp_path / "analysis.json"
    written = write_compression_report(tmp_path, output)
    assert output.is_file()
    assert written == report
