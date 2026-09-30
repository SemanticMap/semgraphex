"""Compression statistics for recursive lossless grammar runs.

The report intentionally separates structural reduction from measured storage.
It reads only persisted run artifacts so the same analysis can be repeated
after Colab RESUME/publication without recomputing the hierarchy.
"""
from __future__ import annotations

import json
from pathlib import Path


def _fraction(part: int | float, whole: int | float) -> float:
    return float(part / whole) if whole else 0.0


def analyze_compression_run(run_dir: str | Path) -> dict[str, object]:
    run = Path(run_dir)
    hierarchy_path = run / "hierarchy.json"
    if not hierarchy_path.is_file():
        raise ValueError("run has no hierarchy.json")
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for item in hierarchy.get("transitions", []):
        codec = item.get("exact_transition_codec") or (
            item.get("dictionary_metrics", {}).get("exact_transition_codec")
        )
        if not codec:
            continue
        archive_bytes = int(codec.get("archive_bytes", 0))
        binary_baseline = int(codec.get("baseline_binary_bytes", 0))
        entry = {
            str(name): int(size)
            for name, size in codec.get("entry_compressed_bytes", {}).items()
        }
        payload = {
            "internal": int(entry.get("internal.bin", 0)),
            "ports": int(entry.get("ports.bin", 0)),
            "residual": int(entry.get("residual.bin", 0)),
            "occurrences": int(entry.get("occurrences.bin", 0)),
            "shapes": int(entry.get("shapes.json", 0)),
            "variants": int(entry.get("variants.json", 0)),
            "rules": int(entry.get("rules.json", 0)),
            "manifest": int(entry.get("manifest.json", 0)),
        }
        largest_name = None
        largest_bytes = 0
        if entry:
            largest_name, largest_bytes = max(
                entry.items(), key=lambda pair: pair[1]
            )
        rows.append({
            "source_level": int(item.get("source_level", len(rows))),
            "target_level": int(item.get("target_level", len(rows) + 1)),
            "fine_nodes": int(item.get("fine_nodes", 0)),
            "coarse_nodes": int(item.get("coarse_nodes", 0)),
            "node_reduction_fraction": (
                1.0
                - _fraction(
                    int(item.get("coarse_nodes", 0)),
                    int(item.get("fine_nodes", 0)),
                )
                if int(item.get("fine_nodes", 0))
                else 0.0
            ),
            "archive_bytes": archive_bytes,
            "binary_baseline_bytes": binary_baseline,
            "compression_ratio_binary": (
                float(codec.get("compression_ratio_binary_baseline"))
                if codec.get("compression_ratio_binary_baseline") is not None
                else None
            ),
            "net_saved_vs_binary_bytes": int(
                codec.get(
                    "net_saved_vs_binary_bytes",
                    binary_baseline - archive_bytes,
                )
            ),
            "shape_count": int(codec.get("shapes", 0)),
            "variant_count": int(codec.get("interface_variants", 0)),
            "occurrence_count": int(codec.get("occurrences", 0)),
            "internal_edge_records": int(codec.get("internal_edge_records", 0)),
            "port_edge_records": int(codec.get("port_edge_records", 0)),
            "residual_edge_records": int(codec.get("residual_edge_records", 0)),
            "payload_bytes": payload,
            "payload_share": {
                name: _fraction(value, archive_bytes)
                for name, value in payload.items()
            },
            "largest_entry": largest_name,
            "largest_entry_bytes": int(largest_bytes),
        })

    hierarchy_codec = hierarchy.get("hierarchy_exact_codec") or {}
    ratios = [
        row["compression_ratio_binary"]
        for row in rows
        if row["compression_ratio_binary"] is not None
    ]
    first_losing = next(
        (
            row["source_level"]
            for row in rows
            if row["compression_ratio_binary"] is not None
            and row["compression_ratio_binary"] >= 1.0
        ),
        None,
    )
    largest_transition = None
    if rows:
        largest_transition = max(
            rows,
            key=lambda row: int(row["archive_bytes"]),
        )["source_level"]

    summary = {
        "transition_count": len(rows),
        "levels": int(hierarchy.get("levels", len(rows) + 1 if rows else 0)),
        "initial_nodes": int(hierarchy.get("initial_nodes", 0)),
        "final_nodes": int(hierarchy.get("final_nodes", 0)),
        "dictionary_size": int(hierarchy.get("dictionary_size", 0)),
        "best_transition_binary_ratio": min(ratios) if ratios else None,
        "worst_transition_binary_ratio": max(ratios) if ratios else None,
        "first_transition_not_beating_binary_baseline": first_losing,
        "largest_transition_archive_level": largest_transition,
        "hierarchy_archive_bytes": hierarchy_codec.get("archive_bytes"),
        "hierarchy_binary_bundle_bytes": hierarchy_codec.get(
            "baseline_level0_binary_bundle_bytes"
        ),
        "hierarchy_compression_ratio_binary_bundle": hierarchy_codec.get(
            "compression_ratio_binary_bundle"
        ),
        "hierarchy_roundtrip_exact": hierarchy_codec.get("roundtrip_exact"),
    }
    return {
        "scope": (
            "measured ZIP bytes versus compact exact binary non-grammar "
            "baseline; node reduction is reported separately"
        ),
        "summary": summary,
        "transitions": rows,
    }


def write_compression_report(
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, object]:
    result = analyze_compression_run(run_dir)
    target = (
        Path(output)
        if output is not None
        else Path(run_dir) / "compression_analysis.json"
    )
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
