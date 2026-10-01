"""Relation-subset ablations for persisted recursive grammar runs.

The retrospective mode reuses accepted figure placements from an existing run
and re-encodes the exact source-level relation layers with a restricted grammar
relation set. This isolates codec/shape fragmentation effects from discovery.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from scipy import sparse

from .transition_grammar import encode_transition_grammar


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_relation_layers(level_dir: str | Path) -> dict[str, sparse.csr_matrix]:
    level = Path(level_dir)
    relation_dir = level / "relations"
    index = _load_json(relation_dir / "index.json")
    return {
        str(relation): sparse.load_npz(relation_dir / str(filename)).tocsr()
        for relation, filename in index.items()
    }


def load_symbol_types(level_dir: str | Path) -> dict[int, str]:
    result: dict[int, str] = {}
    path = Path(level_dir) / "symbolic_nodes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        symbol = row.get("dictionary_type_id")
        if symbol:
            result[int(row["node"])] = str(symbol)
    return result


def load_figure_nodes(transition_dir: str | Path) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    path = Path(transition_dir) / "figure_occurrences.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(tuple(int(node) for node in row["fine_nodes"]))
    return tuple(rows)


def transition_levels(transition_dir: str | Path) -> tuple[int, int]:
    name = Path(transition_dir).name
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "transition":
        raise ValueError(f"unexpected transition directory: {name}")
    return int(parts[1]), int(parts[2])


def baseline_transition_row(
    run_dir: str | Path,
    transition_dir: str | Path,
) -> dict[str, object]:
    run = Path(run_dir)
    transition = Path(transition_dir)
    source_level, target_level = transition_levels(transition)
    metrics = _load_json(transition / "dictionary_metrics.json")
    wishart = _load_json(transition / "wishart.json")
    dynamic_path = run / f"level_{source_level:03d}" / "dynamic_metrics.json"
    dynamic = _load_json(dynamic_path) if dynamic_path.is_file() else {}
    codec = metrics.get("exact_transition_codec") or {}
    entry = codec.get("entry_compressed_bytes") or {}
    archive_bytes = int(codec.get("archive_bytes", 0) or 0)
    residual_bytes = int(entry.get("residual.bin", 0) or 0)
    return {
        "run": run.name,
        "mode": "baseline_multirelation",
        "source_level": source_level,
        "target_level": target_level,
        "node_count": dynamic.get("node_count"),
        "spectral_gap": dynamic.get("spectral_gap"),
        "wishart_clusters": wishart.get("cluster_count"),
        "canonical_types": wishart.get("canonical_type_count"),
        "accepted_occurrences": wishart.get("accepted_occurrences"),
        "dictionary_size": metrics.get("dictionary_size_after_discovery"),
        "reuse_rate": metrics.get("reuse_rate"),
        "recursive_types_total": metrics.get("recursive_types_total"),
        "compression_ratio_binary": codec.get(
            "compression_ratio_binary_baseline"
        ),
        "archive_bytes": archive_bytes or None,
        "residual_bytes": residual_bytes or None,
        "residual_share": (
            residual_bytes / archive_bytes if archive_bytes else None
        ),
        "internal_edge_records": codec.get("internal_edge_records"),
        "port_edge_records": codec.get("port_edge_records"),
        "residual_edge_records": codec.get("residual_edge_records"),
    }


def baseline_run_rows(run_dir: str | Path) -> list[dict[str, object]]:
    run = Path(run_dir)
    rows: list[dict[str, object]] = []
    for transition in sorted(run.glob("transition_*_*")):
        if (transition / "dictionary_metrics.json").is_file():
            rows.append(baseline_transition_row(run, transition))
    return rows


def retrospective_related_relation(
    run_dir: str | Path,
    *,
    grammar_relations: Iterable[str] = ("RelatedTo",),
    report_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    """Re-encode existing accepted figures with a restricted grammar relation set.

    The hierarchy/discovery is not rerun. This is deliberately a codec-only
    ablation: same source level, same accepted figure node sets, new grammar
    relation identity. All excluded relations remain exact residual records.
    """
    run = Path(run_dir)
    relation_tuple = tuple(str(item) for item in grammar_relations)
    if not relation_tuple:
        raise ValueError("grammar_relations must be non-empty")
    reports = Path(report_dir) if report_dir is not None else None
    if reports is not None:
        reports.mkdir(parents=True, exist_ok=True)

    result: list[dict[str, object]] = []
    for transition in sorted(run.glob("transition_*_*")):
        occurrences_path = transition / "figure_occurrences.jsonl"
        if not occurrences_path.is_file():
            continue
        source_level, target_level = transition_levels(transition)
        level_dir = run / f"level_{source_level:03d}"
        report_path = (
            reports
            / f"{run.name}_transition_{source_level:03d}_{target_level:03d}.json"
            if reports is not None
            else None
        )
        if report_path is not None and report_path.is_file():
            row = _load_json(report_path)
            result.append(row)
            continue

        relation_layers = load_relation_layers(level_dir)
        figure_nodes = load_figure_nodes(transition)
        symbol_types = load_symbol_types(level_dir)
        adjacency = sparse.load_npz(level_dir / "adjacency.npz").tocsr()

        with tempfile.TemporaryDirectory(
            prefix=f"semmap-relation-ablation-{source_level:03d}-"
        ) as tmp:
            archive = Path(tmp) / "grammar_exact_v2.zip"
            codec = encode_transition_grammar(
                vertex_count=int(adjacency.shape[0]),
                relation_layers=relation_layers,
                figure_nodes=figure_nodes,
                symbol_types=symbol_types,
                output=archive,
                source_adjacency=adjacency,
                grammar_relations=relation_tuple,
            )

        entry = codec.get("entry_compressed_bytes") or {}
        archive_bytes = int(codec["archive_bytes"])
        residual_bytes = int(entry.get("residual.bin", 0) or 0)
        row = {
            "run": run.name,
            "mode": "retrospective_" + "_".join(relation_tuple),
            "source_level": source_level,
            "target_level": target_level,
            "grammar_relations": list(relation_tuple),
            "same_accepted_figures": True,
            "accepted_occurrences": len(figure_nodes),
            "compression_ratio_binary": codec.get(
                "compression_ratio_binary_baseline"
            ),
            "archive_bytes": archive_bytes,
            "baseline_binary_bytes": codec.get("baseline_binary_bytes"),
            "residual_bytes": residual_bytes,
            "residual_share": (
                residual_bytes / archive_bytes if archive_bytes else None
            ),
            "internal_edge_records": codec.get("internal_edge_records"),
            "port_edge_records": codec.get("port_edge_records"),
            "residual_edge_records": codec.get("residual_edge_records"),
            "grammar_relation_edge_records": codec.get(
                "grammar_relation_edge_records"
            ),
            "non_grammar_relation_edge_records": codec.get(
                "non_grammar_relation_edge_records"
            ),
            "roundtrip_exact": codec.get("roundtrip_exact"),
            "timing_seconds": codec.get("timing_seconds"),
        }
        if report_path is not None:
            report_path.write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        result.append(row)
    return result
