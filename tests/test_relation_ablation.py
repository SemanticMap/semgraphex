"""Tests for persisted relation-subset ablation helpers."""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from semmap_haken.relation_ablation import (
    baseline_run_rows,
    retrospective_related_relation,
)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_retrospective_relatedto_ablation_is_exact(tmp_path):
    run = tmp_path / "run"
    level = run / "level_000"
    relations = level / "relations"
    transition = run / "transition_000_001"
    relations.mkdir(parents=True)
    transition.mkdir(parents=True)

    related = sparse.csr_matrix(
        ([1.0, 2.0], ([0, 1], [1, 2])),
        shape=(4, 4),
    )
    isa = sparse.csr_matrix(
        ([3.0], ([0], [2])),
        shape=(4, 4),
    )
    used = sparse.csr_matrix(
        ([4.0], ([2], [3])),
        shape=(4, 4),
    )
    sparse.save_npz(relations / "RelatedTo.npz", related)
    sparse.save_npz(relations / "IsA.npz", isa)
    sparse.save_npz(relations / "UsedFor.npz", used)
    _write_json(
        relations / "index.json",
        {
            "IsA": "IsA.npz",
            "RelatedTo": "RelatedTo.npz",
            "UsedFor": "UsedFor.npz",
        },
    )
    adjacency = related + isa + used
    sparse.save_npz(level / "adjacency.npz", adjacency)
    (level / "symbolic_nodes.jsonl").write_text(
        "".join(
            json.dumps({"node": i, "dictionary_type_id": None}) + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    _write_json(
        level / "dynamic_metrics.json",
        {"node_count": 4, "spectral_gap": 0.25},
    )
    (transition / "figure_occurrences.jsonl").write_text(
        json.dumps({"fine_nodes": [0, 1, 2]}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        transition / "dictionary_metrics.json",
        {
            "dictionary_size_after_discovery": 1,
            "reuse_rate": 0.0,
            "recursive_types_total": 0,
            "exact_transition_codec": {
                "archive_bytes": 100,
                "compression_ratio_binary_baseline": 1.0,
                "entry_compressed_bytes": {"residual.bin": 50},
            },
        },
    )
    _write_json(
        transition / "wishart.json",
        {
            "cluster_count": 1,
            "canonical_type_count": 1,
            "accepted_occurrences": 1,
        },
    )

    baseline = baseline_run_rows(run)
    assert baseline[0]["residual_share"] == 0.5

    reports = tmp_path / "reports"
    rows = retrospective_related_relation(
        run,
        grammar_relations=("RelatedTo",),
        report_dir=reports,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["roundtrip_exact"] is True
    assert row["internal_edge_records"] == 2
    assert row["port_edge_records"] == 0
    assert row["residual_edge_records"] == 2
    assert row["grammar_relation_edge_records"] == 2
    assert row["non_grammar_relation_edge_records"] == 2
    assert list(reports.glob("*.json"))
