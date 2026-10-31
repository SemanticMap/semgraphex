"""Measured codec-aware pruning of non-overlapping graph-symbol occurrences.

One encoding and one decoder already exist (grammar_codec). Reuse them for
byte-accurate acceptance; analytical cost is only a ranking diagnostic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from semmap_haken.grammar_binary import encoded_uvarint_size
from semmap_haken.grammar_codec import encode_grammar
from semmap_haken.graph_mdl import build_canonical_huffman_codes
from semmap_haken.graphex_components import EdgeRecord


def _cost_snapshot(
    rows: Sequence[tuple[str, tuple[int, ...]]], records: Sequence[EdgeRecord],
    vertex_count: int,
) -> dict:
    """Physical incidences, counted on source assertions rather than CSR nnz."""
    owners = {v: i for i, (_, nodes) in enumerate(rows) for v in nodes}
    if len(owners) != sum(len(nodes) for _, nodes in rows):
        raise ValueError("selected occurrences must not overlap")
    internal = [0] * len(rows)
    ports = [0] * len(rows)
    for edge in records:
        left, right = owners.get(edge.source), owners.get(edge.target)
        if left is not None and left == right:
            internal[left] += 1
        elif left is not None or right is not None:
            ports[min(x for x in (left, right) if x is not None)] += 1
    freq = Counter(key for key, _ in rows)
    codes = build_canonical_huffman_codes(freq)
    dictionary_groups = defaultdict(list)
    for idx, (type_id, nodes) in enumerate(rows):
        dictionary_groups[type_id].append(idx)
    node_bits = 8 * encoded_uvarint_size(max(0, vertex_count - 1))
    edge_bits = 8 * encoded_uvarint_size(max(0, len(records) - 1))
    # Ranking only. Actual ZIP bytes (including rules, ports, weights and
    # Huffman table) decide acceptance below.
    score = {}
    for key, positions in dictionary_groups.items():
        inside = sum(internal[i] for i in positions)
        external = sum(ports[i] for i in positions)
        placement = sum(len(rows[i][1]) for i in positions) * node_bits
        codebits = len(codes[key]) * len(positions)
        # Group rule cost is charged once; even the smallest rule cannot be free.
        rulebits = max(8, len(rows[positions[0]][1]) * node_bits)
        score[key] = (
            inside * (2 * node_bits + edge_bits)
            - (placement + codebits + rulebits + external * node_bits)
        )
    return {
        "internal_records": sum(internal), "port_records": sum(ports),
        "selected_type_counts": dict(sorted(freq.items())),
        "selected_huffman_code_lengths": {k: len(c) for k, c in sorted(codes.items())},
        "actual_port_records_by_type": {
            key: sum(ports[i] for i in positions)
            for key, positions in sorted(dictionary_groups.items())
        },
        "ranking_bits_by_type": score,
    }


def prune_by_archive_bytes(
    *, vertex_count: int, records: Sequence[EdgeRecord],
    candidates: Sequence[tuple[str, tuple[int, ...]]],
    workspace: Path, node_types: Mapping[int, str] | None = None,
    progress_hook=None,
) -> tuple[list[tuple[int, ...]], dict]:
    """Accept a symbol group iff its *measured Huffman ZIP* gets smaller.

    Frequency, internal/port counts and code lengths are recomputed on the
    finally accepted non-overlapping occurrences, not the proposal sample.
    A separate varint archive is always constructed for a fair coding ablation.
    """
    work = Path(workspace)
    groups = defaultdict(list)
    for type_id, nodes in candidates:
        groups[str(type_id)].append(tuple(int(v) for v in nodes))
    initial = [(t, nodes) for t, members in groups.items() for nodes in members]
    initial_cost = _cost_snapshot(initial, records, vertex_count)
    with tempfile.TemporaryDirectory(prefix="semmap-pruning-", dir=work) as tmp:
        tmp = Path(tmp)
        previous = tmp / "baseline.zip"
        zero = encode_grammar(vertex_count, records, (), previous,
                              symbol_coding="huffman", node_types=node_types)
        best_bytes = int(zero["archive_bytes"])
        active = []
        accepted_types = []
        rejected_types = []
        scored = initial_cost["ranking_bits_by_type"]
        ordered = sorted(groups, key=lambda key: (-scored[key], key))
        if progress_hook is not None:
            progress_hook(0, len(ordered))
        for position, type_id in enumerate(ordered, 1):
            trial_rows = active + [(type_id, nodes) for nodes in groups[type_id]]
            archive = tmp / "trial.zip"
            measured = encode_grammar(
                vertex_count, records, [nodes for _, nodes in trial_rows],
                archive, symbol_coding="huffman", node_types=node_types,
            )
            if int(measured["archive_bytes"]) < best_bytes:
                active = trial_rows
                best_bytes = int(measured["archive_bytes"])
                accepted_types.append(type_id)
            else:
                rejected_types.append(type_id)
            if progress_hook is not None:
                progress_hook(position, len(ordered))
        final_cost = _cost_snapshot(active, records, vertex_count)
    return [nodes for _, nodes in active], {
        "candidate_figures": len(initial),
        "candidate_types": len(groups),
        "accepted_figures": len(active),
        "accepted_types": accepted_types,
        "rejected_types": rejected_types,
        "huffman_archive_no_figures_bytes": int(zero["archive_bytes"]),
        "huffman_archive_selected_bytes": best_bytes,
        "measured_saved_vs_empty_grammar_bytes": int(zero["archive_bytes"]) - best_bytes,
        "proposal_exact_counts": initial_cost,
        "accepted_exact_counts": final_cost,
        "evaluation_policy": "monotone exact full ZIP, one group at a time",
    }
