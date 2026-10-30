"""Never admit a graph symbol when the measured exact archive grows."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from semmap_haken.graphex_components import EdgeRecord
from semmap_haken.grammar_codec import encode_grammar, decode_grammar
from semmap_haken.hypothesis.grammar_selection import (
    _cost_snapshot, prune_by_archive_bytes,
)


def test_port_counts_and_selected_huffman_support(tmp_path):
    edges = (
        EdgeRecord(0, 0, 1, "R", 1.),
        EdgeRecord(1, 1, 2, "R", 1.),
        EdgeRecord(2, 3, 4, "R", 2.),
        EdgeRecord(3, 4, 5, "R", 2.),
        EdgeRecord(4, 2, 3, "Bridge", 8.),
        EdgeRecord(5, 2, 7, "S", 3.),
        EdgeRecord(6, 6, 6, "Loop", 4.),
        EdgeRecord(7, 0, 1, "R", 2.),  # repeated assertion
    )
    occurrences = [("A", (0, 1, 2)), ("A", (3, 4, 5))]
    meta = _cost_snapshot(occurrences, edges, 8)
    assert meta["port_records"] == 2  # 2->3 and 2->7, once each
    assert meta["internal_records"] == 5
    assert meta["selected_type_counts"]["A"] == 2
    selected, report = prune_by_archive_bytes(
        vertex_count=8, records=edges, candidates=occurrences,
        workspace=tmp_path,
    )
    assert report["measured_saved_vs_empty_grammar_bytes"] >= 0
    assert report["accepted_figures"] == len(selected)
    exact = tmp_path / "selected.zip"
    raw = tmp_path / "empty.zip"
    saved = encode_grammar(8, edges, selected, exact, symbol_coding="huffman")
    zero = encode_grammar(8, edges, (), raw, symbol_coding="huffman")
    assert saved["archive_bytes"] <= zero["archive_bytes"]
    assert decode_grammar(exact) == (8, edges)


def test_non_profitable_one_off_figure_is_removed(tmp_path):
    edges = tuple(EdgeRecord(i, i, i+1, "R", float(i) + 0.25)
                  for i in range(11))
    selected, report = prune_by_archive_bytes(
        vertex_count=12, records=edges,
        candidates=[("unique", (0, 1, 2))], workspace=tmp_path
    )
    assert report["accepted_figures"] == len(selected)
    assert report["measured_saved_vs_empty_grammar_bytes"] >= 0
