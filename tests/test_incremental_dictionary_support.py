"""Tests for incremental occurrence indexing and adjacency diagnostics."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from semmap_haken.occurrence_index import IndexedOccurrence, OccurrenceIndex
from semmap_haken.relation_adjacency import (
    adjacency_relation_diagnostic,
    sum_relation_layers,
)


def test_occurrence_index_invalidates_only_touching_rows():
    index = OccurrenceIndex.from_rows(
        (
            IndexedOccurrence(1, "GT_A", 0, (0, 1, 2)),
            IndexedOccurrence(2, "GT_A", 3, (3, 4)),
            IndexedOccurrence(3, "GT_B", 5, (5, 6)),
        )
    )
    removed = index.invalidate_nodes((1, 6))
    assert {row.occurrence_id for row in removed} == {1, 3}
    assert [row.occurrence_id for row in index.occurrences_for_type("GT_A")] == [2]
    assert 1 not in index.by_node
    assert 6 not in index.by_node


def test_affected_centers_are_local_radius_support():
    adjacency = sparse.csr_matrix(
        (
            np.ones(6),
            (
                np.array([0, 1, 1, 2, 2, 3]),
                np.array([1, 0, 2, 1, 3, 2]),
            ),
        ),
        shape=(5, 5),
    )
    assert OccurrenceIndex.affected_centers(
        adjacency, (1,), radius=0
    ) == (1,)
    assert OccurrenceIndex.affected_centers(
        adjacency, (1,), radius=1
    ) == (0, 1, 2)
    assert OccurrenceIndex.affected_centers(
        adjacency, (1,), radius=2
    ) == (0, 1, 2, 3)


def test_relation_layers_can_exactly_reconstruct_adjacency():
    a = sparse.csr_matrix(
        (
            np.array([1.0, 2.0]),
            (np.array([0, 1]), np.array([1, 2])),
        ),
        shape=(3, 3),
    )
    b = sparse.csr_matrix(
        (
            np.array([0.5]),
            (np.array([2]), np.array([0])),
        ),
        shape=(3, 3),
    )
    adjacency = sum_relation_layers({"a": a, "b": b})
    report = adjacency_relation_diagnostic(adjacency, {"a": a, "b": b})
    assert report["matches"]
    assert report["max_abs_error"] == 0.0


def test_relation_adjacency_diagnostic_detects_missing_layer_mass():
    a = sparse.csr_matrix(([1.0], ([0], [1])), shape=(2, 2))
    adjacency = sparse.csr_matrix(([2.0], ([0], [1])), shape=(2, 2))
    report = adjacency_relation_diagnostic(adjacency, {"a": a})
    assert not report["matches"]
    assert report["max_abs_error"] == 1.0
