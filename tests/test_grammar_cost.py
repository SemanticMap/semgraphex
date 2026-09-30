"""Tests for grammar-v2 logical cost used during figure selection."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from semmap_haken.grammar_cost import estimate_grammar_occurrence_cost
from semmap_haken.wishart_metrics import EgoCandidate


def _candidate(boundary_count: int) -> EgoCandidate:
    adjacency = sparse.csr_matrix(
        (
            np.ones(2),
            (np.array([0, 1]), np.array([1, 0])),
        ),
        shape=(2, 2),
    )
    boundary = ()
    if boundary_count:
        boundary = ((0, "r", "out", boundary_count),)
    return EgoCandidate(
        center=0,
        nodes=np.array([10, 20]),
        adjacency=adjacency,
        relation_layers={"r": adjacency},
        boundary_signature=boundary,
    )


def test_port_heavy_occurrence_cost_is_penalized():
    compact = estimate_grammar_occurrence_cost(
        _candidate(0),
        occurrence_nodes=(10, 20),
        support=20,
        graph_node_count=1000,
        relation_count=4,
        edge_record_count=5000,
        type_code_bits=3,
    )
    leaky = estimate_grammar_occurrence_cost(
        _candidate(8),
        occurrence_nodes=(10, 20),
        support=20,
        graph_node_count=1000,
        relation_count=4,
        edge_record_count=5000,
        type_code_bits=3,
    )
    assert compact.port_payload_bits == 0
    assert leaky.port_payload_bits > 0
    assert leaky.encoded_bits > compact.encoded_bits


def test_rule_cost_is_amortized_by_support():
    low_support = estimate_grammar_occurrence_cost(
        _candidate(0),
        occurrence_nodes=(10, 20),
        support=1,
        graph_node_count=1000,
        relation_count=4,
        edge_record_count=5000,
        type_code_bits=3,
    )
    high_support = estimate_grammar_occurrence_cost(
        _candidate(0),
        occurrence_nodes=(10, 20),
        support=100,
        graph_node_count=1000,
        relation_count=4,
        edge_record_count=5000,
        type_code_bits=3,
    )
    assert high_support.rule_bits < low_support.rule_bits
    assert high_support.encoded_bits < low_support.encoded_bits
