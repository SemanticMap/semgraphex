"""Tests for exact transition archives emitted before Wishart contraction."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from semmap_haken.grammar_codec import decode_grammar
from semmap_haken.transition_grammar import (
    encode_transition_grammar,
    relation_layers_to_edge_records,
)


def test_relation_layers_roundtrip_through_transition_grammar(tmp_path):
    related = sparse.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            (np.array([0, 1, 2, 3]), np.array([1, 2, 4, 4])),
        ),
        shape=(5, 5),
    )
    isa = sparse.csr_matrix(
        (
            np.array([0.5, 0.75]),
            (np.array([2, 4]), np.array([0, 3])),
        ),
        shape=(5, 5),
    )
    layers = {"RelatedTo": related, "IsA": isa}
    records = relation_layers_to_edge_records(layers)

    archive = tmp_path / "transition.zip"
    report = encode_transition_grammar(
        vertex_count=5,
        relation_layers=layers,
        figure_nodes=((0, 1, 2),),
        symbol_types={1: "GT_CHILD"},
        output=archive,
        source_adjacency=related + isa,
    )

    assert report["roundtrip_exact"]
    assert report["source_relation_layers"] == 2
    assert report["adjacency_from_relations"]["matches"]
    assert report["port_edge_records"] >= 1
    assert decode_grammar(archive) == (5, records)
