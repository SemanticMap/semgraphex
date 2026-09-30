"""Tests for recursive rule consolidation and grammar-induced graph projection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

from semmap_haken.dictionary_graphex import estimate_dictionary_projection
from semmap_haken.recursive_grammar import build_recursive_rule_rows


@dataclass
class FakeType:
    shape_id: str
    interface_variant_id: str
    child_types: tuple[str, ...]
    first_level: int


def test_recursive_rule_dag_depth_and_roots():
    types = {
        "GT_1": FakeType("GS_1", "IV_1", (), 0),
        "GT_2": FakeType("GS_2", "IV_2", ("GT_1",), 1),
        "GT_3": FakeType("GS_3", "IV_3", ("GT_2", "GT_1"), 2),
    }
    rows, summary = build_recursive_rule_rows(types)
    depth = {row["symbol_id"]: row["depth"] for row in rows}
    assert depth == {"GT_1": 1, "GT_2": 2, "GT_3": 3}
    assert summary["roots"] == ["GT_3"]
    assert summary["max_depth"] == 3
    assert summary["recursive_rules"] == 2


def test_recursive_rule_dag_rejects_cycles():
    types = {
        "GT_1": FakeType("GS_1", "IV_1", ("GT_2",), 0),
        "GT_2": FakeType("GS_2", "IV_2", ("GT_1",), 1),
    }
    with pytest.raises(ValueError, match="cycle"):
        build_recursive_rule_rows(types)


def test_dictionary_projection_uses_original_node_mass_and_relation_blocks():
    r = sparse.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0]),
            (np.array([0, 1, 2]), np.array([1, 2, 3])),
        ),
        shape=(4, 4),
    )
    s = sparse.csr_matrix(
        (
            np.array([0.5]),
            (np.array([3]), np.array([0])),
        ),
        shape=(4, 4),
    )
    assignment = np.array([0, 0, 1, 2], dtype=np.int64)
    projection = estimate_dictionary_projection(
        {"r": r, "s": s},
        assignment,
        {0: "GT_A", 1: "GT_B"},
    )
    assert projection["mass"]["GT_A"]["original_node_count"] == 2
    assert projection["mass"]["GT_A"]["probability_mass"] == 0.5
    assert projection["mass"]["GT_B"]["original_node_count"] == 1
    assert projection["mass"]["ATOM"]["original_node_count"] == 1
    assert projection["graphex_components"]["S"].startswith("not estimated")
    assert any(
        block["relation"] == "r"
        and block["source_symbol"] == "GT_A"
        and block["target_symbol"] == "GT_B"
        for block in projection["blocks"]
    )
