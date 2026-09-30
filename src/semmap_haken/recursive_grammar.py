"""Consolidate persistent GraphType composition into an acyclic rule DAG."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def build_recursive_rule_rows(types: Mapping[str, object]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for type_id in sorted(types):
        item = types[type_id]
        children = tuple(str(child) for child in getattr(item, "child_types", ()))
        rows.append({
            "symbol_id": str(type_id),
            "shape_id": str(getattr(item, "shape_id", type_id)),
            "interface_variant_id": str(
                getattr(item, "interface_variant_id", "")
            ),
            "children": list(children),
            "first_level": int(getattr(item, "first_level", 0)),
        })

    children_by_symbol = {
        row["symbol_id"]: tuple(row["children"]) for row in rows
    }
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(symbol: str) -> int:
        if symbol in depths:
            return depths[symbol]
        if symbol in visiting:
            raise ValueError("recursive graph grammar contains a cycle")
        visiting.add(symbol)
        child_depths = [
            depth(child)
            for child in children_by_symbol.get(symbol, ())
            if child in children_by_symbol
        ]
        visiting.remove(symbol)
        result = 1 + (max(child_depths) if child_depths else 0)
        depths[symbol] = result
        return result

    for symbol in children_by_symbol:
        depth(symbol)

    referenced = {
        child
        for children in children_by_symbol.values()
        for child in children
        if child in children_by_symbol
    }
    roots = sorted(set(children_by_symbol) - referenced)
    summary = {
        "rules": len(rows),
        "recursive_rules": sum(bool(row["children"]) for row in rows),
        "max_depth": max(depths.values(), default=0),
        "roots": roots,
        "unresolved_child_symbols": sorted({
            child
            for children in children_by_symbol.values()
            for child in children
            if child not in children_by_symbol
        }),
    }
    for row in rows:
        row["depth"] = depths[row["symbol_id"]]
    return rows, summary


def write_recursive_grammar(
    directory: str | Path,
    types: Mapping[str, object],
) -> dict:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    rows, summary = build_recursive_rule_rows(types)
    (target / "recursive_grammar.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    (target / "recursive_grammar_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
