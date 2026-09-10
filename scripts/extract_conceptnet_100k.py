#!/usr/bin/env python3
"""Build a deterministic 100k-node English ConceptNet subgraph from the official dump.

The extractor makes two streaming passes over the compressed ConceptNet assertions file.
Pass 1 keeps only requested English relations, computes weighted degree and connected
components without materializing edges. Pass 2 writes only assertions whose endpoints
belong to the selected top-degree nodes of the largest connected component.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

DEFAULT_RELATIONS = ("RelatedTo", "IsA", "PartOf", "HasA", "UsedFor", "HasProperty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--target-nodes", type=int, default=100_000)
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--relations", nargs="+", default=list(DEFAULT_RELATIONS))
    return parser.parse_args()


def relation_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def eligible(fields: list[str], relations: set[str], min_weight: float) -> tuple[str, str, str, float] | None:
    if len(fields) != 5:
        return None
    relation = relation_name(fields[1])
    start, end = fields[2], fields[3]
    if relation not in relations or not start.startswith("/c/en/") or not end.startswith("/c/en/") or start == end:
        return None
    try:
        metadata = json.loads(fields[4])
        weight = metadata.get("weight") if isinstance(metadata, dict) else None
    except json.JSONDecodeError:
        return None
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        return None
    weight = float(weight)
    if not math.isfinite(weight) or weight < min_weight:
        return None
    return start, end, relation, weight


class DSU:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    metadata_path = Path(args.metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    relations = set(args.relations)
    if args.target_nodes < 3:
        raise SystemExit("target-nodes must be >= 3")

    node_to_index: dict[str, int] = {}
    index_to_node: list[str] = []
    weighted_degree: list[float] = []
    dsu = DSU()
    relation_counts = Counter()
    total_lines = eligible_edges = 0
    started = time.perf_counter()

    def node_index(uri: str) -> int:
        existing = node_to_index.get(uri)
        if existing is not None:
            return existing
        idx = dsu.add()
        node_to_index[uri] = idx
        index_to_node.append(uri)
        weighted_degree.append(0.0)
        return idx

    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            total_lines += 1
            fields = line.rstrip("\n").split("\t")
            item = eligible(fields, relations, args.min_weight)
            if item is None:
                continue
            start, end, relation, weight = item
            a, b = node_index(start), node_index(end)
            contribution = math.log1p(weight)
            weighted_degree[a] += contribution
            weighted_degree[b] += contribution
            dsu.union(a, b)
            relation_counts[relation] += 1
            eligible_edges += 1
            if total_lines % 5_000_000 == 0:
                print(json.dumps({"phase": "pass1", "lines": total_lines, "eligible_edges": eligible_edges, "nodes": len(index_to_node)}), flush=True)

    component_sizes: Counter[int] = Counter()
    for idx in range(len(index_to_node)):
        component_sizes[dsu.find(idx)] += 1
    if not component_sizes:
        raise SystemExit("no eligible ConceptNet edges found")
    largest_root, largest_size = max(component_sizes.items(), key=lambda item: (item[1], -item[0]))
    candidates = [idx for idx in range(len(index_to_node)) if dsu.find(idx) == largest_root]
    if len(candidates) < args.target_nodes:
        raise SystemExit(f"largest connected component has only {len(candidates)} nodes; need {args.target_nodes}")
    candidates.sort(key=lambda idx: (-weighted_degree[idx], index_to_node[idx]))
    selected_indices = candidates[: args.target_nodes]
    selected = {index_to_node[idx] for idx in selected_indices}
    cutoff_degree = weighted_degree[selected_indices[-1]]

    written = 0
    seen: set[str] = set()
    selected_relation_counts = Counter()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream, output.open("wt", encoding="utf-8", newline="") as sink:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            item = eligible(fields, relations, args.min_weight)
            if item is None:
                continue
            start, end, relation, _weight = item
            if start in selected and end in selected:
                sink.write(line if line.endswith("\n") else line + "\n")
                seen.add(start)
                seen.add(end)
                selected_relation_counts[relation] += 1
                written += 1

    missing = sorted(selected - seen)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "input": str(source),
        "target_nodes": args.target_nodes,
        "selected_nodes_seen_in_induced_edges": len(seen),
        "missing_selected_nodes": missing[:100],
        "missing_selected_node_count": len(missing),
        "total_dump_lines": total_lines,
        "eligible_english_nodes": len(index_to_node),
        "eligible_english_edges": eligible_edges,
        "largest_component_nodes": largest_size,
        "selected_induced_assertions": written,
        "weighted_degree_cutoff": cutoff_degree,
        "relations": sorted(relations),
        "eligible_relation_histogram": dict(sorted(relation_counts.items())),
        "selected_relation_histogram": dict(sorted(selected_relation_counts.items())),
        "selection_rule": "top weighted-degree nodes inside largest connected component; degree uses log1p(weight); URI ascending breaks ties",
        "elapsed_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if missing:
        raise SystemExit(f"induced subgraph lost {len(missing)} selected nodes; refusing to label it {args.target_nodes}-node")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
