#!/usr/bin/env python3
"""Build a deterministic connected 100k-node English ConceptNet subgraph.

The extractor uses three streaming passes over the official compressed assertions dump.
Pass 1 computes weighted degree and full connected components without retaining edges.
Pass 2 builds adjacency only for a bounded high-degree candidate pool and selects exactly
``target_nodes`` by a deterministic degree-prioritized connected traversal. Pass 3 writes
all eligible original assertions induced by those selected nodes.
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

DEFAULT_RELATIONS = ("RelatedTo", "IsA", "PartOf", "HasA", "UsedFor", "HasProperty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--target-nodes", type=int, default=100_000)
    parser.add_argument("--candidate-multiplier", type=float, default=3.0)
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
    if args.candidate_multiplier < 1.0:
        raise SystemExit("candidate-multiplier must be >= 1")

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

    # Pass 1: degrees + connected components over all eligible English edges.
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

    component_sizes: Counter[int] = Counter()
    for idx in range(len(index_to_node)):
        component_sizes[dsu.find(idx)] += 1
    if not component_sizes:
        raise SystemExit("no eligible ConceptNet edges found")
    largest_root, largest_size = max(component_sizes.items(), key=lambda item: (item[1], -item[0]))
    largest_indices = [idx for idx in range(len(index_to_node)) if dsu.find(idx) == largest_root]
    if len(largest_indices) < args.target_nodes:
        raise SystemExit(f"largest connected component has only {len(largest_indices)} nodes; need {args.target_nodes}")
    largest_indices.sort(key=lambda idx: (-weighted_degree[idx], index_to_node[idx]))

    pool_size = min(len(largest_indices), max(args.target_nodes, int(math.ceil(args.target_nodes * args.candidate_multiplier))))
    pool_global = largest_indices[:pool_size]
    pool_uris = [index_to_node[idx] for idx in pool_global]
    pool_degrees = [weighted_degree[idx] for idx in pool_global]
    pool_lookup = {uri: idx for idx, uri in enumerate(pool_uris)}

    # Release full-graph indexing before the bounded candidate adjacency is created.
    del node_to_index, index_to_node, weighted_degree, largest_indices, pool_global, dsu, component_sizes

    # Pass 2: adjacency only inside the high-degree candidate pool.
    pool_adjacency: list[list[int]] = [[] for _ in range(pool_size)]
    candidate_assertions = 0
    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            item = eligible(fields, relations, args.min_weight)
            if item is None:
                continue
            start, end, _relation, _weight = item
            a = pool_lookup.get(start)
            b = pool_lookup.get(end)
            if a is not None and b is not None:
                pool_adjacency[a].append(b)
                pool_adjacency[b].append(a)
                candidate_assertions += 1

    # Find the largest connected component of the candidate-induced graph.
    visited = bytearray(pool_size)
    largest_pool_component: list[int] = []
    for seed in range(pool_size):
        if visited[seed] or not pool_adjacency[seed]:
            continue
        visited[seed] = 1
        queue = deque([seed])
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in pool_adjacency[node]:
                if not visited[neighbour]:
                    visited[neighbour] = 1
                    queue.append(neighbour)
        if len(component) > len(largest_pool_component):
            largest_pool_component = component
    if len(largest_pool_component) < args.target_nodes:
        raise SystemExit(
            f"candidate pool largest induced component has only {len(largest_pool_component)} nodes; "
            f"increase --candidate-multiplier (current {args.candidate_multiplier})"
        )

    component_mask = bytearray(pool_size)
    for idx in largest_pool_component:
        component_mask[idx] = 1

    # Select exactly target_nodes by connected, degree-prioritized growth.
    # Every newly selected node is adjacent to a previously selected node, so the
    # selected induced graph is connected by construction.
    start_idx = min(largest_pool_component, key=lambda idx: (-pool_degrees[idx], pool_uris[idx]))
    selected_mask = bytearray(pool_size)
    queued = bytearray(pool_size)
    selected_order: list[int] = []
    frontier: list[tuple[float, str, int]] = []

    def enqueue(idx: int) -> None:
        if component_mask[idx] and not selected_mask[idx] and not queued[idx]:
            queued[idx] = 1
            heapq.heappush(frontier, (-pool_degrees[idx], pool_uris[idx], idx))

    enqueue(start_idx)
    while frontier and len(selected_order) < args.target_nodes:
        _neg_degree, _uri, node = heapq.heappop(frontier)
        if selected_mask[node]:
            continue
        selected_mask[node] = 1
        selected_order.append(node)
        for neighbour in pool_adjacency[node]:
            enqueue(neighbour)

    if len(selected_order) != args.target_nodes:
        raise SystemExit(f"connected traversal selected only {len(selected_order)} nodes")
    selected = {pool_uris[idx] for idx in selected_order}
    cutoff_degree = min(pool_degrees[idx] for idx in selected_order)
    del pool_lookup, pool_adjacency, visited, component_mask, selected_mask, queued, frontier

    # Pass 3: preserve every original eligible assertion induced by the exact selection.
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
        "schema_version": 2,
        "input": str(source),
        "target_nodes": args.target_nodes,
        "selected_nodes_seen_in_induced_edges": len(seen),
        "missing_selected_nodes": missing[:100],
        "missing_selected_node_count": len(missing),
        "total_dump_lines": total_lines,
        "eligible_english_nodes": len(pool_uris) if False else None,
        "eligible_english_edges": eligible_edges,
        "largest_component_nodes": largest_size,
        "candidate_pool_nodes": pool_size,
        "candidate_pool_assertions": candidate_assertions,
        "candidate_pool_largest_induced_component_nodes": len(largest_pool_component),
        "selected_induced_assertions": written,
        "selected_min_pass1_weighted_degree": cutoff_degree,
        "relations": sorted(relations),
        "eligible_relation_histogram": dict(sorted(relation_counts.items())),
        "selected_relation_histogram": dict(sorted(selected_relation_counts.items())),
        "selection_rule": (
            "largest full eligible component -> top-degree candidate pool -> largest candidate-induced component -> "
            "degree-prioritized connected traversal to exactly target_nodes; degree uses log1p(weight); URI breaks ties"
        ),
        "elapsed_seconds": elapsed,
    }
    # The full eligible node count is not kept after memory release; derive it from the
    # first-pass DSU length before release in future schema revisions if required.
    report.pop("eligible_english_nodes")
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if missing or len(seen) != args.target_nodes:
        raise SystemExit(f"exact-node invariant failed: expected {args.target_nodes}, saw {len(seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
