#!/usr/bin/env python3
"""Build a deterministic connected 100k-node English ConceptNet subgraph.

The extractor uses three streaming passes over a compressed or decompressed official
assertions dump.
Pass 1 computes weighted degree and full connected components without retaining edges.
Pass 2 builds adjacency only for a bounded high-degree candidate pool and selects exactly
``target_nodes`` by a deterministic degree-prioritized connected traversal. Pass 3 writes
all eligible original assertions induced by those selected nodes.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
import gzip
import heapq
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Sequence, TextIO, TypeVar

DEFAULT_RELATIONS = ("RelatedTo", "IsA", "PartOf", "HasA", "UsedFor", "HasProperty")
EligibleEdge = tuple[str, str, str, float]
T = TypeVar("T")

_WORKER_RELATIONS: set[str] = set()
_WORKER_MIN_WEIGHT = 1.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to the gzip-compressed ConceptNet assertions dump")
    input_group.add_argument(
        "--decompressed-input",
        help="Path to an already-decompressed ConceptNet assertions file",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--target-nodes", type=int, default=100_000)
    parser.add_argument("--candidate-multiplier", type=float, default=3.0)
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--relations", nargs="+", default=list(DEFAULT_RELATIONS))
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Number of ordered parser worker processes",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=10_000,
        help="Input lines sent to each parser task",
    )
    return parser.parse_args(argv)


def open_source(path: Path, *, compressed: bool) -> TextIO:
    """Open one repeatable streaming pass over compressed or plain assertions."""
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def relation_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def eligible(fields: list[str], relations: set[str], min_weight: float) -> EligibleEdge | None:
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


def initialize_parser_worker(relations: tuple[str, ...], min_weight: float) -> None:
    """Initialize immutable parser settings once in each worker process."""
    global _WORKER_RELATIONS, _WORKER_MIN_WEIGHT
    _WORKER_RELATIONS = set(relations)
    _WORKER_MIN_WEIGHT = min_weight


def parse_batch(lines: list[str]) -> tuple[int, list[EligibleEdge]]:
    """Parse one batch, returning eligible edges in their original order."""
    parsed: list[EligibleEdge] = []
    for line in lines:
        item = eligible(line.rstrip("\n").split("\t"), _WORKER_RELATIONS, _WORKER_MIN_WEIGHT)
        if item is not None:
            parsed.append(item)
    return len(lines), parsed


def parse_batch_with_lines(lines: list[str]) -> tuple[int, list[tuple[str, EligibleEdge]]]:
    """Parse one batch and retain source text for deterministic output."""
    parsed: list[tuple[str, EligibleEdge]] = []
    for line in lines:
        item = eligible(line.rstrip("\n").split("\t"), _WORKER_RELATIONS, _WORKER_MIN_WEIGHT)
        if item is not None:
            parsed.append((line, item))
    return len(lines), parsed


def line_batches(stream: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    """Yield bounded batches without loading the complete source into memory."""
    batch: list[str] = []
    for line in stream:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ordered_bounded_map(
    executor: ProcessPoolExecutor,
    function: Callable[[list[str]], T],
    batches: Iterable[list[str]],
    *,
    max_pending: int,
) -> Iterator[T]:
    """Map batches in order while bounding submitted work and IPC memory."""
    pending: deque[Future[T]] = deque()
    for batch in batches:
        pending.append(executor.submit(function, batch))
        if len(pending) >= max_pending:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def parsed_batches(
    stream: Iterable[str],
    *,
    batch_size: int,
    executor: ProcessPoolExecutor | None,
    workers: int,
    preserve_lines: bool = False,
) -> Iterator[tuple[int, list[Any]]]:
    """Parse batches serially or through an order-preserving process pool."""
    function = parse_batch_with_lines if preserve_lines else parse_batch
    batches = line_batches(stream, batch_size)
    if executor is None:
        yield from map(function, batches)
        return
    yield from ordered_bounded_map(executor, function, batches, max_pending=workers * 2)


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
    compressed_input = args.input is not None
    source = Path(args.input if compressed_input else args.decompressed_input)
    output = Path(args.output)
    metadata_path = Path(args.metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    relations = set(args.relations)
    if args.target_nodes < 3:
        raise SystemExit("target-nodes must be >= 3")
    if args.candidate_multiplier < 1.0:
        raise SystemExit("candidate-multiplier must be >= 1")

    initialize_parser_worker(tuple(sorted(relations)), args.min_weight)
    executor_context: ContextManager[ProcessPoolExecutor | None]
    if args.workers == 1:
        executor_context = nullcontext(None)
    else:
        executor_context = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_parser_worker,
            initargs=(tuple(sorted(relations)), args.min_weight),
        )

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

    with executor_context as executor:
        # Pass 1: degrees + connected components over all eligible English edges.
        with open_source(source, compressed=compressed_input) as stream:
            for line_count, items in parsed_batches(
                stream, batch_size=args.batch_size, executor=executor, workers=args.workers
            ):
                total_lines += line_count
                for start, end, relation, weight in items:
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

        pool_size = min(
            len(largest_indices),
            max(args.target_nodes, int(math.ceil(args.target_nodes * args.candidate_multiplier))),
        )
        pool_global = largest_indices[:pool_size]
        pool_uris = [index_to_node[idx] for idx in pool_global]
        pool_degrees = [weighted_degree[idx] for idx in pool_global]
        pool_lookup = {uri: idx for idx, uri in enumerate(pool_uris)}

        # Release full-graph indexing before the bounded candidate adjacency is created.
        del node_to_index, index_to_node, weighted_degree, largest_indices, pool_global, dsu, component_sizes

        # Pass 2: adjacency only inside the high-degree candidate pool.
        pool_adjacency: list[list[int]] = [[] for _ in range(pool_size)]
        candidate_assertions = 0
        with open_source(source, compressed=compressed_input) as stream:
            for _line_count, items in parsed_batches(
                stream, batch_size=args.batch_size, executor=executor, workers=args.workers
            ):
                for start, end, _relation, _weight in items:
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
        with open_source(source, compressed=compressed_input) as stream, output.open(
            "wt", encoding="utf-8", newline=""
        ) as sink:
            for _line_count, entries in parsed_batches(
                stream,
                batch_size=args.batch_size,
                executor=executor,
                workers=args.workers,
                preserve_lines=True,
            ):
                for line, (start, end, relation, _weight) in entries:
                    if start in selected and end in selected:
                        sink.write(line if line.endswith("\n") else line + "\n")
                        seen.add(start)
                        seen.add(end)
                        selected_relation_counts[relation] += 1
                        written += 1

    missing = sorted(selected - seen)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 3,
        "input": str(source),
        "input_compression": "gzip" if compressed_input else "none",
        "workers": args.workers,
        "batch_size": args.batch_size,
        "parallel_strategy": "ordered_batched_process_pool" if args.workers > 1 else "single_process",
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
