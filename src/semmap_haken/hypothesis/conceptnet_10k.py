"""Bounded reproducible relation-aware 10k-node ConceptNet induced sample.

A stratified multi-source BFS improves coverage of a *chosen* collection of
relation types. It is NOT an unbiased vertex sample. Null-model results are
conditional on this selection procedure and must be interpreted accordingly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from itertools import zip_longest
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from semmap_haken.conceptnet import Assertion, AssertionFilters, stream_assertions

DEFAULT_RELATIONS = (
    "RelatedTo", "IsA", "PartOf", "UsedFor",
    "HasProperty", "HasA", "Synonym", "Causes", "Antonym",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_conceptnet_sample(
    source: str | Path,
    destination: str | Path,
    *,
    nodes: int = 10_000,
    relations: tuple[str, ...] = DEFAULT_RELATIONS,
    language: str = "en",
    seed: int = 42,
    max_assertions: int = 2_000_000,
    min_weight: float = 0.0,
) -> dict[str, object]:
    """Construct exactly \`nodes\` selected original vertices, or fail.

    Input must be a genuine ConceptNet five-column assertion TSV (.gz and
    .bz2 supported by the existing parser), not aggregated JSONL. Full
    qualifying input is read twice; max_assertions is a *fail-fast safety cap*,
    NEVER a silent prefix sample.
    """
    src, dest = Path(source), Path(destination)
    if src.resolve() == dest.resolve():
        raise ValueError("source and destination must differ")
    if nodes < 2 or max_assertions < 1:
        raise ValueError("nodes >=2 and max_assertions >=1 required")
    if not relations or len(set(relations)) != len(relations):
        raise ValueError("unique, nonempty relation names required")
    if not src.is_file():
        raise FileNotFoundError(src)
    if dest.exists():
        raise FileExistsError(f"destination already exists: {dest}")
    filters = AssertionFilters(language=language, relations=frozenset(relations),
                               min_weight=min_weight)

    node_to_id: dict[str, int] = {}
    names: list[str] = []
    adjacency: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    degree = Counter()
    degree_by_relation: dict[str, Counter[int]] = {r: Counter() for r in relations}
    input_counts = Counter()
    accepted = 0
    loops = 0

    def index(uri: str) -> int:
        existing = node_to_id.get(uri)
        if existing is not None:
            return existing
        number = len(names)
        node_to_id[uri] = number
        names.append(uri)
        return number

    for assertion in stream_assertions(src, filters):
        accepted += 1
        if accepted > max_assertions:
            raise ValueError(
                f"filtered assertions exceed max_assertions={max_assertions}; "
                "provide a bounded, prefiltered ConceptNet assertions TSV"
            )
        if assertion.start_uri == assertion.end_uri:
            loops += 1
            continue
        u = index(assertion.start_uri)
        v = index(assertion.end_uri)
        rel = assertion.relation_name
        degree[u] += 1
        degree[v] += 1
        degree_by_relation[rel][u] += 1
        degree_by_relation[rel][v] += 1
        input_counts[rel] += 1
        adjacency[u][rel].add(v)
        adjacency[v][rel].add(u)

    if len(names) < nodes:
        raise ValueError(f"only {len(names)} available vertices; requested {nodes}")
    unavailable = [r for r in relations if not input_counts[r]]
    if unavailable:
        raise ValueError(f"requested relation types are absent: {unavailable}")
    if nodes < 2 * len(relations):
        raise ValueError("nodes must be >= 2 * requested relation types to preserve seed edges")

    rng = np.random.default_rng(seed)
    ties = rng.random(len(names))
    ordered_rel = list(relations)
    selected: set[int] = set()
    queue: deque[int] = deque()

    def include(node: int) -> None:
        if node not in selected and len(selected) < nodes:
            selected.add(node)
            queue.append(node)

    # Seed a real edge of every chosen type; this prevents a dominant relation
    # from crowding a rarer relation out of the induced graph.
    for rel in ordered_rel:
        seed_vertex = min(degree_by_relation[rel],
                          key=lambda i: (-degree_by_relation[rel][i], ties[i], names[i]))
        neigh = min(adjacency[seed_vertex][rel],
                    key=lambda i: (-degree[i], ties[i], names[i]))
        include(seed_vertex)
        include(neigh)

    # Round-robin incident relation layers for each expanded vertex. Each BFS
    # expansion draws at most one neighbor of each relation per rotation.
    # Input order is NOT used for vertex ranking.
    while len(selected) < nodes:
        if not queue:
            remainder = (i for i in range(len(names)) if i not in selected)
            root = min(remainder, key=lambda i: (-degree[i], ties[i], names[i]))
            include(root)
        u = queue.popleft()
        if u not in adjacency:
            continue
        rotated = ordered_rel[u % len(ordered_rel):] + ordered_rel[:u % len(ordered_rel)]
        neighbors = [
            sorted(adjacency[u].get(rel, ()),
                   key=lambda v: (-degree[v], ties[v], names[v]))
            for rel in rotated
        ]
        for round_neighbors in zip_longest(*neighbors):
            for v in round_neighbors:
                if v is not None:
                    include(v)
                if len(selected) >= nodes:
                    break
            if len(selected) >= nodes:
                break

    selected_names = {names[i] for i in selected}
    dest.parent.mkdir(parents=True, exist_ok=True)
    pending = dest.with_name(dest.name + ".partial")
    counts = Counter()
    observed_nodes: set[str] = set()
    kept = 0
    try:
        with pending.open("w", encoding="utf-8") as target:
            for assertion in stream_assertions(src, filters):
                if (assertion.start_uri not in selected_names or
                    assertion.end_uri not in selected_names or
                    assertion.start_uri == assertion.end_uri):
                    continue
                metadata = {"weight": assertion.weight}
                if assertion.dataset is not None:
                    metadata["dataset"] = assertion.dataset
                if assertion.sources is not None:
                    metadata["sources"] = assertion.sources
                if assertion.license is not None:
                    metadata["license"] = assertion.license
                target.write("\t".join((
                    assertion.assertion_uri, assertion.relation_uri,
                    assertion.start_uri, assertion.end_uri,
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
                )) + "\n")
                counts[assertion.relation_name] += 1
                kept += 1
                observed_nodes.update((assertion.start_uri, assertion.end_uri))
        if len(observed_nodes) != nodes:
            raise ValueError(
                f"selection induced only {len(observed_nodes)} non-isolated vertices "
                f"out of {nodes}; change relation filters or seed"
            )
        if any(counts[r] == 0 for r in relations):
            raise ValueError("relation coverage vanished after induced node selection")
        os.replace(pending, dest)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise

    result: dict[str, object] = {
        "selection_policy": "relation_stratified_multisource_bfs_induced_v1",
        "source_path": str(src),
        "source_sha256": _hash(src),
        "output_path": str(dest),
        "sha256": _hash(dest),
        "seed": int(seed), "language": language,
        "min_weight": float(min_weight),
        "selected_nodes": int(nodes),
        "available_vertices": len(names),
        "eligible_assertions": int(accepted - loops),
        "excluded_loops": int(loops),
        "selected_assertions": int(kept),
        "relations_requested": list(relations),
        "by_relation": {r: int(counts[r]) for r in relations},
        "source_by_relation": {r: int(input_counts[r]) for r in relations},
        "max_assertions": max_assertions,
        "sampling_warning": (
            "Relation-stratified, degree-seeded BFS induces a topology-dependent "
            "sampling bias; this is NOT a uniform random sample of ConceptNet."
        ),
    }
    report_path = dest.with_name(dest.name + ".selection.json")
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create exact-node relation-stratified ConceptNet assertion sample")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--relations", default=",".join(DEFAULT_RELATIONS),
                        help="comma-separated ConceptNet relation types")
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-assertions", type=int, default=2_000_000)
    parser.add_argument("--min-weight", type=float, default=0.)
    args = parser.parse_args(argv)
    rels = tuple(x.strip() for x in args.relations.split(",") if x.strip())
    result = prepare_conceptnet_sample(
        args.input, args.output, nodes=args.nodes, relations=rels,
        language=args.language, seed=args.seed,
        max_assertions=args.max_assertions, min_weight=args.min_weight
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
