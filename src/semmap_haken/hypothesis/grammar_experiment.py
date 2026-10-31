"""nSBM-guided grammar experiment; reuse H1 and exact grammar v2, no new codec.

Graph-tool fits run on CPU/OpenMP. CuPy is optional for degree-ranking and W
projection, not for nSBM/VF2. All claims refer to measured exact archive bytes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from collections import Counter
import io
import json
import os
from pathlib import Path
import time
import zipfile
from typing import Sequence

import numpy as np
from scipy import sparse

from semmap_haken.conceptnet import stream_assertions
from semmap_haken.exact_edge_baseline import encode_edge_baseline
from semmap_haken.grammar_codec import decode_grammar, encode_grammar
from semmap_haken.grammar_cost import estimate_grammar_occurrence_cost
from semmap_haken.graph_dictionary import GraphDictionary
from semmap_haken.graph_mdl import (MdlOccurrence, build_canonical_huffman_codes,
                                   select_nonoverlapping_mdl)
from semmap_haken.graphex_components import EdgeRecord
from semmap_haken.shape_identity import LEVELS
from semmap_haken.wishart_metrics import EgoExtractor, typed_wl_features, _knn_from_features
from semmap_haken.wishart_cluster import wishart_cluster
from semmap_haken.wishart_parallel import (
    initialize_match_worker, match_chunk, spawn_pool,
)

from .graph import TypedGraph
from .grammar_selection import prune_by_archive_bytes
from .scan_checkpoint import (
    atomic_json, load_shard, save_shard, sha256_file, verify_artifacts,
)
from .projection import estimate_block_kernel
from .runner import (
    _available_cpus, _best_fit, _git_revision, _parallel_fit,
    _save_graph, _sha_bytes, _worker_budget, _available_memory_bytes,
)

PROFILE = {
    "10k": {"vertices": 10_000, "discovery": 800, "max_figures": 350, "batch": 384},
    "100k": {"vertices": 100_000, "discovery": 3000, "max_figures": 1200, "batch": 512},
    "smoke": {"vertices": None, "discovery": 50, "max_figures": 20, "batch": 64},
}


def edge_records(graph: TypedGraph) -> tuple[EdgeRecord, ...]:
    """One distinct record per assertion, never aggregate parallel triples."""
    return tuple(
        EdgeRecord(i, int(u), int(v), graph.relation_names[int(r)], float(w))
        for i, (u, v, r, w) in enumerate(
            zip(graph.source, graph.target, graph.relation, graph.weight)
        )
    )


# One atomic writer for reports and checkpoints.
_write_json = atomic_json

def _degree(graph: TypedGraph, device: str) -> tuple[np.ndarray, str]:
    """Compute directed-incidence counts; seed-based sampling remains on CPU."""
    n = graph.vertex_count
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("invalid device")
    if device != "cpu":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            degree = (
                cp.bincount(cp.asarray(graph.source), minlength=n) +
                cp.bincount(cp.asarray(graph.target), minlength=n)
            )
            return cp.asnumpy(degree).astype(np.int64), "cuda"
        except Exception:
            if device == "cuda":
                raise
    return (
        np.bincount(graph.source, minlength=n) +
        np.bincount(graph.target, minlength=n)
    ).astype(np.int64), "cpu"


def _views(graph: TypedGraph, labels: np.ndarray | None,
           full_view: EgoExtractor | None = None) -> dict[str, EgoExtractor]:
    n = graph.vertex_count
    row = np.asarray(graph.source, dtype=np.int64)
    col = np.asarray(graph.target, dtype=np.int64)

    def build(selected: np.ndarray) -> EgoExtractor:
        u, v, rel = row[selected], col[selected], graph.relation[selected]
        layers = {}
        for r, name in enumerate(graph.relation_names):
            mask = rel == r
            layers[name] = sparse.csr_matrix(
                (np.ones(int(mask.sum()), dtype=np.int32), (u[mask], v[mask])),
                shape=(n, n),
            )
            layers[name].sum_duplicates()
        support = sparse.csr_matrix(
            (np.ones(len(u), dtype=np.int32), (u, v)), shape=(n, n)
        )
        # Undirected support is used only for BFS; labels and arrow direction
        # remain in the relation layers.
        support = (support + support.T).tocsr()
        if support.nnz:
            support.data[:] = 1
        return EgoExtractor(support, layers)

    all_edges = np.ones(graph.edge_count, dtype=bool)
    result = {"full": full_view if full_view is not None else build(all_edges)}
    if labels is not None:
        result["within"] = build(labels[row] == labels[col])
    return result


def _centers(
    graph: TypedGraph, labels: np.ndarray | None, budget: int,
    degree: np.ndarray, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = graph.vertex_count
    budget = min(int(budget), n)
    if labels is None:
        return np.sort(rng.choice(n, size=budget, replace=False))
    labels = np.asarray(labels, dtype=np.int64)
    _, inverse = np.unique(labels, return_inverse=True)
    block_count = int(inverse.max()) + 1
    selected: set[int] = set()
    # Balanced quota: do not let the huge RelatedTo block monopolize discovery.
    each = max(1, budget // max(1, block_count))
    for block in range(block_count):
        members = np.flatnonzero(inverse == block)
        if not members.size:
            continue
        draw = min(each, len(members), budget - len(selected))
        if draw <= 0:
            break
        # Favor moderate degrees: huge hubs create expensive port interfaces.
        weights = 1.0 / (1.0 + np.abs(np.log1p(degree[members]) - np.log(5.0)))
        weights = weights / weights.sum()
        selected.update(map(int, rng.choice(members, size=draw, replace=False,
                                             p=weights)))
    if len(selected) < budget:
        pool = np.array(sorted(set(range(n)) - selected), dtype=np.int64)
        extra = rng.choice(pool, size=budget - len(selected), replace=False)
        selected.update(map(int, extra))
    return np.asarray(sorted(selected), dtype=np.int64)


def _fit_partition(
    graph: TypedGraph, out: Path, *, restarts: int, seed: int, cpu_budget: int,
    progress_hook=None,
) -> tuple[np.ndarray, dict]:
    """Use H1's checkpointed spawn/process pool; fit the *full* source graph."""
    stage = out / "fit_data"
    if not (stage / "ids.json").is_file():
        _save_graph(graph, stage)
    workers, _ = _worker_budget(cpu_budget, vertices=graph.vertex_count,
                                edges=graph.edge_count, tasks=restarts)
    threads = max(1, cpu_budget // workers)
    paths = [out / "tasks" / f"compression_nested_restart_{i:03d}.json"
             for i in range(restarts)]
    tasks = [("nested-DC-SBM", seed + i, path)
             for i, path in enumerate(paths)]
    results = _parallel_fit(stage, tasks, workers=workers, threads=threads,
                            progress_hook=progress_hook)
    best, partitions = _best_fit(results, paths)
    informative = [np.asarray(p, dtype=np.int32) for p in partitions
                   if 1 < len(set(p)) < graph.vertex_count]
    if not informative:
        raise RuntimeError("nested DC-SBM yielded no informative partition")
    np.savez_compressed(out / "nsbm_partitions.npz",
                        **{f"level_{i:03d}": p for i, p in enumerate(informative)})
    return informative, {
        "best_mdl_bits": best["mdl_bits"],
        "all_mdl_bits": [float(x["mdl_bits"]) for x in results],
        "blocks_by_level": [len(set(p)) for p in partitions],
        "workers": workers, "threads_per_fit": threads,
        "selected_seed": best["seed"], "source": "full-graph nested DC-SBM",
        "restart_wall_seconds": [float(x["wall_seconds"]) for x in results],
    }


def _discover(
    graph: TypedGraph, views: dict[str, EgoExtractor],
    labels: np.ndarray | None, degree: np.ndarray, *,
    budget: int, max_ego_nodes: int, min_figure_nodes: int,
    seed: int, extraction_workers: int,
    node_types: dict[int, str] | None = None,
) -> GraphDictionary:
    dictionary = GraphDictionary(boundary_sensitive=False)
    centers = _centers(graph, labels, budget, degree, seed)
    # Use the full view plus within-block view only when we have nSBM labels.
    for view in views.values():
        rows = view.extract_centers(
            centers, radius=1, max_ego_nodes=max_ego_nodes,
            symbol_types=node_types, workers=extraction_workers,
        )
        for candidate in rows:
            if len(candidate.nodes) >= min_figure_nodes:
                dictionary.resolve_or_create(candidate, level=0)
    return dictionary


def _vf2_cpu_plan(cpu_budget: int, dictionary: GraphDictionary) -> tuple[int, int]:
    """One CPU budget across extraction and matching, capped by free RAM."""
    cpu = max(1, min(cpu_budget, _available_cpus()))
    if cpu == 1:
        return 1, 1
    extraction = max(1, cpu // 3)
    # Spawn replicates the (NetworkX) dictionary. Be conservative on Colab.
    bytes_per_worker = max(256 * (1 << 20), len(dictionary.types) * (128 << 10))
    memory_workers = max(1, int(_available_memory_bytes() * .5 // bytes_per_worker))
    matching = max(1, min(cpu - extraction, 8, memory_workers))
    return extraction, matching


def _scan(
    views: dict[str, EgoExtractor], dictionary: GraphDictionary, *,
    n: int, batch: int, max_ego_nodes: int, min_figure_nodes: int,
    extraction_workers: int, match_workers: int,
    checkpoint_dir: Path, identity: dict, checkpoint_hook=None,
    node_types: dict[int, str] | None = None, progress_hook=None,
) -> tuple[list[tuple[str, tuple[int, ...]]], Counter, dict]:
    """Bounded, ordered extraction/VF2 pipeline with verified 4k-center shards.

    One future extracts the NEXT batch while the current batch is matched in
    existing spawned VF2 workers. No unbounded queues or nested process pools.
    Valid RESUME shards skip both stages and still count toward progress.
    """
    started = time.perf_counter()
    rows: list[tuple[str, tuple[int, ...]]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[int, ...]] = set()
    chunk_span = batch * max(1, 4096 // batch)
    reused, rebuilt, extracted, matched, completed = 0, 0, 0, 0, 0
    extraction_seconds, vf2_seconds = 0.0, 0.0
    total = n * len(views)
    dictionary_digest = hashlib.sha256(json.dumps(
        [(key, dictionary.types[key].fingerprint,
          dictionary.types[key].prototype) for key in sorted(dictionary.types)],
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    for view_name, view in views.items():
        with (spawn_pool(workers=match_workers, initializer=initialize_match_worker,
                         initargs=(dictionary,))
              if match_workers > 1 else _LocalMatcher(dictionary)) as pool:
            for chunk_start in range(0, n, chunk_span):
                chunk_stop = min(n, chunk_start + chunk_span)
                shard = checkpoint_dir / f"{view_name}_{chunk_start:08d}.json"
                shard_identity = {
                    **identity, "view": view_name, "dictionary": dictionary_digest,
                    "start": chunk_start, "stop": chunk_stop,
                    "batch": batch, "ego": max_ego_nodes,
                    "minimum_nodes": min_figure_nodes,
                }
                saved = load_shard(shard, identity=shard_identity)
                if saved is None:
                    materialized: list[list] = []

                    def extract(start: int):
                        t0 = time.perf_counter()
                        centers = np.arange(start, min(chunk_stop, start + batch),
                                            dtype=np.int64)
                        candidates = tuple(x for x in view.extract_centers(
                            centers, radius=1, max_ego_nodes=max_ego_nodes,
                            workers=extraction_workers, symbol_types=node_types,
                        ) if len(x.nodes) >= min_figure_nodes)
                        return candidates, time.perf_counter() - t0

                    def match_candidates(candidates):
                        nonlocal vf2_seconds, matched
                        # Only a batch's 32-candidate pieces are in flight.
                        pieces = [tuple(candidates[j:j + 32])
                                  for j in range(0, len(candidates), 32)]
                        started_match = time.perf_counter()
                        results = (pool.map(match_chunk, pieces)
                                   if match_workers > 1 else map(pool.match, pieces))
                        for piece, matches in zip(pieces, results, strict=True):
                            for candidate, match in zip(piece, matches, strict=True):
                                if match is not None:
                                    materialized.append([
                                        str(match[0]),
                                        [int(node) for node in candidate.nodes],
                                    ])
                        vf2_seconds += time.perf_counter() - started_match
                        matched += len(candidates)

                    starts = iter(range(chunk_start, chunk_stop, batch))
                    if match_workers > 1:
                        # Prefetch only one batch. The single extraction task
                        # may itself use the existing bounded CPU thread pool.
                        with ThreadPoolExecutor(max_workers=1) as extractor:
                            future = extractor.submit(extract, next(starts))
                            while future is not None:
                                candidates, elapsed = future.result()
                                next_start = next(starts, None)
                                future = (extractor.submit(extract, next_start)
                                          if next_start is not None else None)
                                extraction_seconds += elapsed
                                extracted += len(candidates)
                                match_candidates(candidates)
                    else:
                        # No extra producer thread on single-worker paths.
                        for start in starts:
                            candidates, elapsed = extract(start)
                            extraction_seconds += elapsed
                            extracted += len(candidates)
                            match_candidates(candidates)
                    save_shard(shard, identity=shard_identity, rows=materialized)
                    saved = materialized
                    rebuilt += 1
                else:
                    reused += 1
                for type_id, nodes in saved:
                    key = tuple(nodes)
                    if key in seen:
                        continue
                    if type_id not in dictionary.types:
                        raise ValueError("scan shard references unknown symbol")
                    seen.add(key)
                    rows.append((str(type_id), key))
                    counts[str(type_id)] += 1
                completed += chunk_stop - chunk_start
                if checkpoint_hook is not None:
                    checkpoint_hook(shard)
                if progress_hook is not None:
                    progress_hook(completed, total)
    return rows, counts, {
        "checkpoint_shards_reused": reused,
        "checkpoint_shards_computed": rebuilt,
        "checkpoint_directory": str(checkpoint_dir),
        "centers_processed": completed,
        "centers_total": total,
        "candidates_extracted": extracted,
        "candidates_matched": matched,
        "extraction_seconds": round(extraction_seconds, 6),
        "vf2_seconds": round(vf2_seconds, 6),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "pipeline": "one_batch_prefetch" if match_workers > 1 else "sequential",
    }


class _LocalMatcher:
    """Only serial fallback; main process also owns dictionary mutation."""
    def __init__(self, dictionary: GraphDictionary):
        self.dictionary = dictionary

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def match(self, candidates):
        matches = []
        for candidate in candidates:
            result = self.dictionary.match_with_mapping(candidate)
            matches.append(None if result is None else
                           (result.graph_type.type_id, result.prototype_to_candidate))
        return tuple(matches)


def _wishart_allowed(dictionary: GraphDictionary, *, device: str, workers: int):
    """Actual Wishart density-mode probe over typed-WL exact-type representatives.

    This is a first-level matched-codec comparator, not the full recursive
    Wishart hierarchy from PR #13. Non-significant density modes are rejected.
    """
    ids = sorted(dictionary.types)
    if len(ids) < 3:
        return set(ids), {"cluster_count": 0, "status": "insufficient types"}
    reps = [dictionary.representative(key) for key in ids]
    features = typed_wl_features(reps, iterations=2, dimension=256,
                                 workers=max(1, min(workers, 4)))
    if device == "cpu":
        neighbors = _knn_from_features(
            features, k=min(8, len(ids) - 1), metric="cosine", device="cpu"
        )
    else:
        try:
            from semmap_haken.wishart_gpu import cupy_cosine_neighbors
            indices, distances, metadata = cupy_cosine_neighbors(
                features, k=min(8, len(ids) - 1)
            )
            from semmap_haken.wishart_metrics import NeighborGraph
            neighbors = NeighborGraph(indices, distances, metadata)
        except (ImportError, RuntimeError):
            if device == "cuda":
                raise
            neighbors = _knn_from_features(
                features, k=min(8, len(ids) - 1), metric="cosine", device="cpu"
            )
    clusters = wishart_cluster(neighbors.indices, neighbors.distances,
                               significance=0.35, min_cluster_size=2)
    accepted = {type_id for type_id, label in zip(ids, clusters.labels) if label >= 0}
    return accepted, {
        "cluster_count": clusters.cluster_count,
        "accepted_types": len(accepted),
        "noise_types": len(ids) - len(accepted),
        "neighbor_backend": neighbors.metadata,
    }


def _select(
    rows: list[tuple[str, tuple[int, ...]]], counts: Counter,
    dictionary: GraphDictionary, *, graph: TypedGraph, min_support: int,
    max_figures: int, allowed_types: set[str] | None = None,
    records: tuple[EdgeRecord, ...] | None = None,
) -> tuple[list[tuple[int, ...]], dict]:
    """Rank using *actual* record-level interfaces, then prevent overlaps.

    This is not the accept/reject criterion: prune_by_archive_bytes() makes
    that decision using fully serialized Huffman ZIPs and final frequencies.
    """
    from semmap_haken.grammar_binary import encoded_uvarint_size
    records = records if records is not None else edge_records(graph)
    eligible = {t: f for t, f in counts.items()
                if f >= min_support and (allowed_types is None or t in allowed_types)}
    codes = build_canonical_huffman_codes(eligible)
    incident: list[list[EdgeRecord]] = [[] for _ in range(graph.vertex_count)]
    for edge in records:
        incident[edge.source].append(edge)
        if edge.target != edge.source:
            incident[edge.target].append(edge)
    endpoint_bits = 8 * encoded_uvarint_size(max(0, graph.vertex_count-1))
    record_bits = 8 * encoded_uvarint_size(max(0, graph.edge_count-1))
    relation_bits = 8 * encoded_uvarint_size(max(0, len(graph.relation_names)-1))
    raw_per_edge = record_bits + 2 * endpoint_bits + relation_bits + 64

    scored = []
    true_gain: dict[int, float] = {}
    for i, (type_id, nodes) in enumerate(rows):
        support = eligible.get(type_id, 0)
        if not support:
            continue
        node_set = set(nodes)
        touched = {edge.edge_id: edge
                   for node in nodes for edge in incident[node]}
        internal = sum(
            edge.source in node_set and edge.target in node_set
            for edge in touched.values()
        )
        external = len(touched) - internal
        prototype = dictionary.representative(type_id)
        # Rule definition is paid once, and amortized over *scanned*
        # occurrences here. Accepted frequencies are recomputed after
        # overlap resolution in the measured codec selection.
        rule_estimate = (prototype.adjacency.nnz
                         * (2 * 8 * encoded_uvarint_size(len(nodes)-1)
                            + relation_bits) + len(nodes) * 8) / support
        raw_bits = (internal + external) * raw_per_edge
        coded = (
            len(nodes) * endpoint_bits
            + len(codes[type_id]) + rule_estimate
            + internal * (record_bits + 64)
            + external * (record_bits + endpoint_bits + relation_bits + 8 + 64)
        )
        gain = float(raw_bits - coded)
        true_gain[i] = gain
        # Do not use a lossy logical proxy as a hard reject. Only an actual
        # smaller ZIP may enter the final vocabulary.
        priority = max(1e-3, gain)
        scored.append(MdlOccurrence(
            dictionary_type_id=type_id, candidate_index=i,
            nodes=nodes, raw_bits=max(1.0, coded) + priority,
            encoded_bits=max(1.0, coded),
        ))
    selected = select_nonoverlapping_mdl(scored)
    selected = sorted(selected, key=lambda x: (-true_gain[x.candidate_index],
                                               x.candidate_index))[:max_figures]
    selected_freq = Counter(rows[item.candidate_index][0] for item in selected)
    final_codes = build_canonical_huffman_codes(selected_freq)
    return [tuple(item.nodes) for item in selected], {
        "selection_objective": "record-accurate port proxy, measured ZIP acceptance",
        "dictionary_types": len(dictionary.types),
        "full_scan_occurrences": len(rows),
        "eligible_types": len(eligible),
        "scored_candidates": len(scored),
        "nonoverlapping_figures": len(selected),
        "accepted_figures": len(selected),  # overwritten after measured pruning
        "covered_vertices": sum(len(item.nodes) for item in selected),
        "estimated_mdl_gain_bits": sum(true_gain[item.candidate_index] for item in selected),
        "post_overlap_type_counts": dict(sorted(selected_freq.items())),
        "post_overlap_huffman_code_lengths": {
            key: len(code) for key, code in sorted(final_codes.items())
        },
    }



def _bundle(
    output: Path, codec_path: Path, graph: TypedGraph,
    graph_hash: str, records: tuple[EdgeRecord, ...],
) -> int:
    """Same identifier sidecar for every method; no metadata cost omission."""
    metadata = {
        "node_ids": graph.node_ids,
        "assertion_ids": graph.assertion_ids,
        "relation_names": graph.relation_names,
        "graph_sha256": graph_hash,
    }
    path = output
    pending = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(pending, "w") as out:
        out.writestr("codec.zip", codec_path.read_bytes(),
                     compress_type=zipfile.ZIP_STORED)
        out.writestr("source_ids.json", json.dumps(metadata, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8"),
                     compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(pending) as test:
        ids = json.loads(test.read("source_ids.json"))
        n, edges = decode_grammar(io.BytesIO(test.read("codec.zip"))) if (
            codec_path.name.startswith("grammar_")
        ) else _decode_baseline_bytes(test.read("codec.zip"))
        if n != graph.vertex_count or edges != records:
            raise AssertionError("full bundle edge roundtrip mismatch")
        if tuple(ids["node_ids"]) != graph.node_ids or tuple(ids["assertion_ids"]) != graph.assertion_ids:
            raise AssertionError("full bundle identifiers mismatch")
    os.replace(pending, path)
    return path.stat().st_size


def _decode_baseline_bytes(data: bytes):
    from semmap_haken.exact_edge_baseline import decode_edge_baseline
    return decode_edge_baseline(io.BytesIO(data))


def _compress_method(
    graph: TypedGraph, out: Path, *, method: str, labels: np.ndarray | None,
    degree: np.ndarray, seed: int, discovery: int, max_figures: int,
    max_ego_nodes: int, min_figure_nodes: int, min_support: int,
    cpu_budget: int, batch: int, graph_hash: str, device: str,
    records: tuple[EdgeRecord, ...], full_view: EgoExtractor,
    progress_hook=None,
) -> dict:
    started = time.perf_counter()
    phases: dict[str, float] = {}
    def report(phase, done=1, total=1, fraction=0.0):
        if progress_hook is not None:
            progress_hook(phase, done, total, fraction)
    if method == "shuffled":
        rng = np.random.default_rng(seed + 1001)
        labels = rng.permutation(labels) if labels is not None else None
    elif method in ("plain", "wishart"):
        labels = None
    # Identical full-graph census and candidate limit for every strategy.
    # nSBM only changes the proposal ranking; independent Wishart never
    # receives nSBM partitions.
    views = {"full": full_view}
    extraction_workers = max(1, cpu_budget // 3)
    discovery_started = time.perf_counter()
    dictionary = _discover(
        graph, views, labels, degree, budget=discovery,
        max_ego_nodes=max_ego_nodes, min_figure_nodes=min_figure_nodes,
        seed=seed, extraction_workers=extraction_workers,
    )
    phases["discovery"] = time.perf_counter() - discovery_started
    report("discovery", fraction=0.12)
    extraction_workers, match_workers = _vf2_cpu_plan(cpu_budget, dictionary)
    rows, counts, scan_meta = _scan(
        views, dictionary, n=graph.vertex_count, batch=batch,
        max_ego_nodes=max_ego_nodes, min_figure_nodes=min_figure_nodes,
        extraction_workers=extraction_workers, match_workers=match_workers,
        checkpoint_dir=out / "scan" / method,
        progress_hook=lambda done, total: report("VF2", done, total,
                                                 0.12 + 0.58 * done / max(1, total)),
        identity={"graph": graph_hash, "method": method, "seed": seed,
                  "discovery": discovery, "selection": "full_census_v2",
                  "git_revision": _git_revision()},
    )
    phases["scan_wall"] = scan_meta["wall_seconds"]
    phases["extraction"] = scan_meta["extraction_seconds"]
    phases["vf2"] = scan_meta["vf2_seconds"]
    report("scan_completed", fraction=0.70)
    wishart_started = time.perf_counter()
    wishart = None
    allowed = None
    if method in ("wishart", "nsbm_wishart"):
        allowed, wishart = _wishart_allowed(
            dictionary, device=device, workers=match_workers
        )
    phases["wishart"] = time.perf_counter() - wishart_started
    selection_started = time.perf_counter()
    figures, selection = _select(
        rows, counts, dictionary, graph=graph, min_support=min_support,
        max_figures=max_figures, allowed_types=allowed, records=records,
    )
    phases["selection"] = time.perf_counter() - selection_started
    report("selection", fraction=0.78)
    type_by_nodes = {nodes: type_id for type_id, nodes in rows}
    pruning_started = time.perf_counter()
    figures, measured = prune_by_archive_bytes(
        vertex_count=graph.vertex_count, records=records,
        candidates=[(type_by_nodes[tuple(nodes)], tuple(nodes)) for nodes in figures],
        workspace=out,
        progress_hook=lambda done, total: report("measured_pruning", done, total,
                                                 0.78 + 0.10 * done / max(1, total)),
    )
    phases["measured_pruning"] = time.perf_counter() - pruning_started
    report("measured_pruning", fraction=0.88)
    selection.update(
        accepted_figures=len(figures),
        measured_codec_pruning=measured,
    )
    results = {}
    encoding_started = time.perf_counter()
    for coding in ("varint", "huffman"):
        path = out / f"grammar_{method}_{coding}.zip"
        report = encode_grammar(
            graph.vertex_count, records, figures, path, symbol_coding=coding
        )
        bundle_bytes = _bundle(
            out / f"bundle_{method}_{coding}.zip", path, graph, graph_hash, records
        )
        results[coding] = {
            "archive_bytes": report["archive_bytes"],
            "full_bundle_bytes": bundle_bytes,
            "archive_sha256": sha256_file(path),
            "bundle_sha256": sha256_file(out / f"bundle_{method}_{coding}.zip"),
            "ratio_vs_binary_edges": report["compression_ratio_binary_baseline"],
            "roundtrip_exact": report["roundtrip_exact"],
            "shapes": report["shapes"],
            "interfaces": report["interface_variants"],
            "stream_bytes": report["entry_compressed_bytes"],
        }
    phases["archives_and_validation"] = time.perf_counter() - encoding_started
    report("archives", fraction=0.99)
    phases["total_wall"] = time.perf_counter() - started
    result = {
        "method": method, "selection": selection,
        "search": {"discovery_budget": discovery,
                   "full_scan_centers": graph.vertex_count,
                   "used_nsbm_labels": method in ("nsbm", "nsbm_wishart", "shuffled"),
                   "scan_checkpoint": scan_meta},
        "wishart": wishart, "coding": results,
        "budget": {"cpu": cpu_budget, "vf2_processes": match_workers,
                   "ego_extraction_threads": extraction_workers},
        "seconds": time.perf_counter() - started,
        "phase_timing_seconds": {k: round(v, 6) for k, v in phases.items()},
    }
    _write_json(out / f"method_{method}.json", result)
    return result



class _RunProgress:
    """One durable progress file and bounded console event per stage."""

    def __init__(self, target: Path, total_steps: int):
        self.target = target
        self.total = max(1, total_steps)
        self.done = 0
        self.started = time.perf_counter()

    def event(self, phase: str, fraction: float = 0.0,
              completed: int = 0, total: int = 1) -> None:
        fraction = max(0.0, min(1.0, fraction))
        payload = {
            "phase": phase, "step": min(self.done + 1, self.total),
            "steps_total": self.total,
            "completed": int(completed), "total": int(total),
            "phase_percent": round(100.0 * fraction, 2),
            "overall_percent": round(100.0 * min(
                self.total, self.done + fraction) / self.total, 2),
            "elapsed_seconds": round(time.perf_counter() - self.started, 2),
        }
        atomic_json(self.target / "progress.json", payload)
        print("PROGRESS " + json.dumps(payload, ensure_ascii=False), flush=True)

    def advance(self, phase: str) -> None:
        self.event(phase, 1.0, 1, 1)
        self.done += 1

    def complete(self) -> None:
        self.done = self.total - 1
        self.advance("completed")

def run(
    graph: TypedGraph, out: str | Path, *, profile: str = "10k",
    seed: int = 42, restarts: int = 3, cpu_budget: int = 12,
    device: str = "auto", modes: Sequence[str] = ("nsbm", "plain", "shuffled", "wishart", "nsbm_wishart"),
    resume: bool = False, partition: np.ndarray | Sequence[np.ndarray] | None = None,
    recursive: bool = True, max_recursive_levels: int = 3,
    discovery: int | None = None, max_ego_nodes: int = 7,
    min_figure_nodes: int = 3, min_support: int = 3,
    isomorphism_level: str = "recursive",
) -> dict:
    if isomorphism_level not in LEVELS:
        raise ValueError("unknown isomorphism level")
    if profile not in PROFILE:
        raise ValueError("unknown profile")
    expected = PROFILE[profile]["vertices"]
    if expected is not None and graph.vertex_count != expected:
        raise ValueError(f"profile {profile} requires exactly {expected} vertices")
    if not modes or any(x not in ("nsbm", "plain", "shuffled", "wishart", "nsbm_wishart") for x in modes):
        raise ValueError("unknown compression method (nsbm/plain/shuffled/wishart/nsbm_wishart)")
    if (restarts < 1 or cpu_budget < 1 or min_support < 1 or
            max_ego_nodes < min_figure_nodes or min_figure_nodes < 2 or
            max_recursive_levels < 1):
        raise ValueError("invalid experiment parameters")
    cpu_budget = min(cpu_budget, _available_cpus())
    target = Path(out)
    hash_graph = _sha_bytes(graph)
    config = {
        "profile": profile, "seed": seed, "restarts": restarts,
        "cpu_budget": cpu_budget, "device": device, "modes": list(modes),
        "discovery": discovery or PROFILE[profile]["discovery"],
        "max_ego_nodes": max_ego_nodes, "min_figure_nodes": min_figure_nodes,
        "min_support": min_support,
        "recursive": bool(recursive),
        "max_recursive_levels": int(max_recursive_levels),
        "isomorphism_level": isomorphism_level,
    }
    manifest = {"schema": "nsbm-huffman-experiment-v1",
                "git_revision": _git_revision(), "graph_sha256": hash_graph,
                "config": config, "partition_provided": partition is not None}
    manifest_path = target / "manifest.json"
    if resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("RESUME manifest/input/code/config mismatch")
        if (target / "COMPLETED").is_file():
            try:
                prior = json.loads((target / "report.json").read_text(encoding="utf-8"))
                ok = all(verify_artifacts(prior["methods"][m], target) for m in modes)
                if ok and recursive and "nsbm" in modes:
                    rec = prior.get("recursive") or {}
                    path = Path(rec.get("archive", "missing"))
                    ok = (path.is_file() and rec.get("archive_sha256") ==
                          sha256_file(path))
                if ok:
                    return prior
            except (ValueError, OSError, KeyError, TypeError):
                pass
            (target / "COMPLETED").unlink()  # repair corrupt/incomplete report

    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError("NEW requires empty or absent run directory")
        target.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, manifest)
    started = time.perf_counter()
    timing = {}
    needs_fit = partition is None and any(m in ("nsbm", "shuffled", "nsbm_wishart") for m in modes)
    needs_projection = needs_fit or partition is not None
    needs_recursive = recursive and "nsbm" in modes and needs_projection
    planned_steps = (3 + int(needs_fit) + int(needs_projection)
                     + len(modes) + (max_recursive_levels if needs_recursive else 0))
    progress = _RunProgress(target, planned_steps)
    progress.event("start")
    t0 = time.perf_counter()
    degree, degree_backend = _degree(graph, device)
    timing["degree"] = time.perf_counter() - t0
    progress.advance("degree")
    fit = None
    all_partitions: list[np.ndarray] = []
    if needs_fit:
        t0 = time.perf_counter()
        all_partitions, fit = _fit_partition(
            graph, target, restarts=restarts, seed=seed, cpu_budget=cpu_budget,
            progress_hook=lambda done, total: progress.event(
                "nSBM restarts", done / max(1, total), done, total),
        )
        timing["nsbm_fit"] = time.perf_counter() - t0
        progress.advance("nSBM fit")
        partition = all_partitions[0]
        np.save(target / "partition.npy", partition, allow_pickle=False)
        _write_json(target / "fit_summary.json", fit)
    elif partition is not None:
        array = np.asarray(partition, dtype=np.int32)
        if array.ndim == 1:
            all_partitions = [array]
        elif array.ndim == 2:
            all_partitions = list(array)
        else:
            raise ValueError("partitions must be one- or two-dimensional")
        if any(p.shape != (graph.vertex_count,) or np.any(p < 0)
               for p in all_partitions):
            raise ValueError("invalid supplied partition")
        partition = all_partitions[0]
        fit = {"source": "external (smoke or precomputed)",
               "blocks_by_level": [len(set(p)) for p in all_partitions]}
    if partition is not None and fit is None:
        fit = json.loads((target / "fit_summary.json").read_text())
    projection = None
    if partition is not None:
        t0 = time.perf_counter()
        projected = estimate_block_kernel(graph, partition, device=device)
        projection = {"backend": projected["backend"],
                      "nonzero_entries": len(projected["entries"])}
        _write_json(target / "empirical_W.json", projected)
        timing["block_projection"] = time.perf_counter() - t0
        progress.advance("W projection")
    t0 = time.perf_counter()
    records = edge_records(graph)
    shared_full_view = _views(graph, None)["full"]
    baseline_path = target / "binary_exact.zip"
    baseline = encode_edge_baseline(graph.vertex_count, records, baseline_path)
    baseline_bundle = _bundle(target / "bundle_binary_exact.zip", baseline_path,
                              graph, hash_graph, records)
    timing["binary_baseline"] = time.perf_counter() - t0
    progress.advance("binary baseline")
    results = {}
    for mode in modes:
        stage_path = target / f"method_{mode}.json"
        if resume and stage_path.is_file():
            try:
                obj = json.loads(stage_path.read_text())
                if verify_artifacts(obj, target):
                    results[mode] = obj
                    progress.advance(f"{mode} (checkpoint)")
                    continue
            except (ValueError, OSError, KeyError, TypeError):
                pass  # incomplete report is not a valid checkpoint

        if mode in ("nsbm", "shuffled", "nsbm_wishart") and partition is None:
            raise RuntimeError("nSBM and shuffled baselines require partition")
        results[mode] = _compress_method(
            graph, target, method=mode,
            labels=partition, degree=degree, seed=seed,
            discovery=config["discovery"],
            max_figures=PROFILE[profile]["max_figures"],
            max_ego_nodes=max_ego_nodes, min_figure_nodes=min_figure_nodes,
            min_support=min_support, cpu_budget=cpu_budget,
            batch=PROFILE[profile]["batch"], graph_hash=hash_graph,
            device=device, records=records, full_view=shared_full_view,
            progress_hook=lambda phase, done, total, fraction: progress.event(
                f"{mode}: {phase}", fraction, done, total),
        )
        timing[f"method_{mode}"] = results[mode]["seconds"]
        progress.advance(f"{mode} finished")
    recursive_result = None
    if recursive and "nsbm" in modes and all_partitions:
        from .recursive_codec import run_recursive
        def recursion_progress(level, phase, done, total, fraction=0.0):
            if phase == "level_done":
                progress.advance(f"recursive L{level} finished")
            else:
                progress.event(f"recursive L{level}: {phase}",
                               fraction, done, total)
        t0 = time.perf_counter()
        recursive_result = run_recursive(
            graph, all_partitions, target / "recursive",
            seed=seed, cpu_budget=cpu_budget, discovery=config["discovery"],
            max_figures=PROFILE[profile]["max_figures"],
            max_ego_nodes=max_ego_nodes,
            min_figure_nodes=min_figure_nodes, min_support=min_support,
            batch=PROFILE[profile]["batch"], device=device,
            graph_hash=hash_graph, code_revision=_git_revision(),
            max_levels=max_recursive_levels, resume=resume,
            isomorphism_level=isomorphism_level,
            progress_hook=recursion_progress,
        )
        timing["recursive"] = time.perf_counter() - t0
    output = {
        "status": "complete", "counts": {"vertices": graph.vertex_count,
            "assertions": graph.edge_count, "relations": len(graph.relation_names)},
        "fit": fit, "projection": projection,
        "recursive": recursive_result,
        "degree_backend": degree_backend, "cpu_budget": cpu_budget,
        "baseline": {"archive_bytes": baseline["archive_bytes"],
                     "full_bundle_bytes": baseline_bundle},
        "methods": results,
        "ratios_full_bundle": {
            m: {coding: x["full_bundle_bytes"] / baseline_bundle
                for coding, x in row["coding"].items()}
            for m, row in results.items()
        },
        "total_wall_seconds": time.perf_counter() - started,
        "phase_timing_seconds": {k: round(v, 6) for k, v in timing.items()},
        "isomorphism_level": isomorphism_level,
        "warning": "One sample, experimental grammar: negative compression gains must be retained; nSBM structural MDL differs from actual archive size.",
    }
    _write_json(target / "report.json", output)
    (target / "COMPLETED").write_text("complete\n", encoding="utf-8")
    progress.complete()
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="nSBM-guided exact grammar + Huffman benchmark")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profile", choices=tuple(PROFILE), default="10k")
    ap.add_argument("--mode", choices=("NEW", "RESUME"), default="NEW")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--cpu-budget", type=int, default=12)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--methods", default="nsbm,plain,shuffled,wishart,nsbm_wishart")
    ap.add_argument("--discovery", type=int)
    ap.add_argument("--max-ego-nodes", type=int, default=7)
    ap.add_argument("--min-figure-nodes", type=int, default=3)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--max-recursive-levels", type=int, default=3)
    ap.add_argument("--isomorphism-level", choices=LEVELS, default="recursive")
    ap.add_argument("--no-recursive", action="store_true")
    args = ap.parse_args(argv)
    graph = TypedGraph.from_assertions(stream_assertions(args.input))
    results = run(
        graph, args.output, profile=args.profile, seed=args.seed,
        restarts=args.restarts, cpu_budget=args.cpu_budget, device=args.device,
        modes=tuple(x.strip() for x in args.methods.split(",")),
        resume=args.mode == "RESUME", discovery=args.discovery,
        max_ego_nodes=args.max_ego_nodes, min_figure_nodes=args.min_figure_nodes,
        min_support=args.min_support,
        recursive=not args.no_recursive,
        max_recursive_levels=args.max_recursive_levels,
        isomorphism_level=args.isomorphism_level,
    )
    print(json.dumps({"status": results["status"],
                      "counts": results["counts"],
                      "ratios_full_bundle": results["ratios_full_bundle"],
                      "cpu_budget": results["cpu_budget"],
                      "degree_backend": results["degree_backend"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
