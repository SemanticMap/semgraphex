"""Recursive Wishart discovery with a persistent exact graph dictionary."""

from __future__ import annotations

import gc
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .graph_build import PreparedGraph
from .graph_dictionary import GraphDictionary, WishartFamilyRegistry
from .graph_mdl import (
    MdlOccurrence,
    build_canonical_huffman_codes,
    estimate_dictionary_prototype_bits,
    estimated_occurrence_cost,
    select_nonoverlapping_mdl,
)
from .quotient import membership_matrix
from .wishart_cluster import WishartClustering, wishart_cluster
from .wishart_config import DictionaryOptions, WishartOptions
from .wishart_dynamics import cluster_transition_metrics, compute_dynamic_snapshot
from .wishart_metrics import (
    EgoCandidate,
    build_neighbor_graph,
    extract_ego_candidates,
    relation_layers_from_prepared,
)


@dataclass(frozen=True)
class FigureOccurrence:
    dictionary_type_id: str
    wishart_family_id: str | None
    wishart_cluster_label: int | None
    candidate_index: int
    center: int
    nodes: tuple[int, ...]
    prototype_to_fine_nodes: tuple[int, ...]
    kth_radius: float
    raw_bits: float
    encoded_bits: float
    mdl_gain: float


@dataclass(frozen=True)
class TransitionPlan:
    fine_to_coarse: np.ndarray
    occurrences: tuple[FigureOccurrence, ...]
    dictionary_type_by_coarse: dict[int, str]
    wishart_family_by_coarse: dict[int, str]


@dataclass(frozen=True)
class WishartRunSummary:
    metric: str
    levels: int
    stop_reason: str
    elapsed_seconds: float
    initial_nodes: int
    final_nodes: int
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ScannedOccurrence:
    candidate_index: int
    center: int
    nodes: tuple[int, ...]
    prototype_to_fine_nodes: tuple[int, ...]
    dictionary_type_id: str


def _empty_clustering(count: int) -> WishartClustering:
    return WishartClustering(
        labels=np.full(count, -1, dtype=np.int64),
        kth_radius=np.zeros(count, dtype=np.float64),
        log_density=np.zeros(count, dtype=np.float64),
        cluster_peaks={},
        completed_clusters=(),
        cluster_sizes={},
        cluster_masses={},
    )


def _partition_from_occurrences(
    node_count: int,
    occurrences: Sequence[FigureOccurrence],
) -> TransitionPlan:
    groups: list[tuple[int, ...]] = [tuple(sorted(item.nodes)) for item in occurrences]
    used = {node for group in groups for node in group}
    groups.extend((node,) for node in range(node_count) if node not in used)
    groups = sorted(groups, key=lambda group: (min(group), len(group), group))
    assignment = np.empty(node_count, dtype=np.int64)

    type_by_nodes = {
        tuple(sorted(item.nodes)): item.dictionary_type_id for item in occurrences
    }
    family_by_nodes = {
        tuple(sorted(item.nodes)): item.wishart_family_id
        for item in occurrences
        if item.wishart_family_id is not None
    }
    type_by_coarse: dict[int, str] = {}
    family_by_coarse: dict[int, str] = {}
    for coarse, group in enumerate(groups):
        assignment[list(group)] = coarse
        if len(group) > 1 and group in type_by_nodes:
            type_by_coarse[coarse] = type_by_nodes[group]
        if len(group) > 1 and group in family_by_nodes:
            family_by_coarse[coarse] = family_by_nodes[group]
    return TransitionPlan(
        assignment,
        tuple(occurrences),
        type_by_coarse,
        family_by_coarse,
    )


def _contract_matrix(
    matrix: sparse.spmatrix,
    assignment: np.ndarray,
    *,
    aggregation: str,
) -> sparse.csr_matrix:
    p = membership_matrix(assignment)
    coarse = (p.T @ matrix.tocsr() @ p).tocsr()
    if aggregation == "mean_density":
        sizes = np.asarray(p.sum(axis=0)).ravel()
        rows = np.repeat(np.arange(coarse.shape[0]), np.diff(coarse.indptr))
        scale = sizes[rows] * sizes[coarse.indices]
        coarse.data = coarse.data / scale
    coarse.eliminate_zeros()
    return coarse


def _compose_memberships(
    memberships: Mapping[int, Sequence[str]],
    assignment: np.ndarray,
) -> dict[int, tuple[str, ...]]:
    result: dict[int, list[str]] = {}
    for fine, coarse in enumerate(np.asarray(assignment, dtype=np.int64)):
        result.setdefault(int(coarse), []).extend(str(x) for x in memberships[int(fine)])
    return {
        coarse: tuple(sorted(values))
        for coarse, values in sorted(result.items())
    }


def _compose_symbol_types(
    previous: Mapping[int, str],
    plan: TransitionPlan,
) -> dict[int, str]:
    result: dict[int, str] = dict(plan.dictionary_type_by_coarse)
    groups: dict[int, list[int]] = {}
    for fine, coarse in enumerate(np.asarray(plan.fine_to_coarse, dtype=np.int64)):
        groups.setdefault(int(coarse), []).append(int(fine))
    for coarse, fine_nodes in groups.items():
        if coarse in result or len(fine_nodes) != 1:
            continue
        inherited = previous.get(fine_nodes[0])
        if inherited is not None:
            result[coarse] = inherited
    return result


def _relation_filename(relation: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", relation).strip("_") or "relation"
    return safe + ".npz"


def _write_level(
    directory: Path,
    *,
    level: int,
    adjacency: sparse.csr_matrix,
    relation_layers: Mapping[str, sparse.csr_matrix],
    memberships: Mapping[int, Sequence[str]],
    symbol_types: Mapping[int, str],
    options: WishartOptions,
) -> tuple[dict[str, object], np.ndarray]:
    level_dir = directory / f"level_{level:03d}"
    level_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(level_dir / "adjacency.npz", adjacency)

    relation_dir = level_dir / "relations"
    relation_dir.mkdir(exist_ok=True)
    relation_index: dict[str, str] = {}
    for relation, layer in sorted(relation_layers.items()):
        filename = _relation_filename(relation)
        sparse.save_npz(relation_dir / filename, layer)
        relation_index[relation] = filename
    (relation_dir / "index.json").write_text(
        json.dumps(relation_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    membership_payload = {
        str(index): {
            "original_concepts": list(values),
            "concept_concat": " | ".join(values),
        }
        for index, values in sorted(memberships.items())
    }
    (level_dir / "membership.json").write_text(
        json.dumps(membership_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (level_dir / "symbolic_nodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "node": int(index),
                    "dictionary_type_id": symbol_types.get(int(index)),
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(adjacency.shape[0])
        ),
        encoding="utf-8",
    )

    snapshot, vectors = compute_dynamic_snapshot(
        adjacency,
        slow_modes=options.slow_modes,
        mfpt_pairs=options.mfpt_pairs,
        mfpt_walks_per_pair=options.mfpt_walks_per_pair,
        mfpt_max_steps=options.mfpt_max_steps,
        betweenness_samples=options.betweenness_samples,
        clustering_samples=options.clustering_samples,
        distance_samples=options.distance_samples,
        seed=options.random_seed + 1009 * level,
    )
    (level_dir / "dynamic_metrics.json").write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        level_dir / "dynamic_vectors.npz",
        stationary_mass=vectors.stationary_mass,
        slow_eigenvalues=vectors.slow_eigenvalues,
        slow_eigenvectors=vectors.slow_eigenvectors,
        betweenness=vectors.betweenness,
        sampled_distances=vectors.sampled_distances,
    )
    return snapshot.to_dict(), vectors.stationary_mass


def _discover_types(
    candidates: Sequence[EgoCandidate],
    dictionary: GraphDictionary,
    *,
    level: int,
) -> tuple[tuple[str, ...], Counter[str]]:
    type_ids: list[str] = []
    counts: Counter[str] = Counter()
    for candidate in candidates:
        graph_type = dictionary.resolve_or_create(candidate, level=level)
        type_ids.append(graph_type.type_id)
        counts[graph_type.type_id] += 1
    return tuple(type_ids), counts


def _scan_known_types(
    adjacency: sparse.csr_matrix,
    relation_layers: Mapping[str, sparse.csr_matrix],
    dictionary: GraphDictionary,
    *,
    level: int,
    discovery_candidates: Sequence[EgoCandidate],
    discovery_type_ids: Sequence[str],
    symbol_types: Mapping[int, str],
    wishart_options: WishartOptions,
    dictionary_options: DictionaryOptions,
) -> tuple[tuple[_ScannedOccurrence, ...], Counter[str]]:
    if dictionary_options.frequency_scan == "discovery":
        rows: list[_ScannedOccurrence] = []
        for index, candidate in enumerate(discovery_candidates):
            matched = dictionary.match_with_mapping(candidate)
            if matched is None:
                continue
            rows.append(
                _ScannedOccurrence(
                    candidate_index=index,
                    center=int(candidate.center),
                    nodes=tuple(int(x) for x in candidate.nodes),
                    prototype_to_fine_nodes=tuple(
                        int(candidate.nodes[local_index])
                        for local_index in matched.prototype_to_candidate
                    ),
                    dictionary_type_id=matched.graph_type.type_id,
                )
            )
        scanned = tuple(rows)
    else:
        rows: list[_ScannedOccurrence] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        next_index = 0
        batch_size = dictionary_options.frequency_scan_batch_size
        for start in range(0, adjacency.shape[0], batch_size):
            stop = min(adjacency.shape[0], start + batch_size)
            candidates = extract_ego_candidates(
                adjacency,
                relation_layers,
                radius=wishart_options.radius,
                max_ego_nodes=wishart_options.max_ego_nodes,
                candidate_limit=stop - start,
                seed=wishart_options.random_seed + 4001 * level + start,
                symbol_types=symbol_types,
                candidate_centers=range(start, stop),
            )
            for candidate in candidates:
                matched = dictionary.match_with_mapping(candidate)
                if matched is None:
                    continue
                graph_type = matched.graph_type
                nodes = tuple(int(x) for x in candidate.nodes)
                key = (graph_type.type_id, nodes)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    _ScannedOccurrence(
                        candidate_index=next_index,
                        center=int(candidate.center),
                        nodes=nodes,
                        prototype_to_fine_nodes=tuple(
                            int(candidate.nodes[local_index])
                            for local_index in matched.prototype_to_candidate
                        ),
                        dictionary_type_id=graph_type.type_id,
                    )
                )
                next_index += 1
            del candidates
        scanned = tuple(rows)

    counts: Counter[str] = Counter(item.dictionary_type_id for item in scanned)
    dictionary.record_candidate_counts(level=level, counts=counts)
    return scanned, counts


def _cluster_dictionary_types(
    dictionary: GraphDictionary,
    family_registry: WishartFamilyRegistry,
    *,
    level: int,
    counts: Mapping[str, int],
    options: WishartOptions,
    dictionary_options: DictionaryOptions,
) -> tuple[
    tuple[str, ...],
    WishartClustering,
    dict[str, tuple[int | None, str | None, float]],
    dict[str, object],
]:
    if options.clustering_domain != "canonical_types":
        raise ValueError(
            "feature/wishart-graph-dictionary currently requires "
            "wishart.clustering_domain=canonical_types"
        )
    type_ids = tuple(
        sorted(
            type_id
            for type_id, count in counts.items()
            if count >= dictionary_options.min_support
        )
    )
    if len(type_ids) <= 1:
        clustering = _empty_clustering(len(type_ids))
        return type_ids, clustering, {
            type_id: (None, None, 0.0) for type_id in type_ids
        }, {
            "metric": options.metric,
            "candidate_count": len(type_ids),
            "backend": "not_run_too_few_types",
        }

    representatives = tuple(dictionary.representative(type_id) for type_id in type_ids)
    neighbors = build_neighbor_graph(
        representatives,
        metric=options.metric,
        k=options.k_neighbors,
        wl_iterations=options.wl_iterations,
        feature_dim=options.feature_dim,
        graphlet_size=options.graphlet_size,
        graphlet_samples=options.graphlet_samples,
        transport_rank=options.transport_rank,
        transport_max_candidates=options.transport_max_candidates,
        fgw_alpha=options.fgw_alpha,
        relation_js_block_size=options.relation_js_block_size,
        seed=options.random_seed + 3001 * level,
    )
    if options.density_weight == "occurrence_frequency":
        sample_weights = np.array([counts[type_id] for type_id in type_ids], dtype=float)
        min_mass: float | None = options.min_cluster_mass
    else:
        sample_weights = np.ones(len(type_ids), dtype=float)
        min_mass = None
    clustering = wishart_cluster(
        neighbors.indices,
        neighbors.distances,
        significance=options.significance,
        min_cluster_size=options.min_cluster_size,
        sample_weights=sample_weights,
        min_cluster_mass=min_mass,
    )

    info: dict[str, tuple[int | None, str | None, float]] = {}
    for cluster_label in sorted(set(int(x) for x in clustering.labels if x >= 0)):
        member_types = {
            type_ids[index]
            for index, label in enumerate(clustering.labels)
            if int(label) == cluster_label
        }
        family = family_registry.resolve(level=level, member_types=member_types)
        for index, type_id in enumerate(type_ids):
            if int(clustering.labels[index]) == cluster_label:
                info[type_id] = (
                    cluster_label,
                    family.family_id,
                    float(clustering.kth_radius[index]),
                )
    for type_id in type_ids:
        info.setdefault(type_id, (None, None, 0.0))
    return type_ids, clustering, info, neighbors.metadata


def _entropy_bits(counts: Mapping[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    return float(
        -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
            if count > 0
        )
    )


def _score_and_select_occurrences(
    scanned: Sequence[_ScannedOccurrence],
    dictionary: GraphDictionary,
    *,
    level: int,
    counts: Mapping[str, int],
    type_info: Mapping[str, tuple[int | None, str | None, float]],
    graph_node_count: int,
    relation_count: int,
    dictionary_options: DictionaryOptions,
    min_figure_nodes: int,
    max_figures: int,
) -> tuple[tuple[FigureOccurrence, ...], dict[str, object], dict[str, str]]:
    eligible_counts = {
        type_id: int(count)
        for type_id, count in counts.items()
        if int(count) >= dictionary_options.min_support
    }
    huffman = (
        build_canonical_huffman_codes(eligible_counts)
        if dictionary_options.huffman
        else {}
    )
    fallback_bits = max(1.0, math.log2(max(2, len(eligible_counts))))
    scored: list[MdlOccurrence] = []
    payload_by_index: dict[int, tuple[_ScannedOccurrence, float, float]] = {}

    for item in scanned:
        if len(item.nodes) < min_figure_nodes:
            continue
        support = eligible_counts.get(item.dictionary_type_id, 0)
        if support <= 0:
            continue
        representative = dictionary.representative(item.dictionary_type_id)
        prototype_bits = estimate_dictionary_prototype_bits(
            representative,
            relation_count=relation_count,
        )
        type_code_bits = float(
            len(huffman[item.dictionary_type_id])
            if item.dictionary_type_id in huffman
            else fallback_bits
        )
        raw_bits, encoded_bits = estimated_occurrence_cost(
            representative,
            graph_node_count=graph_node_count,
            relation_count=relation_count,
            type_code_bits=type_code_bits,
            dictionary_amortized_bits=prototype_bits / support,
        )
        if raw_bits - encoded_bits <= dictionary_options.min_mdl_gain_bits:
            continue
        mdl = MdlOccurrence(
            dictionary_type_id=item.dictionary_type_id,
            candidate_index=item.candidate_index,
            nodes=item.nodes,
            raw_bits=raw_bits,
            encoded_bits=encoded_bits,
        )
        scored.append(mdl)
        payload_by_index[item.candidate_index] = (item, raw_bits, encoded_bits)

    selected_mdl = select_nonoverlapping_mdl(
        scored,
        local_improvement=dictionary_options.local_improvement,
    )
    selected_mdl = tuple(
        sorted(selected_mdl, key=lambda item: item.mdl_gain, reverse=True)[:max_figures]
    )
    selected: list[FigureOccurrence] = []
    for mdl in selected_mdl:
        item, raw_bits, encoded_bits = payload_by_index[mdl.candidate_index]
        cluster_label, family_id, kth_radius = type_info.get(
            item.dictionary_type_id,
            (None, None, 0.0),
        )
        gain = float(raw_bits - encoded_bits)
        selected.append(
            FigureOccurrence(
                dictionary_type_id=item.dictionary_type_id,
                wishart_family_id=family_id,
                wishart_cluster_label=cluster_label,
                candidate_index=item.candidate_index,
                center=item.center,
                nodes=item.nodes,
                prototype_to_fine_nodes=item.prototype_to_fine_nodes,
                kth_radius=kth_radius,
                raw_bits=raw_bits,
                encoded_bits=encoded_bits,
                mdl_gain=gain,
            )
        )
        dictionary.record_accepted(
            level=level,
            type_id=item.dictionary_type_id,
            mdl_gain_bits=gain,
        )

    raw_total = float(sum(item.raw_bits for item in selected))
    encoded_total = float(sum(item.encoded_bits for item in selected))
    weighted_code_length = 0.0
    total_frequency = sum(eligible_counts.values())
    if total_frequency > 0:
        weighted_code_length = sum(
            eligible_counts[type_id] * len(code)
            for type_id, code in huffman.items()
        ) / total_frequency
    metrics = {
        "eligible_types": len(eligible_counts),
        "scanned_occurrences": len(scanned),
        "selected_occurrences": len(selected),
        "raw_bits_proxy": raw_total,
        "encoded_bits_proxy": encoded_total,
        "mdl_gain_bits_proxy": raw_total - encoded_total,
        "mdl_ratio_proxy": (
            encoded_total / raw_total if raw_total > 0 else None
        ),
        "type_entropy_bits": _entropy_bits(eligible_counts),
        "mean_huffman_code_length": weighted_code_length,
        "covered_nodes": int(sum(len(item.nodes) for item in selected)),
    }
    return tuple(selected), metrics, huffman


def _write_dictionary_artifacts(
    directory: Path,
    dictionary: GraphDictionary,
    family_registry: WishartFamilyRegistry,
    huffman: Mapping[str, str],
) -> None:
    dictionary_dir = directory / "dictionary"
    dictionary.write(dictionary_dir)
    family_registry.write(dictionary_dir)
    accepted_counts = dictionary.frequency_counts(accepted=True)
    candidate_counts = dictionary.frequency_counts(accepted=False)
    (dictionary_dir / "huffman.json").write_text(
        json.dumps(
            {
                type_id: {
                    "code": code,
                    "bits": len(code),
                    "accepted_frequency": accepted_counts.get(type_id, 0),
                    "candidate_frequency": candidate_counts.get(type_id, 0),
                }
                for type_id, code in sorted(huffman.items())
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (dictionary_dir / "grammar.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "type_id": item.type_id,
                    "children": list(item.child_types),
                    "recursive": bool(item.child_types),
                    "first_level": item.first_level,
                },
                sort_keys=True,
            )
            + "\n"
            for item in sorted(dictionary.types.values(), key=lambda value: value.type_id)
        ),
        encoding="utf-8",
    )


def _write_transition(
    directory: Path,
    *,
    level: int,
    metric_metadata: Mapping[str, object],
    clustering: WishartClustering,
    type_ids: Sequence[str],
    discovery_candidate_count: int,
    plan: TransitionPlan,
    cluster_rows: list[dict[str, object]],
    memberships: Mapping[int, Sequence[str]],
    dictionary_metrics: Mapping[str, object],
) -> None:
    transition_dir = directory / f"transition_{level:03d}_{level + 1:03d}"
    transition_dir.mkdir(parents=True, exist_ok=True)

    wishart_payload = {
        "metric": dict(metric_metadata),
        "clustering_domain": "canonical_types",
        "cluster_count": clustering.cluster_count,
        "cluster_sizes": {str(k): v for k, v in clustering.cluster_sizes.items()},
        "cluster_masses": {str(k): v for k, v in clustering.cluster_masses.items()},
        "cluster_peaks": {str(k): v for k, v in clustering.cluster_peaks.items()},
        "completed_clusters": list(clustering.completed_clusters),
        "noise_count": int(np.sum(clustering.labels < 0)),
        "discovery_candidate_count": int(discovery_candidate_count),
        "canonical_type_count": len(type_ids),
        "accepted_occurrences": len(plan.occurrences),
    }
    (transition_dir / "wishart.json").write_text(
        json.dumps(wishart_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        transition_dir / "type_labels.npz",
        type_ids=np.asarray(type_ids, dtype=str),
        labels=clustering.labels,
        kth_radius=clustering.kth_radius,
        log_density=clustering.log_density,
    )
    (transition_dir / "dictionary_metrics.json").write_text(
        json.dumps(dict(dictionary_metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    occurrence_rows = []
    for item in plan.occurrences:
        concepts: list[str] = []
        for node in item.nodes:
            concepts.extend(str(x) for x in memberships[int(node)])
        occurrence_rows.append(
            {
                "dictionary_type_id": item.dictionary_type_id,
                "wishart_family_id": item.wishart_family_id,
                "wishart_cluster_label": item.wishart_cluster_label,
                "candidate_index": item.candidate_index,
                "center": item.center,
                "fine_nodes": list(item.nodes),
                "prototype_to_fine_nodes": list(item.prototype_to_fine_nodes),
                "kth_radius": item.kth_radius,
                "raw_bits_proxy": item.raw_bits,
                "encoded_bits_proxy": item.encoded_bits,
                "mdl_gain_bits_proxy": item.mdl_gain,
                "original_concepts": sorted(concepts),
                "concept_concat": " | ".join(sorted(concepts)),
            }
        )
    (transition_dir / "figure_occurrences.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in occurrence_rows),
        encoding="utf-8",
    )

    enriched = []
    for row in cluster_rows:
        coarse = int(row["coarse_node"])
        item = dict(row)
        item["dictionary_type_id"] = plan.dictionary_type_by_coarse.get(coarse)
        item["wishart_family_id"] = plan.wishart_family_by_coarse.get(coarse)
        item["is_compression_figure"] = coarse in plan.dictionary_type_by_coarse
        enriched.append(item)
    (transition_dir / "cluster_dynamics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched),
        encoding="utf-8",
    )
    np.save(transition_dir / "fine_to_coarse.npy", plan.fine_to_coarse)


def run_wishart_hierarchy(
    graph: PreparedGraph,
    *,
    directed: bool,
    options: WishartOptions,
    dictionary_options: DictionaryOptions | None = None,
    output_dir: str | Path,
    checkpoint_hook: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> WishartRunSummary:
    """Learn persistent graph symbols, contract profitable instances, and repeat."""
    started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETED").exists():
        raise ValueError(f"output already completed: {destination}")

    dictionary_options = dictionary_options or DictionaryOptions()
    if not dictionary_options.enabled:
        raise ValueError(
            "graph dictionary is disabled; use feature/wishart-recursive-coarsening "
            "for the legacy contraction-only experiment"
        )

    current = graph.adjacency.tocsr().astype(np.float64, copy=False)
    relation_layers = relation_layers_from_prepared(graph, directed=directed)
    memberships: dict[int, tuple[str, ...]] = {
        i: (node_id,) for i, node_id in enumerate(graph.node_ids)
    }
    symbol_types: dict[int, str] = {}
    dictionary = GraphDictionary(
        boundary_sensitive=dictionary_options.boundary_sensitive
    )
    family_registry = WishartFamilyRegistry(
        match_jaccard=dictionary_options.family_match_jaccard
    )
    current_huffman: dict[str, str] = {}

    initial_nodes = current.shape[0]
    stop_reason = "max_levels"
    level_summaries: list[dict[str, object]] = []
    transition_summaries: list[dict[str, object]] = []

    for level in range(options.max_levels + 1):
        dynamic_summary, stationary_mass = _write_level(
            destination,
            level=level,
            adjacency=current,
            relation_layers=relation_layers,
            memberships=memberships,
            symbol_types=symbol_types,
            options=options,
        )
        level_summaries.append(
            {
                "level": level,
                "node_count": int(current.shape[0]),
                "adjacency_nnz": int(current.nnz),
                "dictionary_size": len(dictionary.types),
                "recursive_types": int(
                    sum(bool(item.child_types) for item in dictionary.types.values())
                ),
                "dynamic_metrics": dynamic_summary,
            }
        )
        if checkpoint_hook is not None:
            checkpoint_hook(
                destination,
                {
                    "stage": "level",
                    "level": level,
                    "node_count": int(current.shape[0]),
                    "adjacency_nnz": int(current.nnz),
                    "dictionary_size": len(dictionary.types),
                },
            )

        if level >= options.max_levels:
            stop_reason = "max_levels"
            break
        if current.shape[0] <= options.min_graph_nodes:
            stop_reason = "min_graph_nodes"
            break

        dictionary_size_before = len(dictionary.types)
        candidates = extract_ego_candidates(
            current,
            relation_layers,
            radius=options.radius,
            max_ego_nodes=options.max_ego_nodes,
            candidate_limit=options.candidate_limit,
            seed=options.random_seed + 2003 * level,
            symbol_types=symbol_types,
        )
        if not candidates:
            stop_reason = "no_candidates"
            break

        discovery_type_ids, _ = _discover_types(
            candidates,
            dictionary,
            level=level,
        )
        if len(dictionary.types) > dictionary_options.max_dictionary_size:
            stop_reason = "max_dictionary_size"
            break

        scanned, full_counts = _scan_known_types(
            current,
            relation_layers,
            dictionary,
            level=level,
            discovery_candidates=candidates,
            discovery_type_ids=discovery_type_ids,
            symbol_types=symbol_types,
            wishart_options=options,
            dictionary_options=dictionary_options,
            min_figure_nodes=options.min_figure_nodes,
            max_figures=options.max_figures_per_level,
        )

        type_ids, clustering, type_info, metric_metadata = _cluster_dictionary_types(
            dictionary,
            family_registry,
            level=level,
            counts=full_counts,
            options=options,
            dictionary_options=dictionary_options,
        )

        occurrences, mdl_metrics, selection_huffman = _score_and_select_occurrences(
            scanned,
            dictionary,
            level=level,
            counts=full_counts,
            type_info=type_info,
            graph_node_count=current.shape[0],
            relation_count=len(relation_layers),
            dictionary_options=dictionary_options,
            min_figure_nodes=options.min_figure_nodes,
            max_figures=options.max_figures_per_level,
        )
        current_huffman = (
            build_canonical_huffman_codes(
                dictionary.frequency_counts(accepted=True)
            )
            if dictionary_options.huffman
            else {}
        )
        mdl_metrics["selection_huffman_codes"] = len(selection_huffman)
        mdl_metrics["global_huffman_codes"] = len(current_huffman)

        observed_types = {
            type_id
            for type_id, count in full_counts.items()
            if count >= dictionary_options.min_support
        }
        new_types = {
            type_id
            for type_id in observed_types
            if dictionary.types[type_id].first_level == level
        }
        reused_types = observed_types - new_types
        dictionary_metrics = {
            **mdl_metrics,
            "dictionary_size_before": dictionary_size_before,
            "dictionary_size_after_discovery": len(dictionary.types),
            "observed_supported_types": len(observed_types),
            "new_types": len(new_types),
            "reused_types": len(reused_types),
            "dictionary_novelty": (
                len(new_types) / len(observed_types) if observed_types else 0.0
            ),
            "reuse_rate": (
                len(reused_types) / len(observed_types) if observed_types else 0.0
            ),
            "recursive_types_total": int(
                sum(bool(item.child_types) for item in dictionary.types.values())
            ),
        }
        _write_dictionary_artifacts(
            destination,
            dictionary,
            family_registry,
            current_huffman,
        )

        if not occurrences:
            stop_reason = "no_positive_mdl"
            transition_summaries.append(
                {
                    "source_level": level,
                    "target_level": level,
                    "fine_nodes": int(current.shape[0]),
                    "coarse_nodes": int(current.shape[0]),
                    "compression_ratio": 1.0,
                    "wishart_clusters": clustering.cluster_count,
                    "dictionary_metrics": dictionary_metrics,
                }
            )
            break

        plan = _partition_from_occurrences(current.shape[0], occurrences)
        coarse_count = int(plan.fine_to_coarse.max()) + 1
        if coarse_count >= current.shape[0]:
            stop_reason = "no_reduction"
            break

        cluster_rows = cluster_transition_metrics(
            current,
            plan.fine_to_coarse,
            relation_layers=relation_layers,
            stationary_mass=stationary_mass,
            memberships=memberships,
        )
        _write_transition(
            destination,
            level=level,
            metric_metadata=metric_metadata,
            clustering=clustering,
            type_ids=type_ids,
            discovery_candidate_count=len(candidates),
            plan=plan,
            cluster_rows=cluster_rows,
            memberships=memberships,
            dictionary_metrics=dictionary_metrics,
        )
        if checkpoint_hook is not None:
            checkpoint_hook(
                destination,
                {
                    "stage": "transition",
                    "source_level": level,
                    "target_level": level + 1,
                    "wishart_clusters": clustering.cluster_count,
                    "dictionary_size": len(dictionary.types),
                    "figure_occurrences": len(occurrences),
                    "mdl_gain_bits_proxy": dictionary_metrics["mdl_gain_bits_proxy"],
                },
            )

        old_count = current.shape[0]
        previous_symbol_types = symbol_types
        current = _contract_matrix(
            current,
            plan.fine_to_coarse,
            aggregation=options.aggregation,
        )
        relation_layers = {
            relation: _contract_matrix(
                layer,
                plan.fine_to_coarse,
                aggregation=options.aggregation,
            )
            for relation, layer in relation_layers.items()
        }
        memberships = _compose_memberships(memberships, plan.fine_to_coarse)
        symbol_types = _compose_symbol_types(previous_symbol_types, plan)

        transition_summaries.append(
            {
                "source_level": level,
                "target_level": level + 1,
                "fine_nodes": int(old_count),
                "coarse_nodes": int(current.shape[0]),
                "compression_ratio": float(old_count / current.shape[0]),
                "wishart_clusters": clustering.cluster_count,
                "canonical_types": len(type_ids),
                "figure_occurrences": len(occurrences),
                "dictionary_metrics": dictionary_metrics,
            }
        )

        del candidates, discovery_type_ids, scanned, occurrences, plan, cluster_rows
        gc.collect()

    _write_dictionary_artifacts(
        destination,
        dictionary,
        family_registry,
        current_huffman,
    )
    summary = WishartRunSummary(
        metric=options.metric,
        levels=len(level_summaries),
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - started,
        initial_nodes=int(initial_nodes),
        final_nodes=int(current.shape[0]),
        output_dir=str(destination),
    )
    (destination / "hierarchy.json").write_text(
        json.dumps(
            {
                **summary.to_dict(),
                "options": asdict(options),
                "dictionary_options": asdict(dictionary_options),
                "dictionary_size": len(dictionary.types),
                "wishart_family_count": len(family_registry.families),
                "levels_detail": level_summaries,
                "transitions": transition_summaries,
                "mdl_note": (
                    "Bit counts are a transparent structural MDL proxy; they are "
                    "not measured bytes of a finalized lossless binary codec."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "COMPLETED").write_text("complete\n", encoding="utf-8")
    if checkpoint_hook is not None:
        checkpoint_hook(
            destination,
            {
                "stage": "completed",
                "levels": len(level_summaries),
                "stop_reason": stop_reason,
                "final_nodes": int(current.shape[0]),
                "dictionary_size": len(dictionary.types),
            },
        )
    return summary
