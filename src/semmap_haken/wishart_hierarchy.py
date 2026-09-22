"""Recursive discovery of unknown compression figures with Wishart clustering."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .graph_build import PreparedGraph
from .quotient import membership_matrix
from .wishart_cluster import WishartClustering, wishart_cluster
from .wishart_config import WishartOptions
from .wishart_dynamics import (
    cluster_transition_metrics,
    compute_dynamic_snapshot,
)
from .wishart_metrics import (
    EgoCandidate,
    build_neighbor_graph,
    extract_ego_candidates,
    relation_layers_from_prepared,
)


@dataclass(frozen=True)
class FigureOccurrence:
    figure_type: int
    candidate_index: int
    center: int
    nodes: tuple[int, ...]
    kth_radius: float


@dataclass(frozen=True)
class TransitionPlan:
    fine_to_coarse: np.ndarray
    occurrences: tuple[FigureOccurrence, ...]
    figure_type_by_coarse: dict[int, int]


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


def _select_occurrences(
    candidates: Sequence[EgoCandidate],
    clustering: WishartClustering,
    *,
    min_figure_nodes: int,
    max_figures: int,
) -> tuple[FigureOccurrence, ...]:
    order = np.lexsort((
        np.array([candidate.center for candidate in candidates], dtype=np.int64),
        clustering.kth_radius,
    ))
    occupied: set[int] = set()
    accepted: list[FigureOccurrence] = []
    for candidate_index in order:
        candidate_index = int(candidate_index)
        label = int(clustering.labels[candidate_index])
        if label < 0:
            continue
        nodes = tuple(int(x) for x in candidates[candidate_index].nodes)
        if len(nodes) < min_figure_nodes:
            continue
        if any(node in occupied for node in nodes):
            continue
        accepted.append(FigureOccurrence(
            figure_type=label,
            candidate_index=candidate_index,
            center=int(candidates[candidate_index].center),
            nodes=nodes,
            kth_radius=float(clustering.kth_radius[candidate_index]),
        ))
        occupied.update(nodes)
        if len(accepted) >= max_figures:
            break
    return tuple(accepted)


def _partition_from_occurrences(
    node_count: int,
    occurrences: Sequence[FigureOccurrence],
) -> TransitionPlan:
    groups: list[tuple[int, ...]] = [tuple(sorted(item.nodes)) for item in occurrences]
    used = {node for group in groups for node in group}
    groups.extend((node,) for node in range(node_count) if node not in used)
    # Stable coarse index independent of density-order details.
    groups = sorted(groups, key=lambda group: (min(group), len(group), group))
    assignment = np.empty(node_count, dtype=np.int64)
    type_by_nodes = {tuple(sorted(item.nodes)): int(item.figure_type) for item in occurrences}
    type_by_coarse: dict[int, int] = {}
    for coarse, group in enumerate(groups):
        assignment[list(group)] = coarse
        if len(group) > 1 and group in type_by_nodes:
            type_by_coarse[coarse] = type_by_nodes[group]
    return TransitionPlan(assignment, tuple(occurrences), type_by_coarse)


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


def _write_transition(
    directory: Path,
    *,
    level: int,
    metric_metadata: Mapping[str, object],
    clustering: WishartClustering,
    candidates: Sequence[EgoCandidate],
    plan: TransitionPlan,
    cluster_rows: list[dict[str, object]],
    memberships: Mapping[int, Sequence[str]],
) -> None:
    transition_dir = directory / f"transition_{level:03d}_{level + 1:03d}"
    transition_dir.mkdir(parents=True, exist_ok=True)

    wishart_payload = {
        "metric": dict(metric_metadata),
        "cluster_count": clustering.cluster_count,
        "cluster_sizes": {str(k): v for k, v in clustering.cluster_sizes.items()},
        "cluster_peaks": {str(k): v for k, v in clustering.cluster_peaks.items()},
        "completed_clusters": list(clustering.completed_clusters),
        "noise_count": int(np.sum(clustering.labels < 0)),
        "candidate_count": len(candidates),
        "accepted_occurrences": len(plan.occurrences),
    }
    (transition_dir / "wishart.json").write_text(
        json.dumps(wishart_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        transition_dir / "candidate_labels.npz",
        labels=clustering.labels,
        kth_radius=clustering.kth_radius,
        log_density=clustering.log_density,
    )

    occurrence_rows = []
    for item in plan.occurrences:
        concepts: list[str] = []
        for node in item.nodes:
            concepts.extend(str(x) for x in memberships[int(node)])
        occurrence_rows.append({
            "figure_type": item.figure_type,
            "candidate_index": item.candidate_index,
            "center": item.center,
            "fine_nodes": list(item.nodes),
            "kth_radius": item.kth_radius,
            "original_concepts": sorted(concepts),
            "concept_concat": " | ".join(sorted(concepts)),
        })
    (transition_dir / "figure_occurrences.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in occurrence_rows),
        encoding="utf-8",
    )

    enriched = []
    for row in cluster_rows:
        coarse = int(row["coarse_node"])
        row = dict(row)
        row["figure_type"] = plan.figure_type_by_coarse.get(coarse)
        row["is_compression_figure"] = coarse in plan.figure_type_by_coarse
        enriched.append(row)
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
    output_dir: str | Path,
) -> WishartRunSummary:
    """Discover structural modes, contract non-overlapping instances, and repeat."""
    started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETED").exists():
        raise ValueError(f"output already completed: {destination}")

    current = graph.adjacency.tocsr().astype(np.float64, copy=False)
    relation_layers = relation_layers_from_prepared(graph, directed=directed)
    memberships: dict[int, tuple[str, ...]] = {
        i: (node_id,) for i, node_id in enumerate(graph.node_ids)
    }
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
            options=options,
        )
        level_summaries.append({
            "level": level,
            "node_count": int(current.shape[0]),
            "adjacency_nnz": int(current.nnz),
            "dynamic_metrics": dynamic_summary,
        })

        if level >= options.max_levels:
            stop_reason = "max_levels"
            break
        if current.shape[0] <= options.min_graph_nodes:
            stop_reason = "min_graph_nodes"
            break

        candidates = extract_ego_candidates(
            current,
            relation_layers,
            radius=options.radius,
            max_ego_nodes=options.max_ego_nodes,
            candidate_limit=options.candidate_limit,
            seed=options.random_seed + 2003 * level,
        )
        if len(candidates) <= options.k_neighbors:
            stop_reason = "too_few_candidates"
            break

        neighbors = build_neighbor_graph(
            candidates,
            metric=options.metric,
            k=options.k_neighbors,
            wl_iterations=options.wl_iterations,
            feature_dim=options.feature_dim,
            graphlet_size=options.graphlet_size,
            graphlet_samples=options.graphlet_samples,
            transport_rank=options.transport_rank,
            transport_max_candidates=options.transport_max_candidates,
            fgw_alpha=options.fgw_alpha,
            seed=options.random_seed + 3001 * level,
        )
        clustering = wishart_cluster(
            neighbors.indices,
            neighbors.distances,
            significance=options.significance,
            min_cluster_size=options.min_cluster_size,
        )
        occurrences = _select_occurrences(
            candidates,
            clustering,
            min_figure_nodes=options.min_figure_nodes,
            max_figures=options.max_figures_per_level,
        )
        if not occurrences:
            stop_reason = "no_nonoverlapping_figures"
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
            metric_metadata=neighbors.metadata,
            clustering=clustering,
            candidates=candidates,
            plan=plan,
            cluster_rows=cluster_rows,
            memberships=memberships,
        )

        old_count = current.shape[0]
        current = _contract_matrix(
            current, plan.fine_to_coarse, aggregation=options.aggregation
        )
        relation_layers = {
            relation: _contract_matrix(
                layer, plan.fine_to_coarse, aggregation=options.aggregation
            )
            for relation, layer in relation_layers.items()
        }
        memberships = _compose_memberships(memberships, plan.fine_to_coarse)
        transition_summaries.append({
            "source_level": level,
            "target_level": level + 1,
            "fine_nodes": int(old_count),
            "coarse_nodes": int(current.shape[0]),
            "compression_ratio": float(old_count / current.shape[0]),
            "wishart_clusters": clustering.cluster_count,
            "figure_occurrences": len(occurrences),
        })

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
        json.dumps({
            **summary.to_dict(),
            "options": asdict(options),
            "levels_detail": level_summaries,
            "transitions": transition_summaries,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return summary
