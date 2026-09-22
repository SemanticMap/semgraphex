"""Candidate ego-subgraphs and pluggable distances for Wishart discovery."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .graph_build import PreparedGraph


@dataclass(frozen=True)
class EgoCandidate:
    center: int
    nodes: np.ndarray
    adjacency: sparse.csr_matrix
    relation_layers: dict[str, sparse.csr_matrix]


@dataclass(frozen=True)
class NeighborGraph:
    indices: np.ndarray
    distances: np.ndarray
    metadata: dict[str, object]


def _bucket(token: str, dimension: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimension


def relation_layers_from_prepared(
    graph: PreparedGraph,
    *,
    directed: bool,
) -> dict[str, sparse.csr_matrix]:
    """Reconstruct relation-specific sparse layers from selected edge provenance."""
    index = {uri: i for i, uri in enumerate(graph.node_ids)}
    by_relation: dict[str, dict[tuple[int, int], float]] = {}
    for row in graph.selected_edges:
        start = index.get(str(row.get("start_uri")))
        end = index.get(str(row.get("end_uri")))
        if start is None or end is None:
            continue
        relation = str(row.get("relation", "Unknown"))
        value = float(row.get("aggregate_contribution", 1.0))
        layer = by_relation.setdefault(relation, {})
        layer[(start, end)] = layer.get((start, end), 0.0) + value
        if not directed and start != end:
            layer[(end, start)] = layer.get((end, start), 0.0) + value
    result: dict[str, sparse.csr_matrix] = {}
    n = graph.adjacency.shape[0]
    for relation, values in sorted(by_relation.items()):
        if not values:
            result[relation] = sparse.csr_matrix((n, n), dtype=np.float64)
            continue
        rows, cols, data = zip(
            *((i, j, value) for (i, j), value in sorted(values.items()))
        )
        result[relation] = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n, n), dtype=np.float64
        )
    return result


def _ego_nodes(adjacency: sparse.csr_matrix, center: int, radius: int, cap: int) -> np.ndarray:
    """Sparse BFS with deterministic node-index tie breaking."""
    seen = {int(center)}
    frontier = [int(center)]
    for _ in range(radius):
        next_frontier: list[int] = []
        for node in frontier:
            start, stop = adjacency.indptr[node], adjacency.indptr[node + 1]
            for nbr in sorted(int(x) for x in adjacency.indices[start:stop]):
                if nbr not in seen:
                    seen.add(nbr)
                    next_frontier.append(nbr)
                    if len(seen) >= cap:
                        return np.array(sorted(seen), dtype=np.int64)
        frontier = next_frontier
        if not frontier:
            break
    return np.array(sorted(seen), dtype=np.int64)


def extract_ego_candidates(
    adjacency: sparse.spmatrix,
    relation_layers: Mapping[str, sparse.spmatrix],
    *,
    radius: int,
    max_ego_nodes: int,
    candidate_limit: int,
    seed: int,
) -> tuple[EgoCandidate, ...]:
    graph = adjacency.tocsr()
    n = graph.shape[0]
    if n == 0:
        return ()
    centers = np.arange(n, dtype=np.int64)
    if n > candidate_limit:
        rng = np.random.default_rng(seed)
        centers = np.sort(rng.choice(centers, size=candidate_limit, replace=False))
    candidates: list[EgoCandidate] = []
    typed = {name: matrix.tocsr() for name, matrix in relation_layers.items()}
    for center in centers:
        nodes = _ego_nodes(graph, int(center), radius, max_ego_nodes)
        if nodes.size < 2:
            continue
        sub = graph[nodes][:, nodes].tocsr()
        layers = {
            name: matrix[nodes][:, nodes].tocsr()
            for name, matrix in typed.items()
            if matrix[nodes][:, nodes].nnz
        }
        candidates.append(EgoCandidate(int(center), nodes, sub, layers))
    return tuple(candidates)


def relation_histogram_features(candidates: Sequence[EgoCandidate]) -> tuple[np.ndarray, tuple[str, ...]]:
    relations = tuple(sorted({r for c in candidates for r in c.relation_layers}))
    index = {r: i for i, r in enumerate(relations)}
    features = np.zeros((len(candidates), len(relations)), dtype=np.float64)
    for row, candidate in enumerate(candidates):
        for relation, layer in candidate.relation_layers.items():
            features[row, index[relation]] = float(np.abs(layer.data).sum())
        total = features[row].sum()
        if total > 0:
            features[row] /= total
    return features, relations


def typed_wl_features(
    candidates: Sequence[EgoCandidate],
    *,
    iterations: int,
    dimension: int,
) -> sparse.csr_matrix:
    """Hashed edge-type-aware WL subtree features, without concept labels."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for candidate_index, candidate in enumerate(candidates):
        n = candidate.adjacency.shape[0]
        labels = [f"d:{candidate.adjacency.getrow(i).nnz}" for i in range(n)]
        counts: dict[int, float] = {}
        for label in labels:
            b = _bucket("0|" + label, dimension)
            counts[b] = counts.get(b, 0.0) + 1.0
        for iteration in range(1, iterations + 1):
            updated: list[str] = []
            for node in range(n):
                messages: list[str] = []
                for relation, layer in sorted(candidate.relation_layers.items()):
                    out_start, out_stop = layer.indptr[node], layer.indptr[node + 1]
                    for nbr in layer.indices[out_start:out_stop]:
                        messages.append(f"o:{relation}:{labels[int(nbr)]}")
                    column = layer[:, node].tocoo()
                    for nbr in column.row:
                        messages.append(f"i:{relation}:{labels[int(nbr)]}")
                token = labels[node] + "|" + "|".join(sorted(messages))
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=12).hexdigest()
                updated.append(digest)
                b = _bucket(f"{iteration}|{digest}", dimension)
                counts[b] = counts.get(b, 0.0) + 1.0
            labels = updated
        for column, value in counts.items():
            rows.append(candidate_index)
            cols.append(column)
            data.append(value)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(candidates), dimension), dtype=np.float64
    )
    return normalize(matrix, norm="l2", copy=False)


def _motif_signature(candidate: EgoCandidate, nodes: tuple[int, ...]) -> str:
    sub = candidate.adjacency[list(nodes)][:, list(nodes)]
    binary = sub.copy()
    binary.data = np.ones_like(binary.data)
    degrees = tuple(sorted(np.asarray(binary.sum(axis=1)).ravel().astype(int)))
    rel_counts: list[str] = []
    for relation, layer in sorted(candidate.relation_layers.items()):
        count = int(layer[list(nodes)][:, list(nodes)].nnz)
        if count:
            rel_counts.append(f"{relation}:{count}")
    return f"k={len(nodes)}|e={binary.nnz}|deg={degrees}|rel={','.join(rel_counts)}"


def graphlet_features(
    candidates: Sequence[EgoCandidate],
    *,
    graphlet_size: int,
    samples: int,
    dimension: int,
    seed: int,
) -> sparse.csr_matrix:
    """Pre-sampled typed induced graphlet histogram using stable feature hashing."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for candidate_index, candidate in enumerate(candidates):
        n = candidate.adjacency.shape[0]
        if n < graphlet_size:
            continue
        all_count = math.comb(n, graphlet_size)
        rng = np.random.default_rng(seed + 104729 * candidate.center)
        if all_count <= samples:
            selected = list(combinations(range(n), graphlet_size))
        else:
            selected_set: set[tuple[int, ...]] = set()
            while len(selected_set) < samples:
                selected_set.add(tuple(sorted(
                    int(x) for x in rng.choice(n, graphlet_size, replace=False)
                )))
            selected = sorted(selected_set)
        counts: dict[int, float] = {}
        for nodes in selected:
            signature = _motif_signature(candidate, nodes)
            b = _bucket(signature, dimension)
            counts[b] = counts.get(b, 0.0) + 1.0
        for column, value in counts.items():
            rows.append(candidate_index)
            cols.append(column)
            data.append(value)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(candidates), dimension), dtype=np.float64
    )
    return normalize(matrix, norm="l2", copy=False)


def _knn_from_features(features: sparse.spmatrix | np.ndarray, k: int, metric: str) -> NeighborGraph:
    count = features.shape[0]
    if count <= 1:
        return NeighborGraph(np.empty((count, 0), dtype=np.int64), np.empty((count, 0)), {"backend": "features"})
    k = min(k, count - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    return NeighborGraph(indices[:, 1:].astype(np.int64), distances[:, 1:].astype(np.float64), {"backend": "features", "metric": metric})


def _knn_from_distance_matrix(matrix: np.ndarray, k: int, metadata: dict[str, object]) -> NeighborGraph:
    count = matrix.shape[0]
    if count <= 1:
        return NeighborGraph(np.empty((count, 0), dtype=np.int64), np.empty((count, 0)), metadata)
    k = min(k, count - 1)
    order = np.argsort(matrix, axis=1, kind="stable")
    indices = np.empty((count, k), dtype=np.int64)
    distances = np.empty((count, k), dtype=np.float64)
    for i in range(count):
        row = [int(j) for j in order[i] if int(j) != i][:k]
        indices[i] = row
        distances[i] = matrix[i, row]
    return NeighborGraph(indices, distances, metadata)


def relation_js_neighbors(candidates: Sequence[EgoCandidate], *, k: int) -> NeighborGraph:
    features, relations = relation_histogram_features(candidates)
    n = features.shape[0]
    distances = np.zeros((n, n), dtype=np.float64)
    # scipy's Jensen-Shannon function already returns sqrt(JS divergence).
    for i in range(n):
        for j in range(i + 1, n):
            if not features[i].any() and not features[j].any():
                value = 0.0
            elif not features[i].any() or not features[j].any():
                value = 1.0
            else:
                value = float(jensenshannon(features[i], features[j], base=2.0))
            distances[i, j] = distances[j, i] = value
    return _knn_from_distance_matrix(distances, k, {"backend": "pairwise", "metric": "sqrt_js", "relations": relations})


def _spectral_sample_coordinates(candidate: EgoCandidate, dimension: int) -> np.ndarray:
    """Permutation-invariant structural coordinates used by low-rank GW.

    The coordinates come from the leading nontrivial eigenvectors of the
    symmetrized normalized adjacency. They are not ConceptNet text embeddings.
    """
    adjacency = ((candidate.adjacency + candidate.adjacency.T) * 0.5).toarray()
    n = adjacency.shape[0]
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)
    degrees = adjacency.sum(axis=1)
    inv = np.zeros_like(degrees, dtype=np.float64)
    mask = degrees > 0
    inv[mask] = 1.0 / np.sqrt(degrees[mask])
    normalized = inv[:, None] * adjacency * inv[None, :]
    values, vectors = np.linalg.eigh(normalized)
    order = np.argsort(values)[::-1]
    take = order[1 : 1 + min(dimension, max(1, n - 1))]
    if len(take) == 0:
        return np.zeros((n, 1), dtype=np.float64)
    return vectors[:, take] * np.sqrt(np.maximum(np.abs(values[take]), 1e-12))[None, :]


def _transport_support(candidate: EgoCandidate, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic degree-landmark support and shortest-path structure cost."""
    adjacency = candidate.adjacency.tocsr()
    n = adjacency.shape[0]
    degree = np.asarray((adjacency != 0).sum(axis=1)).ravel()
    order = np.lexsort((np.arange(n), -degree))
    keep = np.sort(order[: min(rank, n)])
    binary = (adjacency[keep][:, keep] != 0).astype(float)
    distances = shortest_path(binary, directed=False, unweighted=True)
    finite = distances[np.isfinite(distances)]
    replacement = float(finite.max() + 1.0) if finite.size else 1.0
    distances[~np.isfinite(distances)] = replacement
    scale = distances.max()
    if scale > 0:
        distances = distances / scale
    return keep, distances


def _node_relation_features(candidate: EgoCandidate, keep: np.ndarray, relations: tuple[str, ...]) -> np.ndarray:
    result = np.zeros((len(keep), len(relations) * 2), dtype=np.float64)
    local_index = {int(node): i for i, node in enumerate(keep)}
    rel_index = {relation: i for i, relation in enumerate(relations)}
    for relation, layer in candidate.relation_layers.items():
        ridx = rel_index[relation]
        for local_node, output_row in local_index.items():
            result[output_row, 2 * ridx] = float(np.abs(layer.getrow(local_node).data).sum())
            result[output_row, 2 * ridx + 1] = float(np.abs(layer.getcol(local_node).data).sum())
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return result / norms


def transport_neighbors(
    candidates: Sequence[EgoCandidate],
    *,
    k: int,
    metric: str,
    rank: int,
    max_candidates: int,
    fgw_alpha: float,
) -> NeighborGraph:
    """Landmark-reduced GW or FGW via POT.

    `lowrank_gw` uses POT's low-rank GW solver (Scetbon--Peyre--Cuturi)
    on structural spectral coordinates. `fgw` uses deterministic degree
    landmarks to keep the exact FGW pair solver bounded.
    """
    if len(candidates) > max_candidates:
        raise ValueError(
            f"{metric} is quadratic over candidates; got {len(candidates)} > "
            f"transport_max_candidates={max_candidates}. Reduce candidate_limit."
        )
    try:
        import ot  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "GW/FGW metrics require POT. Install with: pip install -e '.[wishart]'"
        ) from error

    relations = tuple(sorted({r for c in candidates for r in c.relation_layers}))
    n = len(candidates)
    matrix = np.zeros((n, n), dtype=np.float64)

    if metric == "lowrank_gw":
        coordinates = [
            _spectral_sample_coordinates(candidate, min(8, rank))
            for candidate in candidates
        ]
        for i in range(n):
            for j in range(i + 1, n):
                plan_rank = min(rank, len(coordinates[i]), len(coordinates[j]))
                _, _, _, log = ot.gromov.lowrank_gromov_wasserstein_samples(
                    coordinates[i],
                    coordinates[j],
                    rank=max(1, plan_rank),
                    reg=0.0,
                    seed_init=49,
                    log=True,
                )
                value = float(log.get("value", 0.0))
                matrix[i, j] = matrix[j, i] = math.sqrt(max(0.0, value))
        metadata = {
            "backend": "POT",
            "metric": metric,
            "lowrank_solver": "ot.gromov.lowrank_gromov_wasserstein_samples",
            "plan_rank": rank,
            "structural_coordinates": "normalized-adjacency spectral embedding",
        }
    else:
        supports = [_transport_support(candidate, rank) for candidate in candidates]
        node_features = [
            _node_relation_features(candidate, keep, relations)
            for candidate, (keep, _) in zip(candidates, supports, strict=True)
        ]
        for i in range(n):
            keep_i, c_i = supports[i]
            p = np.full(len(keep_i), 1.0 / len(keep_i))
            for j in range(i + 1, n):
                keep_j, c_j = supports[j]
                q = np.full(len(keep_j), 1.0 / len(keep_j))
                feature_cost = cdist(node_features[i], node_features[j], metric="sqeuclidean")
                value = ot.gromov.fused_gromov_wasserstein2(
                    feature_cost, c_i, c_j, p, q,
                    loss_fun="square_loss", alpha=fgw_alpha, log=False,
                )
                matrix[i, j] = matrix[j, i] = math.sqrt(max(0.0, float(value)))
        metadata = {
            "backend": "POT",
            "metric": metric,
            "landmark_rank": rank,
            "relations": relations,
            "fgw_alpha": fgw_alpha,
        }
    return _knn_from_distance_matrix(matrix, k, metadata)


def build_neighbor_graph(
    candidates: Sequence[EgoCandidate],
    *,
    metric: str,
    k: int,
    wl_iterations: int,
    feature_dim: int,
    graphlet_size: int,
    graphlet_samples: int,
    transport_rank: int,
    transport_max_candidates: int,
    fgw_alpha: float,
    seed: int,
) -> NeighborGraph:
    if len(candidates) <= 1:
        return NeighborGraph(np.empty((len(candidates), 0), dtype=np.int64), np.empty((len(candidates), 0)), {"metric": metric})
    if metric == "typed_wl":
        features = typed_wl_features(candidates, iterations=wl_iterations, dimension=feature_dim)
        result = _knn_from_features(features, k, "cosine")
    elif metric == "graphlet":
        features = graphlet_features(
            candidates, graphlet_size=graphlet_size, samples=graphlet_samples,
            dimension=feature_dim, seed=seed,
        )
        result = _knn_from_features(features, k, "cosine")
    elif metric == "relation_js":
        result = relation_js_neighbors(candidates, k=k)
    elif metric in {"lowrank_gw", "fgw"}:
        result = transport_neighbors(
            candidates, k=k, metric=metric, rank=transport_rank,
            max_candidates=transport_max_candidates, fgw_alpha=fgw_alpha,
        )
    else:
        raise ValueError(f"unknown Wishart metric: {metric}")
    return NeighborGraph(
        result.indices,
        result.distances,
        {**result.metadata, "metric": metric, "candidate_count": len(candidates)},
    )
