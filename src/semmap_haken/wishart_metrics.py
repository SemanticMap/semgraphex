"""Candidate ego-subgraphs and pluggable distances for Wishart discovery."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import warnings

from .graph_build import PreparedGraph


@dataclass(frozen=True)
class EgoCandidate:
    center: int
    nodes: np.ndarray
    adjacency: sparse.csr_matrix
    relation_layers: dict[str, sparse.csr_matrix]
    boundary_signature: tuple[tuple[int, str, str, int], ...] = ()
    node_types: tuple[str | None, ...] = ()


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


class EgoExtractor:
    """Reuse one read-only CSR view of the graph across bounded center batches.

    SciPy CSR slicing releases the GIL for the expensive native operations.
    Threads avoid copying large CSR graphs to worker processes (and avoid
    CUDA/fork hazards after initializing the accelerator).
    """

    def __init__(
        self,
        adjacency: sparse.spmatrix,
        relation_layers: Mapping[str, sparse.spmatrix],
    ) -> None:
        self.graph = adjacency.tocsr()
        self.typed = {name: layer.tocsr() for name, layer in relation_layers.items()}
        self.incoming = {name: layer.T.tocsr() for name, layer in self.typed.items()}
        # Calculated once, not 100k times inside the per-center loop.
        self.out_degree = {
            name: np.diff(matrix.indptr)
            for name, matrix in self.typed.items()
        }
        self.in_degree = {
            name: np.diff(matrix.indptr)
            for name, matrix in self.incoming.items()
        }

    def _candidate(
        self,
        center: int,
        *,
        radius: int,
        max_ego_nodes: int,
        symbol_types: Mapping[int, str],
    ) -> EgoCandidate | None:
        nodes = _ego_nodes(self.graph, center, radius, max_ego_nodes)
        if nodes.size < 2:
            return None
        sub = self.graph[nodes][:, nodes].tocsr()
        layers: dict[str, sparse.csr_matrix] = {}
        boundary: list[tuple[int, str, str, int]] = []
        for name, matrix in self.typed.items():
            local = matrix[nodes][:, nodes].tocsr()
            if local.nnz:
                layers[name] = local
            incoming = self.incoming[name]
            internal_out = np.diff(local.indptr)
            internal_in = np.bincount(
                local.indices, minlength=len(nodes)
            )
            external_out = self.out_degree[name][nodes] - internal_out
            external_in = self.in_degree[name][nodes] - internal_in
            for index in range(len(nodes)):
                if external_out[index] > 0:
                    boundary.append((index, name, "out", int(external_out[index])))
                if external_in[index] > 0:
                    boundary.append((index, name, "in", int(external_in[index])))
        local_types = tuple(symbol_types.get(int(node)) for node in nodes)
        return EgoCandidate(
            int(center),
            nodes,
            sub,
            layers,
            boundary_signature=tuple(sorted(boundary)),
            node_types=local_types,
        )

    def extract_centers(
        self,
        centers: Sequence[int],
        *,
        radius: int,
        max_ego_nodes: int,
        symbol_types: Mapping[int, str] | None = None,
        workers: int = 1,
        work_batch_size: int = 64,
    ) -> tuple[EgoCandidate, ...]:
        if workers < 1 or work_batch_size < 1:
            raise ValueError("workers and work_batch_size must be positive")
        selected = np.asarray(centers, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("candidate_centers must be one-dimensional")
        if np.any(selected < 0) or np.any(selected >= self.graph.shape[0]):
            raise ValueError("candidate_centers contain out-of-range node indices")
        symbols = symbol_types or {}

        def batch_extract(batch: np.ndarray) -> list[EgoCandidate]:
            rows = (
                self._candidate(
                    int(center),
                    radius=radius,
                    max_ego_nodes=max_ego_nodes,
                    symbol_types=symbols,
                )
                for center in batch
            )
            return [candidate for candidate in rows if candidate is not None]

        batches = (
            selected[index:index + work_batch_size]
            for index in range(0, len(selected), work_batch_size)
        )
        if workers == 1:
            return tuple(item for batch in batches for item in batch_extract(batch))
        # executor.map returns batches in submission order: seeded experiments
        # have the same candidate ordering regardless of thread scheduling.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return tuple(
                item
                for completed in executor.map(batch_extract, batches)
                for item in completed
            )


def extract_ego_candidates(
    adjacency: sparse.spmatrix,
    relation_layers: Mapping[str, sparse.spmatrix],
    *,
    radius: int,
    max_ego_nodes: int,
    candidate_limit: int,
    seed: int,
    symbol_types: Mapping[int, str] | None = None,
    candidate_centers: Sequence[int] | None = None,
    workers: int = 1,
) -> tuple[EgoCandidate, ...]:
    """Extract deterministic ego candidates, optionally with bounded threads."""
    graph = adjacency.tocsr()
    n = graph.shape[0]
    if n == 0:
        return ()
    if candidate_centers is None:
        centers = np.arange(n, dtype=np.int64)
        if n > candidate_limit:
            rng = np.random.default_rng(seed)
            centers = np.sort(rng.choice(centers, size=candidate_limit, replace=False))
    else:
        centers = np.asarray(candidate_centers, dtype=np.int64)
    extractor = EgoExtractor(graph, relation_layers)
    return extractor.extract_centers(
        centers,
        radius=radius,
        max_ego_nodes=max_ego_nodes,
        symbol_types=symbol_types,
        workers=workers,
    )


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
    workers: int = 1,
) -> sparse.csr_matrix:
    """Hashed typed WL features; parallel rows use spawn and ordered concatenation."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and len(candidates) > 1:
        from .wishart_parallel import ordered_feature_matrix
        return ordered_feature_matrix(
            "typed_wl", candidates, workers=workers,
            options={"iterations": iterations, "dimension": dimension},
            batch_size=max(1, min(64, (len(candidates) + workers * 2 - 1) // (workers * 2))),
        )
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for candidate_index, candidate in enumerate(candidates):
        n = candidate.adjacency.shape[0]
        node_types = candidate.node_types or tuple(None for _ in range(n))
        labels = [
            f"d:{candidate.adjacency.getrow(i).nnz}|t:{node_types[i] or 'ATOM'}"
            for i in range(n)
        ]
        # Repeated sparse column slicing is expensive on Colab. Build incoming
        # CSR layers once so both directions are O(local degree) per node.
        typed_layers = [
            (relation, layer.tocsr(), layer.T.tocsr())
            for relation, layer in sorted(candidate.relation_layers.items())
        ]
        counts: dict[int, float] = {}
        for label in labels:
            b = _bucket("0|" + label, dimension)
            counts[b] = counts.get(b, 0.0) + 1.0
        for iteration in range(1, iterations + 1):
            updated: list[str] = []
            for node in range(n):
                messages: list[str] = []
                for relation, outgoing, incoming in typed_layers:
                    out_start, out_stop = outgoing.indptr[node], outgoing.indptr[node + 1]
                    for nbr in outgoing.indices[out_start:out_stop]:
                        messages.append(f"o:{relation}:{labels[int(nbr)]}")
                    in_start, in_stop = incoming.indptr[node], incoming.indptr[node + 1]
                    for nbr in incoming.indices[in_start:in_stop]:
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
    workers: int = 1,
) -> sparse.csr_matrix:
    """Typed graphlet histograms; spawn workers preserve per-center RNG streams."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and len(candidates) > 1:
        from .wishart_parallel import ordered_feature_matrix
        return ordered_feature_matrix(
            "graphlet", candidates, workers=workers,
            options={
                "graphlet_size": graphlet_size, "samples": samples,
                "dimension": dimension, "seed": seed,
            },
            batch_size=max(1, min(64, (len(candidates) + workers * 2 - 1) // (workers * 2))),
        )
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


def _knn_from_features(
    features: sparse.spmatrix | np.ndarray,
    k: int,
    metric: str,
    *,
    device: str = "cpu",
    gpu_batch_size: int = 128,
    min_gpu_types: int = 128,
) -> NeighborGraph:
    count = features.shape[0]
    if count <= 1:
        return NeighborGraph(
            np.empty((count, 0), dtype=np.int64),
            np.empty((count, 0), dtype=np.float64),
            {"backend": "features", "device": "cpu"},
        )
    k = min(k, count - 1)
    from .wishart_gpu import cuda_cosine_neighbors, resolve_device

    actual = resolve_device(device)
    if actual == "cuda" and (device == "cuda" or count >= min_gpu_types):
        if metric != "cosine":
            raise ValueError("CUDA feature search currently supports cosine only")
        try:
            indices, distances, metadata = cuda_cosine_neighbors(
                features, k=k, query_batch_size=gpu_batch_size,
            )
            return NeighborGraph(indices, distances, metadata)
        except RuntimeError as error:
            if device != "auto" or "out of memory" not in str(error).lower():
                raise
            # Retry the entire neighbor graph on CPU; never silently mix
            # results from GPU and CPU after a partial GPU failure.
            fallback = "cuda_out_of_memory"
            warnings.warn(
                "CUDA cosine kNN ran out of GPU memory; "
                "recomputing the full neighbor graph on CPU",
                RuntimeWarning, stacklevel=2,
            )

    else:
        fallback = "small_type_space" if actual == "cuda" else None
        if fallback is not None:
            warnings.warn(
                f"CUDA available but type_count={count} < min_gpu_types="
                f"{min_gpu_types}; using sklearn CPU for this level",
                RuntimeWarning, stacklevel=2,
            )

    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    # Duplicate feature vectors can make sklearn return a different identical
    # point before 'self'; explicitly exclude the query index in every row.
    output_indices = np.empty((count, k), dtype=np.int64)
    output_distances = np.empty((count, k), dtype=np.float64)
    for row in range(count):
        retained = [(int(index), float(distance))
                    for index, distance in zip(indices[row], distances[row])
                    if int(index) != row]
        if len(retained) < k:
            additional = model.kneighbors(
                features[row], n_neighbors=min(count, k + 2),
            )
            retained = [
                (int(index), float(distance))
                for index, distance in zip(additional[1][0], additional[0][0])
                if int(index) != row
            ]
        retained = sorted(retained, key=lambda item: (item[1], item[0]))[:k]
        output_indices[row] = [item[0] for item in retained]
        output_distances[row] = [item[1] for item in retained]
    metadata: dict[str, object] = {
        "backend": "sklearn_cpu",
        "metric": metric,
        "device": "cpu",
    }
    if fallback is not None:
        metadata["fallback_reason"] = fallback
    return NeighborGraph(output_indices, output_distances, metadata)


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


def _sqrt_js_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized sqrt Jensen-Shannon distances for normalized row histograms."""
    p = left[:, None, :]
    q = right[None, :, :]
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_q = np.where(q > 0, q * np.log2(q / m), 0.0)
    divergence = 0.5 * (kl_p.sum(axis=2) + kl_q.sum(axis=2))
    distances = np.sqrt(np.maximum(divergence, 0.0))

    left_zero = np.isclose(left.sum(axis=1), 0.0)
    right_zero = np.isclose(right.sum(axis=1), 0.0)
    if np.any(left_zero) or np.any(right_zero):
        both_zero = left_zero[:, None] & right_zero[None, :]
        one_zero = left_zero[:, None] ^ right_zero[None, :]
        distances[both_zero] = 0.0
        distances[one_zero] = 1.0
    return distances


def relation_js_neighbors(
    candidates: Sequence[EgoCandidate],
    *,
    k: int,
    block_size: int = 128,
) -> NeighborGraph:
    """Top-k sqrt(JS) neighbors without materializing an N x N matrix.

    Memory is O(block_size * N + N * k), which is substantially safer in Colab
    than the previous all-pairs distance matrix.
    """
    features, relations = relation_histogram_features(candidates)
    n = features.shape[0]
    if n <= 1:
        return NeighborGraph(
            np.empty((n, 0), dtype=np.int64),
            np.empty((n, 0), dtype=np.float64),
            {"backend": "chunked_pairwise", "metric": "sqrt_js", "relations": relations},
        )
    k = min(k, n - 1)
    block_size = max(1, min(int(block_size), n))
    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float64)

    for block_start in range(0, n, block_size):
        block_stop = min(n, block_start + block_size)
        block = _sqrt_js_block(features[block_start:block_stop], features)
        local_rows = np.arange(block_stop - block_start)
        global_rows = np.arange(block_start, block_stop)
        block[local_rows, global_rows] = np.inf
        partition = np.argpartition(block, kth=k - 1, axis=1)[:, :k]
        part_distances = np.take_along_axis(block, partition, axis=1)
        order = np.argsort(part_distances, axis=1, kind="stable")
        indices[block_start:block_stop] = np.take_along_axis(partition, order, axis=1)
        distances[block_start:block_stop] = np.take_along_axis(part_distances, order, axis=1)

    return NeighborGraph(
        indices,
        distances,
        {
            "backend": "chunked_pairwise",
            "metric": "sqrt_js",
            "relations": relations,
            "block_size": block_size,
        },
    )

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
    binary = (adjacency != 0).astype(float)
    # Paths may cross non-landmark vertices: compute on the full ego graph
    # before restricting distances to the selected landmarks.
    distances = shortest_path(binary, directed=False, unweighted=True, indices=keep)[:, keep]
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
            "feature_source": "directed relation frequency profiles (no text embeddings)",
            "structural_cost": "full-ego shortest paths between degree landmarks",
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
    relation_js_block_size: int,
    seed: int,
    device: str = "cpu",
    gpu_batch_size: int = 128,
    min_gpu_types: int = 128,
    cpu_workers: int = 1,
    fgw_exact_types: int = 128,
    fgw_shortlist: int = 32,
    fgw_epsilon: float = 0.08,
    fgw_outer_iterations: int = 20,
    fgw_sinkhorn_iterations: int = 60,
    fgw_cache_pairs: int = 2048,
    fgw_type_ids: Sequence[str] | None = None,
    fgw_cache_dir: str | None = None,
    fgw_checkpoint_hook=None,
) -> NeighborGraph:
    if len(candidates) <= 1:
        return NeighborGraph(np.empty((len(candidates), 0), dtype=np.int64), np.empty((len(candidates), 0)), {"metric": metric})
    if metric == "typed_wl":
        features = typed_wl_features(
            candidates, iterations=wl_iterations, dimension=feature_dim,
            workers=cpu_workers,
        )
        result = _knn_from_features(
            features, k, "cosine", device=device,
            gpu_batch_size=gpu_batch_size, min_gpu_types=min_gpu_types,
        )
    elif metric == "graphlet":
        features = graphlet_features(
            candidates, graphlet_size=graphlet_size, samples=graphlet_samples,
            dimension=feature_dim, seed=seed, workers=cpu_workers,
        )
        result = _knn_from_features(
            features, k, "cosine", device=device,
            gpu_batch_size=gpu_batch_size, min_gpu_types=min_gpu_types,
        )
    elif metric == "relation_js":
        if device == "cuda":
            raise ValueError("CUDA is supported for typed_wl and graphlet only")
        result = relation_js_neighbors(
            candidates, k=k, block_size=relation_js_block_size
        )
    elif metric == "fgw" and device == "cuda":
        # Real CUDA computation: batched torch entropic FGW, not POT/CPU EMD.
        from .wishart_fgw_accel import accelerated_fgw_neighbors

        result = accelerated_fgw_neighbors(
            candidates, k=k, rank=transport_rank,
            max_candidates=transport_max_candidates, alpha=fgw_alpha,
            exact_types=fgw_exact_types, shortlist=fgw_shortlist,
            epsilon=fgw_epsilon, outer_iterations=fgw_outer_iterations,
            sinkhorn_iterations=fgw_sinkhorn_iterations,
            pair_batch_size=gpu_batch_size, cache_pairs=fgw_cache_pairs,
            cpu_workers=cpu_workers, device="cuda", type_ids=fgw_type_ids,
            cache_dir=fgw_cache_dir, checkpoint_hook=fgw_checkpoint_hook,
        )
    elif metric in {"lowrank_gw", "fgw"}:
        if device == "cuda":
            raise ValueError("CUDA is not supported for lowrank_gw")
        if metric == "fgw" and len(candidates) > transport_max_candidates:
            raise ValueError(
                f"fgw canonical types={len(candidates)} > "
                f"transport_max_candidates={transport_max_candidates}"
            )
        if metric == "fgw" and len(candidates) > fgw_exact_types:
            raise ValueError(
                "Large FGW canonical-type sets require --device cuda; "
                "CPU POT all-pairs is intentionally capped by fgw_exact_types"
            )
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
