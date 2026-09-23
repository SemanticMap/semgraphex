"""Dynamical diagnostics persisted at every Wishart coarsening level."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh

from .quotient import membership_matrix


@dataclass(frozen=True)
class DynamicSnapshot:
    node_count: int
    edge_nnz: int
    mean_degree: float
    degree_second_moment: float
    spreading_threshold: float | None
    percolation_threshold: float | None
    spectral_gap: float | None
    synchronizability: float | None
    mfpt_mean: float | None
    mfpt_hit_rate: float | None
    congestion_threshold: float | None
    max_betweenness: float | None
    macro_clustering: float | None
    intercluster_distance_mean: float | None
    intercluster_distance_median: float | None
    intercluster_distance_p95: float | None
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicVectors:
    stationary_mass: np.ndarray
    slow_eigenvalues: np.ndarray
    slow_eigenvectors: np.ndarray
    betweenness: np.ndarray
    sampled_distances: np.ndarray


def _undirected_nonnegative(adjacency: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = adjacency.tocsr().astype(np.float64, copy=False)
    if matrix.nnz and np.min(matrix.data) < 0:
        raise ValueError("dynamic diagnostics require non-negative edge weights")
    matrix = ((matrix + matrix.T) * 0.5).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _normalized_adjacency(matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(degrees, dtype=np.float64)
    mask = degrees > 0
    inv[mask] = 1.0 / np.sqrt(degrees[mask])
    d = sparse.diags(inv, format="csr")
    return (d @ matrix @ d).tocsr(), degrees


def _spectral_diagnostics(
    matrix: sparse.csr_matrix,
    *,
    slow_modes: int,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    n = matrix.shape[0]
    if n < 3 or matrix.nnz == 0:
        return np.empty(0), np.empty((n, 0)), None, None
    normalized, degrees = _normalized_adjacency(matrix)
    k = min(max(2, slow_modes + 1), n - 1)
    try:
        values, vectors = eigsh(normalized, k=k, which="LA")
        order = np.argsort(values)[::-1]
        values = np.asarray(values[order], dtype=np.float64)
        vectors = np.asarray(vectors[:, order], dtype=np.float64)
        spectral_gap = float(values[0] - values[1]) if len(values) > 1 else None
        slow_values = values[1 : 1 + slow_modes]
        slow_vectors = vectors[:, 1 : 1 + slow_modes]
    except Exception:
        slow_values = np.empty(0)
        slow_vectors = np.empty((n, 0))
        spectral_gap = None

    laplacian = sparse.diags(degrees, format="csr") - matrix
    synchronizability: float | None = None
    if n >= 3:
        try:
            small = eigsh(laplacian, k=2, which="SM", return_eigenvectors=False)
            lambda2 = float(np.sort(np.maximum(small, 0.0))[1])
            lambda_max = float(
                eigsh(laplacian, k=1, which="LA", return_eigenvectors=False)[0]
            )
            if lambda2 > 1e-14:
                synchronizability = lambda_max / lambda2
        except Exception:
            synchronizability = None
    return slow_values, slow_vectors, spectral_gap, synchronizability


def _sample_nodes(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=count, replace=False))


def _binary_topology(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    topology = (matrix != 0).astype(np.uint8).tocsr()
    topology.setdiag(0)
    topology.eliminate_zeros()
    return topology


def _sampled_betweenness_unweighted(
    topology: sparse.csr_matrix,
    *,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate unnormalized undirected betweenness using sampled Brandes sources.

    This avoids materializing a NetworkX graph, which is expensive in Colab.
    The n/k scaling matches the standard source-sampling idea; the final 1/2
    corrects double counting for undirected paths.
    """
    n = topology.shape[0]
    result = np.zeros(n, dtype=np.float64)
    if n < 2 or topology.nnz == 0:
        return result
    sources = _sample_nodes(n, min(sample_count, n), rng)
    for source in sources:
        source = int(source)
        stack: list[int] = []
        predecessors: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[source] = 1.0
        distance = np.full(n, -1, dtype=np.int64)
        distance[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            vertex = queue[head]
            head += 1
            stack.append(vertex)
            start, stop = topology.indptr[vertex], topology.indptr[vertex + 1]
            for neighbor in topology.indices[start:stop]:
                neighbor = int(neighbor)
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[vertex] + 1
                    queue.append(neighbor)
                if distance[neighbor] == distance[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        dependency = np.zeros(n, dtype=np.float64)
        while stack:
            node = stack.pop()
            if sigma[node] > 0:
                coefficient = (1.0 + dependency[node]) / sigma[node]
                for predecessor in predecessors[node]:
                    dependency[predecessor] += sigma[predecessor] * coefficient
            if node != source:
                result[node] += dependency[node]

    scale = (n / float(len(sources))) * 0.5
    return result * scale


def _sampled_clustering(
    topology: sparse.csr_matrix,
    *,
    sample_count: int,
    rng: np.random.Generator,
) -> float | None:
    n = topology.shape[0]
    if n == 0:
        return None
    sample = _sample_nodes(n, sample_count, rng)
    values: list[float] = []
    for node in sample:
        start, stop = topology.indptr[int(node)], topology.indptr[int(node) + 1]
        neighbors = topology.indices[start:stop]
        degree = len(neighbors)
        if degree < 2:
            values.append(0.0)
            continue
        # For symmetric topology nnz of the neighbor-induced subgraph counts
        # each undirected edge twice, exactly the numerator 2*T.
        neighbor_edges_twice = topology[neighbors][:, neighbors].nnz
        values.append(float(neighbor_edges_twice / (degree * (degree - 1))))
    return float(np.mean(values)) if values else None


def _sampled_path_distances(
    topology: sparse.csr_matrix,
    *,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = topology.shape[0]
    if n < 2 or topology.nnz == 0:
        return np.empty((0, 3), dtype=np.float64)
    sources = _sample_nodes(n, min(sample_count, n), rng)
    rows: list[tuple[float, float, float]] = []
    for source in sources:
        distances = shortest_path(
            topology,
            directed=False,
            unweighted=True,
            indices=int(source),
        )
        reachable = np.flatnonzero(np.isfinite(distances))
        reachable = reachable[reachable != int(source)]
        if reachable.size == 0:
            continue
        target = int(rng.choice(reachable))
        rows.append((float(source), float(target), float(distances[target])))
    return np.asarray(rows, dtype=np.float64).reshape((-1, 3))


def _mfpt_monte_carlo(
    matrix: sparse.csr_matrix,
    *,
    pairs: int,
    walks_per_pair: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    n = matrix.shape[0]
    if n < 2:
        return None, None
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    valid = np.flatnonzero(degrees > 0)
    if len(valid) < 2:
        return None, None

    pair_list: list[tuple[int, int]] = []
    for _ in range(pairs):
        source, target = rng.choice(valid, size=2, replace=False)
        pair_list.append((int(source), int(target)))

    passage_times: list[int] = []
    attempts = 0
    hits = 0
    for source, target in pair_list:
        for _ in range(walks_per_pair):
            attempts += 1
            node = source
            for step in range(1, max_steps + 1):
                start, stop = matrix.indptr[node], matrix.indptr[node + 1]
                neighbors = matrix.indices[start:stop]
                weights = matrix.data[start:stop]
                if len(neighbors) == 0 or weights.sum() <= 0:
                    break
                probability = weights / weights.sum()
                node = int(rng.choice(neighbors, p=probability))
                if node == target:
                    hits += 1
                    passage_times.append(step)
                    break
    if attempts == 0:
        return None, None
    return (
        float(np.mean(passage_times)) if passage_times else None,
        hits / float(attempts),
    )


def compute_dynamic_snapshot(
    adjacency: sparse.spmatrix,
    *,
    slow_modes: int,
    mfpt_pairs: int,
    mfpt_walks_per_pair: int,
    mfpt_max_steps: int,
    betweenness_samples: int,
    clustering_samples: int,
    distance_samples: int,
    seed: int,
) -> tuple[DynamicSnapshot, DynamicVectors]:
    """Compute sparse-safe/controlled-cost diagnostics on a symmetrized graph."""
    matrix = _undirected_nonnegative(adjacency)
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    strengths = np.asarray(matrix.sum(axis=1)).ravel()
    topological_degree = np.asarray((matrix != 0).sum(axis=1)).ravel().astype(np.float64)
    total_strength = strengths.sum()
    stationary = (
        strengths / total_strength
        if total_strength > 0
        else np.zeros(n, dtype=np.float64)
    )
    mean_degree = float(np.mean(topological_degree)) if n else 0.0
    second = float(np.mean(topological_degree ** 2)) if n else 0.0

    spreading = None if second <= 0 else mean_degree / second
    denominator = second - mean_degree
    percolation = None if denominator <= 0 else mean_degree / denominator

    slow_values, slow_vectors, gap, synchronizability = _spectral_diagnostics(
        matrix, slow_modes=slow_modes
    )

    mfpt_mean, mfpt_hit_rate = _mfpt_monte_carlo(
        matrix, pairs=mfpt_pairs, walks_per_pair=mfpt_walks_per_pair,
        max_steps=mfpt_max_steps, rng=rng,
    )

    topology = _binary_topology(matrix)
    betweenness = _sampled_betweenness_unweighted(
        topology,
        sample_count=betweenness_samples,
        rng=rng,
    )
    max_b = float(betweenness.max(initial=0.0)) if betweenness.size else None
    congestion = (
        (n - 1.0) / max_b
        if max_b is not None and max_b > 0
        else None
    )

    clustering = _sampled_clustering(
        topology,
        sample_count=clustering_samples,
        rng=rng,
    )

    sampled_pairs = _sampled_path_distances(
        topology,
        sample_count=distance_samples,
        rng=rng,
    )
    distance_values = sampled_pairs[:, 2] if sampled_pairs.size else np.empty(0)
    mean_distance = float(distance_values.mean()) if distance_values.size else None
    median_distance = float(np.median(distance_values)) if distance_values.size else None
    p95_distance = float(np.quantile(distance_values, 0.95)) if distance_values.size else None

    caveats = (
        "directed inputs are symmetrized for these diagnostics",
        "MFPT is Monte-Carlo estimated with a finite step cap",
        "betweenness uses source-sampled unweighted Brandes directly on CSR topology",
        "spreading threshold uses <k>/<k^2> heterogeneous-mean-field proxy",
        "percolation threshold uses <k>/(<k^2>-<k>) configuration-model proxy",
        "congestion threshold uses (N-1)/max unnormalized betweenness proxy",
    )
    snapshot = DynamicSnapshot(
        node_count=n,
        edge_nnz=int(matrix.nnz),
        mean_degree=mean_degree,
        degree_second_moment=second,
        spreading_threshold=spreading,
        percolation_threshold=percolation,
        spectral_gap=gap,
        synchronizability=synchronizability,
        mfpt_mean=mfpt_mean,
        mfpt_hit_rate=mfpt_hit_rate,
        congestion_threshold=congestion,
        max_betweenness=max_b,
        macro_clustering=clustering,
        intercluster_distance_mean=mean_distance,
        intercluster_distance_median=median_distance,
        intercluster_distance_p95=p95_distance,
        caveats=caveats,
    )
    vectors = DynamicVectors(
        stationary_mass=stationary,
        slow_eigenvalues=slow_values,
        slow_eigenvectors=slow_vectors,
        betweenness=betweenness,
        sampled_distances=sampled_pairs,
    )
    return snapshot, vectors


def cluster_transition_metrics(
    adjacency: sparse.spmatrix,
    fine_to_coarse: np.ndarray,
    *,
    relation_layers: Mapping[str, sparse.spmatrix],
    stationary_mass: np.ndarray,
    memberships: Mapping[int, Sequence[str]],
) -> list[dict[str, object]]:
    """Per-supernode flow, stationary mass and sparse exit-probability profile."""
    matrix = adjacency.tocsr()
    assignment = np.asarray(fine_to_coarse, dtype=np.int64)
    p = membership_matrix(assignment)
    block = (p.T @ matrix @ p).tocsr()
    relation_blocks = {
        relation: (p.T @ layer.tocsr() @ p).tocsr()
        for relation, layer in relation_layers.items()
    }
    rows: list[dict[str, object]] = []
    coarse_count = block.shape[0]
    for coarse in range(coarse_count):
        members = np.flatnonzero(assignment == coarse)
        concepts: list[str] = []
        for member in members:
            concepts.extend(str(x) for x in memberships[int(member)])
        concepts = sorted(concepts)

        row_start, row_stop = block.indptr[coarse], block.indptr[coarse + 1]
        targets = block.indices[row_start:row_stop]
        weights = block.data[row_start:row_stop]
        external = float(sum(
            weight for target, weight in zip(targets, weights, strict=True)
            if int(target) != coarse
        ))
        total_out = float(weights.sum())
        exit_probabilities = {
            str(int(target)): float(weight / total_out)
            for target, weight in zip(targets, weights, strict=True)
            if int(target) != coarse and total_out > 0
        }
        relation_flow: dict[str, float] = {}
        for relation, rel_block in relation_blocks.items():
            start, stop = rel_block.indptr[coarse], rel_block.indptr[coarse + 1]
            rel_targets = rel_block.indices[start:stop]
            rel_weights = rel_block.data[start:stop]
            value = float(sum(
                weight for target, weight in zip(rel_targets, rel_weights, strict=True)
                if int(target) != coarse
            ))
            if value:
                relation_flow[relation] = value

        rows.append({
            "coarse_node": coarse,
            "fine_nodes": [int(x) for x in members],
            "original_concepts": concepts,
            "concept_concat": " | ".join(concepts),
            "external_flow": external,
            "external_flow_by_relation": relation_flow,
            "stationary_mass": float(stationary_mass[members].sum()) if len(members) else 0.0,
            "exit_probabilities": exit_probabilities,
        })
    return rows
