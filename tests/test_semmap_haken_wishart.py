from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from semmap_haken.graph_dictionary import GraphDictionary, WishartFamilyRegistry
from semmap_haken.graph_mdl import MdlOccurrence, build_canonical_huffman_codes, select_nonoverlapping_mdl
from semmap_haken.wishart_cluster import wishart_cluster
from semmap_haken.wishart_metrics import (
    EgoCandidate,
    relation_js_neighbors,
    typed_wl_features,
)


def _layer(edges: list[tuple[int, int]], n: int = 3) -> sparse.csr_matrix:
    if not edges:
        return sparse.csr_matrix((n, n), dtype=float)
    rows, cols = zip(*edges)
    return sparse.csr_matrix(
        (np.ones(len(edges)), (rows, cols)), shape=(n, n), dtype=float
    )


def test_wishart_finds_two_separated_modes_without_predefined_cluster_count():
    points = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    nn = NearestNeighbors(n_neighbors=3).fit(points)
    distances, indices = nn.kneighbors(points)
    result = wishart_cluster(
        indices[:, 1:],
        distances[:, 1:],
        significance=0.2,
        min_cluster_size=2,
    )
    assert result.cluster_count == 2
    assert set(result.labels[:3]) == {result.labels[0]}
    assert set(result.labels[3:]) == {result.labels[3]}
    assert result.labels[0] != result.labels[3]


def test_relation_js_keeps_same_relation_profile_closest():
    adjacency = _layer([(0, 1), (0, 2)])
    a = EgoCandidate(0, np.arange(3), adjacency, {"IsA": adjacency})
    b = EgoCandidate(1, np.arange(3), adjacency, {"IsA": adjacency * 2})
    c = EgoCandidate(2, np.arange(3), adjacency, {"UsedFor": adjacency})
    neighbors = relation_js_neighbors((a, b, c), k=1)
    assert neighbors.indices[0, 0] == 1
    assert neighbors.distances[0, 0] == 0.0
    assert neighbors.distances[2, 0] > 0.0


def test_typed_wl_changes_when_relation_type_changes():
    adjacency = _layer([(0, 1), (1, 0), (0, 2), (2, 0)])
    a = EgoCandidate(0, np.arange(3), adjacency, {"IsA": adjacency})
    b = EgoCandidate(1, np.arange(3), adjacency, {"UsedFor": adjacency})
    features = typed_wl_features((a, b), iterations=2, dimension=128).toarray()
    assert not np.allclose(features[0], features[1])


def test_dynamic_snapshot_runs_on_small_sparse_graph():
    from semmap_haken.wishart_dynamics import compute_dynamic_snapshot

    adjacency = sparse.csr_matrix(
        np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], dtype=float)
    )
    snapshot, vectors = compute_dynamic_snapshot(
        adjacency,
        slow_modes=2,
        mfpt_pairs=2,
        mfpt_walks_per_pair=2,
        mfpt_max_steps=20,
        betweenness_samples=4,
        clustering_samples=4,
        distance_samples=4,
        seed=7,
    )
    assert snapshot.node_count == 4
    assert vectors.stationary_mass.shape == (4,)
    assert np.isclose(vectors.stationary_mass.sum(), 1.0)
    assert snapshot.mean_degree == 2.0


def test_partition_contracts_each_figure_instance_separately():
    from semmap_haken.wishart_hierarchy import (
        FigureOccurrence,
        _partition_from_occurrences,
    )

    occurrences = (
        FigureOccurrence(
            dictionary_type_id="GT_000001",
            wishart_family_id="WF_000001",
            wishart_cluster_label=3,
            candidate_index=0,
            center=0,
            nodes=(0, 1),
            prototype_to_fine_nodes=(0, 1),
            kth_radius=0.1,
            raw_bits=20.0,
            encoded_bits=8.0,
            mdl_gain=12.0,
        ),
        FigureOccurrence(
            dictionary_type_id="GT_000001",
            wishart_family_id="WF_000001",
            wishart_cluster_label=3,
            candidate_index=4,
            center=4,
            nodes=(4, 5),
            prototype_to_fine_nodes=(4, 5),
            kth_radius=0.1,
            raw_bits=20.0,
            encoded_bits=8.0,
            mdl_gain=12.0,
        ),
    )
    plan = _partition_from_occurrences(6, occurrences)
    assert plan.fine_to_coarse[0] == plan.fine_to_coarse[1]
    assert plan.fine_to_coarse[4] == plan.fine_to_coarse[5]
    assert plan.fine_to_coarse[0] != plan.fine_to_coarse[4]
    assert set(plan.dictionary_type_by_coarse.values()) == {"GT_000001"}
    assert set(plan.wishart_family_by_coarse.values()) == {"WF_000001"}


def test_graph_dictionary_deduplicates_isomorphic_typed_candidates() -> None:
    adjacency = _layer([(0, 1), (1, 0), (1, 2), (2, 1)])
    layer = adjacency.copy()
    a = EgoCandidate(
        10,
        np.array([10, 11, 12]),
        adjacency,
        {"IsA": layer},
        boundary_signature=((0, "IsA", "out", 2),),
        node_types=(None, None, None),
    )
    permutation = np.array([2, 1, 0])
    b = EgoCandidate(
        20,
        np.array([20, 21, 22]),
        adjacency[permutation][:, permutation].tocsr(),
        {"IsA": layer[permutation][:, permutation].tocsr()},
        boundary_signature=((2, "IsA", "out", 2),),
        node_types=(None, None, None),
    )
    dictionary = GraphDictionary(boundary_sensitive=True)
    first = dictionary.resolve_or_create(a, level=0)
    second = dictionary.resolve_or_create(b, level=0)
    assert first.type_id == second.type_id
    assert len(dictionary.types) == 1


def test_graph_dictionary_separates_relation_and_recursive_symbol_types() -> None:
    adjacency = _layer([(0, 1), (1, 0), (1, 2), (2, 1)])
    is_a = EgoCandidate(
        0, np.arange(3), adjacency, {"IsA": adjacency},
        node_types=(None, None, None),
    )
    used_for = EgoCandidate(
        1, np.arange(3), adjacency, {"UsedFor": adjacency},
        node_types=(None, None, None),
    )
    recursive = EgoCandidate(
        2, np.arange(3), adjacency, {"IsA": adjacency},
        node_types=("GT_CHILD", None, None),
    )
    dictionary = GraphDictionary(boundary_sensitive=False)
    ids = {
        dictionary.resolve_or_create(is_a, level=0).type_id,
        dictionary.resolve_or_create(used_for, level=0).type_id,
        dictionary.resolve_or_create(recursive, level=1).type_id,
    }
    assert len(ids) == 3


def test_wishart_family_registry_reuses_family_by_member_overlap() -> None:
    registry = WishartFamilyRegistry(match_jaccard=0.5)
    first = registry.resolve(level=0, member_types={"GT_1", "GT_2", "GT_3"})
    second = registry.resolve(level=1, member_types={"GT_2", "GT_3", "GT_4"})
    assert first.family_id == second.family_id
    assert second.levels_seen == (0, 1)


def test_canonical_huffman_assigns_shorter_codes_to_frequent_types() -> None:
    codes = build_canonical_huffman_codes({"GT_A": 100, "GT_B": 20, "GT_C": 5})
    assert len(codes["GT_A"]) <= len(codes["GT_B"])
    assert len(codes["GT_B"]) <= len(codes["GT_C"])
    assert len(set(codes.values())) == 3


def test_mdl_overlap_selection_prefers_higher_total_gain() -> None:
    occurrences = (
        MdlOccurrence("GT_A", 0, (0, 1, 2), raw_bits=30.0, encoded_bits=10.0),
        MdlOccurrence("GT_B", 1, (0, 1), raw_bits=20.0, encoded_bits=5.0),
        MdlOccurrence("GT_C", 2, (3, 4), raw_bits=18.0, encoded_bits=5.0),
    )
    selected = select_nonoverlapping_mdl(occurrences, local_improvement=True)
    assert {item.dictionary_type_id for item in selected} == {"GT_A", "GT_C"}
    assert all(item.mdl_gain > 0 for item in selected)


def test_weighted_wishart_can_filter_by_cluster_occurrence_mass() -> None:
    points = np.array([[0.0], [0.1], [5.0], [5.1]])
    nn = NearestNeighbors(n_neighbors=2).fit(points)
    distances, indices = nn.kneighbors(points)
    result = wishart_cluster(
        indices[:, 1:],
        distances[:, 1:],
        significance=0.2,
        min_cluster_size=2,
        sample_weights=np.array([10.0, 10.0, 1.0, 1.0]),
        min_cluster_mass=5.0,
    )
    assert result.cluster_count == 1
    assert result.cluster_masses == {0: 20.0}


def test_graph_dictionary_returns_prototype_to_occurrence_mapping() -> None:
    adjacency = _layer([(0, 1), (1, 0), (1, 2), (2, 1)])
    original = EgoCandidate(
        0,
        np.array([10, 11, 12]),
        adjacency,
        {"IsA": adjacency},
        node_types=(None, None, None),
    )
    permutation = np.array([2, 1, 0])
    permuted = EgoCandidate(
        1,
        np.array([20, 21, 22]),
        adjacency[permutation][:, permutation].tocsr(),
        {"IsA": adjacency[permutation][:, permutation].tocsr()},
        node_types=(None, None, None),
    )
    dictionary = GraphDictionary(boundary_sensitive=False)
    graph_type = dictionary.resolve_or_create(original, level=0)
    matched = dictionary.match_with_mapping(permuted)
    assert matched is not None
    assert matched.graph_type.type_id == graph_type.type_id
    fine_nodes = tuple(
        int(permuted.nodes[index]) for index in matched.prototype_to_candidate
    )
    assert set(fine_nodes) == {20, 21, 22}
    assert len(fine_nodes) == 3
