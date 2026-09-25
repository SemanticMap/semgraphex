"""Connected sampling and batched, resumable FGW regression coverage."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from semmap_haken.graph_build import _select_nodes
from semmap_haken.wishart_config import WishartOptions
from semmap_haken.wishart_metrics import EgoCandidate
from semmap_haken.wishart_fgw_accel import (
    _torch_fgw_batch, accelerated_fgw_neighbors,
)


def _candidate(center, edges, size=3, relation="IsA"):
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    graph = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(size, size)
    )
    return EgoCandidate(center, np.arange(size), graph, {relation: graph})


def test_connected_sample_preserves_connectivity_after_node_cap():
    # The lexicographically first four nodes are disconnected; BFS must not
    # produce the disconnected induced prefix.
    names = ["a", "b", "c", "d", "e", "f", "g"]
    links = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3)]
    graph = _candidate(0, links, size=7).adjacency
    sample, ids = _select_nodes(graph, names, component="largest_connected_sample", max_nodes=4)
    ncomponents, _ = connected_components(sample, directed=False)
    assert sample.shape == (4, 4)
    assert len(ids) == 4
    assert ncomponents == 1


def test_100k_gpu_configuration_contract():
    from pathlib import Path
    import yaml
    path = Path(__file__).resolve().parents[1] / "configs" / "wishart_conceptnet_fgw_100k_cuda_colab.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["dataset"]["max_nodes"] == 100000
    assert doc["dataset"]["component"] == "largest_connected_sample"
    assert doc["wishart"]["candidate_limit"] == 25000
    assert doc["wishart"]["transport_rank"] == 16
    assert doc["wishart"]["k_neighbors"] == 5
    assert doc["wishart"]["fgw_alpha"] == 0.5
    assert doc["colab"]["device"] == "cuda"
    WishartOptions.from_mapping(doc["wishart"])


def test_torch_fgw_batch_cpu_smoke_and_marginals():
    pytest.importorskip("torch")
    C = np.array([[0, 1], [1, 0]], dtype=float)
    F = np.array([[1, 0], [0, 1]], dtype=float)
    result = _torch_fgw_batch(
        [(C, F)], [(C, F)], alpha=0.5, epsilon=0.08,
        outer_iterations=12, sinkhorn_iterations=60, device="cpu",
    )
    assert result.shape == (1,)
    assert np.isfinite(result).all()
    assert (result >= 0).all()


def test_fgw_shortlist_and_pair_cache_resume(tmp_path):
    pytest.importorskip("torch")
    candidates = [
        _candidate(0, [(0, 1), (1, 2)]),
        _candidate(1, [(0, 1), (1, 2)]),
        _candidate(2, [(0, 1), (0, 2)]),
        _candidate(3, [(0, 1), (1, 2)], relation="RelatedTo"),
    ]
    completed = []
    kwargs = dict(
        k=1, rank=3, max_candidates=4, alpha=0.5,
        exact_types=2, shortlist=2, epsilon=0.08,
        outer_iterations=10, sinkhorn_iterations=60,
        pair_batch_size=2, cache_pairs=2, cpu_workers=2,
        device="cpu", type_ids=["A", "B", "C", "D"],
        cache_dir=tmp_path, checkpoint_hook=completed.append,
    )
    first = accelerated_fgw_neighbors(candidates, **kwargs)
    assert first.indices.shape == (4, 1)
    assert np.isfinite(first.distances).all()
    assert first.metadata["fgw_pair_exhaustive"] is False
    assert first.metadata["fgw_pair_count"] < first.metadata["full_pair_count"]
    assert completed
    completed.clear()
    second = accelerated_fgw_neighbors(candidates, **kwargs)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.distances, second.distances)
    assert not completed  # resumed cached blocks, no GPU/CPU recomputation


def test_fgw_cuda_smoke_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on CI host")
    C = np.array([[0, 1], [1, 0]], dtype=float)
    F = np.array([[1, 0], [0, 1]], dtype=float)
    result = _torch_fgw_batch(
        [(C, F)], [(C, F)], alpha=0.5, epsilon=0.08,
        outer_iterations=10, sinkhorn_iterations=60, device="cuda",
    )
    torch.cuda.synchronize()
    assert np.isfinite(result).all()
