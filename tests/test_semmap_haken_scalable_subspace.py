from __future__ import annotations

import numpy as np

from semmap_haken.metrics import slow_subspace_comparison


def _legacy_projector_distance(fine_basis: np.ndarray, coarse_basis: np.ndarray) -> float:
    fine_q, _ = np.linalg.qr(np.asarray(fine_basis, dtype=np.float64))
    coarse_q, _ = np.linalg.qr(np.asarray(coarse_basis, dtype=np.float64))
    rank = min(fine_q.shape[1], coarse_q.shape[1])
    fine_q, coarse_q = fine_q[:, :rank], coarse_q[:, :rank]
    fine_projection = fine_q @ fine_q.T
    coarse_projection = coarse_q @ coarse_q.T
    return float(np.linalg.norm(fine_projection - coarse_projection, ord="fro") / np.sqrt(2.0))


def test_scalable_subspace_distance_matches_legacy_projector_formula() -> None:
    rng = np.random.default_rng(1729)
    for node_count, fine_rank, coarse_rank in ((8, 2, 2), (17, 4, 3), (32, 5, 5)):
        fine = rng.normal(size=(node_count, fine_rank))
        coarse = rng.normal(size=(node_count, coarse_rank))
        expected = _legacy_projector_distance(fine, coarse)
        actual = slow_subspace_comparison(fine, coarse)
        assert np.isclose(actual.projection_distance, expected, rtol=1e-12, atol=1e-12)
        assert actual.compared_rank == min(fine_rank, coarse_rank)


def test_scalable_subspace_distance_handles_100k_without_n_by_n_projection() -> None:
    rng = np.random.default_rng(1729)
    fine = rng.normal(size=(100_000, 3))
    coarse = rng.normal(size=(100_000, 3))
    result = slow_subspace_comparison(fine, coarse)
    assert np.isfinite(result.projection_distance)
    assert result.compared_rank == 3
    assert result.principal_angles.shape == (3,)
